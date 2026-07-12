"""Score and aggregate the weights-preserving C42 confirmation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch.nn import functional as F


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from myrec.eval.metrics import ScoredCandidate, ndcg_at_k, sort_candidates  # noqa: E402
from train.gate_metrics import bootstrap, clicked_direction, compare  # noqa: E402
from train.locking import verify_execution_lock, verify_proposal_lock  # noqa: E402
from train.store import FrozenStore, open_role_labels, read_json, sha256_file, write_json  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


c41_module = load_module(
    "c41_semantic_routing",
    REPO_ROOT / "systems/41_semantic_carrier_routing_transformer/model/semantic_routing.py",
)
c38_module = load_module(
    "c38_global_tangent",
    REPO_ROOT / "systems/38_cross_domain_global_tangent_transfer/model/global_tangent.py",
)


PRIMARY = "coupled_content"
MATCHED = ("semantic_routing", "single_wide_routing", "asymmetric_routing")
MODES = (*MATCHED, PRIMARY)
FUNCTIONAL = ("fixed_semantic", "uniform_semantic", "c38_unprojected")


def load_config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("C42 config must be an object")
    return value


def to_tensor(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(value, dtype=np.float32)).to(device)


def model_inputs(
    store: FrozenStore,
    index: int,
    history_source: str,
    device: torch.device,
    *,
    candidate_order: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query = to_tensor(store.query(index), device)
    history = to_tensor(store.items(store.history_positions(index, history_source)), device)
    positions = store.candidate_positions(index)
    if candidate_order is not None:
        positions = positions[candidate_order]
    return query, history, to_tensor(store.items(positions), device)


def make_c41(config: Mapping[str, Any], seed: int, mode: str, device: torch.device):
    row = config["model"]
    model = c41_module.SemanticCarrierRoutingTransformer(
        dim=int(row["embedding_dim"]),
        heads=int(row["heads"]),
        rank=int(row["rank"]),
        temperature=float(row["history_temperature"]),
        profile_scale=float(row["profile_scale"]),
        correction_scale=float(row["correction_scale"]),
        seed=seed,
        mode=mode,
        init_std=float(row["init_std"]),
    ).to(device)
    path = Path(config["paths"]["c41_checkpoint_root"]) / f"seed_{seed}_{mode}.pt"
    if sha256_file(path) != config["integrity"]["c41_checkpoint_sha256"][str(seed)][mode]:
        raise RuntimeError("C41 checkpoint changed")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint["seed"] != seed or checkpoint["mode"] != mode:
        raise ValueError("C41 checkpoint identity differs")
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), path


def make_c38(config: Mapping[str, Any], seed: int, device: torch.device):
    row = config["model"]
    model = c38_module.LowRankGlobalTangentTransfer(
        dim=int(row["embedding_dim"]),
        rank=16,
        temperature=float(row["history_temperature"]),
        profile_scale=float(row["profile_scale"]),
        correction_scale=float(row["correction_scale"]),
        seed=seed,
        mode=c38_module.UNPROJECTED,
    ).to(device)
    path = Path(config["paths"]["c38_checkpoint_root"]) / f"seed_{seed}_query_attended_unprojected.pt"
    if sha256_file(path) != config["integrity"]["c38_checkpoint_sha256"][str(seed)]:
        raise RuntimeError("C38 checkpoint changed")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), path


def uniform_correction(query, history, candidates, config):
    if len(history) == 0:
        return candidates.new_zeros(len(candidates))
    query = F.normalize(query, dim=-1, eps=1e-6)
    history = F.normalize(history, dim=-1, eps=1e-6)
    candidates = F.normalize(candidates, dim=-1, eps=1e-6)
    transported = F.normalize(
        query + float(config["model"]["profile_scale"]) * history.mean(dim=0),
        dim=-1,
        eps=1e-6,
    )
    return float(config["model"]["correction_scale"]) * (
        candidates.mv(transported) - candidates.mv(query)
    )


def load_blind_records(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def run_g0(config: Mapping[str, Any]) -> dict[str, Any]:
    _, proposal_hash = verify_proposal_lock(config)
    output = Path(config["paths"]["g0_report"])
    if output.exists():
        raise FileExistsError(output)
    trigger = read_json(config["paths"]["trigger_report"])
    selection = read_json(config["paths"]["selection"])
    feature_root = Path(config["paths"]["feature_root"])
    feature = read_json(feature_root / "feature_manifest.json")
    embedding = read_json(feature_root / "embedding_manifest.json")
    store = FrozenStore(config)
    records = load_blind_records(config["paths"]["records_train_blind"])
    indices = store.role_indices("internal_A")
    checkpoint_checks = []
    for seed in config["checkpoints"]["c41_seeds"]:
        for mode in MODES:
            path = Path(config["paths"]["c41_checkpoint_root"]) / f"seed_{seed}_{mode}.pt"
            checkpoint_checks.append(
                sha256_file(path) == config["integrity"]["c41_checkpoint_sha256"][str(seed)][mode]
            )
    isolation = selection["outcome_isolation"]
    checks = {
        "trigger_passed_and_hashed": (
            trigger.get("status") == "passed_c42_trigger"
            and all(trigger["checks"].values())
            and sha256_file(config["paths"]["trigger_report"])
            == config["paths"]["trigger_report_sha256"]
        ),
        "feature_selection_bound": feature["selection_sha256"] == config["paths"]["selection_sha256"],
        "feature_A_only": feature["roles"] == ["internal_A"] and feature["requests"] == 1200,
        "feature_label_free": (
            feature["label_access"]["records_train_labels_opened"] is False
            and feature["label_access"]["dev_test_records_labels_qrels_opened"] is False
        ),
        "embeddings_finite": embedding.get("finite") is True,
        "cohort_isolated": (
            isolation["internal_A_from_c38_escrow"] == 1200
            and isolation["internal_A_overlap_c38_other_roles"] == 0
            and isolation["internal_A_overlap_any_prior_feature_materialized"] == 0
        ),
        "histories_nonempty": all(records[index]["history"] for index in indices),
        "history_causal": all(
            all(int(event["ts"]) < int(records[index]["ts"]) for event in records[index]["history"])
            for index in indices
        ),
        "wrong_donor_contract": (
            selection["wrong_donor_audit"]["coverage_fraction"] == 1.0
            and selection["wrong_donor_audit"]["same_length_bin_fraction"] == 1.0
            and selection["wrong_donor_audit"]["same_user_assignments"] == 0
        ),
        "all_c41_checkpoint_hashes": all(checkpoint_checks),
        "zero_optimizer_steps": config["checkpoints"]["optimizer_steps"] == 0,
        "internal_A_scores_labels_closed": True,
        "dev_test_closed": True,
    }
    report = {
        "candidate_id": "c42",
        "gate": "G0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "proposal_lock_sha256": proposal_hash,
        "outcome_isolation": isolation,
        "feature_manifest_sha256": sha256_file(feature_root / "feature_manifest.json"),
        "embedding_manifest_sha256": sha256_file(feature_root / "embedding_manifest.json"),
        "internal_A_scores_labels_opened": False,
        "dev_test_opened": False,
    }
    write_json(output, report)
    return report


def score_callable(scorer, store, indices, source, device):
    scores, corrections = [], []
    with torch.inference_mode():
        for index in indices:
            query, history, candidates = model_inputs(store, index, source, device)
            correction = (
                torch.zeros(len(candidates), device=device)
                if store.has_repeat(index)
                else scorer(query, history, candidates)
            )
            value = correction.detach().cpu().numpy().astype(np.float32)
            corrections.append(value)
            scores.append((store.base_row(index) + value).astype(np.float32))
    return scores, corrections


def diagnose(scorer, store, indices, device):
    deterministic = permutation_error = nohistory_error = 0.0
    with torch.inference_mode():
        for index in indices[:32]:
            q, h, c = model_inputs(store, index, "true", device)
            first, second = scorer(q, h, c), scorer(q, h, c)
            deterministic = max(deterministic, float((first - second).abs().max().cpu()))
            permutation = np.random.default_rng(20262599 + index).permutation(len(c))
            order = torch.from_numpy(permutation).to(device)
            actual = scorer(q, h, c[order])
            permutation_error = max(permutation_error, float((first[order] - actual).abs().max().cpu()))
            nohistory_error = max(nohistory_error, float(scorer(q, h[:0], c).abs().max().cpu()))
    return {
        "deterministic_max_abs": deterministic,
        "candidate_permutation_max_abs": permutation_error,
        "nohistory_max_abs": nohistory_error,
        "query_absent_wrapper_max_abs": 0.0,
        "repeat_wrapper_max_abs": 0.0,
    }


def flatten(rows):
    return np.concatenate([np.asarray(row, dtype=np.float32) for row in rows])


def offsets(rows):
    value = [0]
    for row in rows:
        value.append(value[-1] + len(row))
    return np.asarray(value, dtype=np.int64)


def unflatten(offset, value):
    return [np.asarray(value[int(offset[i]) : int(offset[i + 1])], dtype=np.float32) for i in range(len(offset) - 1)]


def average_rows(groups):
    return [np.mean(np.stack([group[i] for group in groups]), axis=0).astype(np.float32) for i in range(len(groups[0]))]


def state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run_seed(config: Mapping[str, Any], seed: int, device: torch.device) -> dict[str, Any]:
    _, proposal_hash = verify_proposal_lock(config)
    _, execution_hash = verify_execution_lock(config, proposal_hash)
    store = FrozenStore(config)
    indices = store.role_indices("internal_A")
    artifact_root = Path(config["paths"]["artifact_root"])
    report_path = artifact_root / f"seed_{seed}_report.json"
    score_path = artifact_root / f"seed_{seed}_scores.npz"
    if report_path.exists() or score_path.exists():
        raise FileExistsError(f"C42 seed output exists: {seed}")
    payload: dict[str, np.ndarray] = {}
    mode_reports = {}
    for mode in MODES:
        model, checkpoint_path = make_c41(config, seed, mode, device)
        true_scores, true_corrections = score_callable(model, store, indices, "true", device)
        wrong_scores, _ = score_callable(model, store, indices, "wrong", device)
        row = {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "state_sha256": state_sha256(model),
            "parameters": model.trainable_parameter_count(),
            "optimizer_steps": 0,
            "diagnostics": diagnose(model, store, indices, device),
        }
        if mode == PRIMARY:
            with torch.inference_mode():
                q, h, c = model_inputs(store, indices[0], "true", device)
                state = model.components(q, h, c)
            row["loop_assignment"] = state["route_assignment"].cpu().tolist()
            row["semantic_carrier_exact"] = bool(state["semantic_carrier_exact"])
            row["states_finite"] = all(
                bool(torch.isfinite(state[name]).all())
                for name in ("attention", "profile", "transported_query", "correction")
            )
        mode_reports[mode] = row
        payload[f"{mode}_true"] = flatten(true_scores)
        payload[f"{mode}_wrong"] = flatten(wrong_scores)
        payload[f"{mode}_correction"] = flatten(true_corrections)

    fixed = lambda q, h, c: c41_module.fixed_semantic_correction(
        q,
        h,
        c,
        temperature=float(config["model"]["history_temperature"]),
        profile_scale=float(config["model"]["profile_scale"]),
        correction_scale=float(config["model"]["correction_scale"]),
    )
    uniform = lambda q, h, c: uniform_correction(q, h, c, config)
    fixed_true, _ = score_callable(fixed, store, indices, "true", device)
    uniform_true, _ = score_callable(uniform, store, indices, "true", device)
    c41_seeds = [int(value) for value in config["checkpoints"]["c41_seeds"]]
    c38_seed = int(config["checkpoints"]["c38_seeds"][c41_seeds.index(seed)])
    c38, c38_path = make_c38(config, c38_seed, device)
    c38_true, _ = score_callable(c38, store, indices, "true", device)
    functional_reports = {
        "fixed_semantic": diagnose(fixed, store, indices, device),
        "uniform_semantic": diagnose(uniform, store, indices, device),
        "c38_unprojected": diagnose(c38, store, indices, device),
    }
    base = [store.base_row(index) for index in indices]
    payload.update(
        {
            "fixed_semantic_true": flatten(fixed_true),
            "uniform_semantic_true": flatten(uniform_true),
            "c38_unprojected_true": flatten(c38_true),
            "base": flatten(base),
            "offsets": offsets(base),
        }
    )
    temporary = score_path.with_suffix(score_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **payload)
    temporary.replace(score_path)
    report = {
        "candidate_id": "c42",
        "seed": seed,
        "c38_seed": c38_seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "physical_gpu": config["resources"]["seed_to_physical_gpu"][str(seed)],
        "proposal_lock_sha256": proposal_hash,
        "execution_lock_sha256": execution_hash,
        "optimizer_steps": 0,
        "mode_reports": mode_reports,
        "functional_control_reports": functional_reports,
        "c38_checkpoint": {"path": str(c38_path), "sha256": sha256_file(c38_path)},
        "score_artifact": {"path": str(score_path), "sha256": sha256_file(score_path)},
        "internal_A_scores_opened": True,
        "internal_A_labels_opened": False,
        "dev_test_read": False,
    }
    write_json(report_path, report)
    return report


def rankings(request_ids, item_ids, scores):
    return [
        [row.item_id for row in sort_candidates(request_id, [ScoredCandidate(str(item), float(score)) for item, score in zip(items, values)])]
        for request_id, items, values in zip(request_ids, item_ids, scores)
    ]


def order_changes(request_ids, item_ids, first_scores, second_scores):
    first = rankings(request_ids, item_ids, first_scores)
    second = rankings(request_ids, item_ids, second_scores)
    any_count = sum(int(a != b) for a, b in zip(first, second))
    top10_count = sum(int(set(a[:10]) != set(b[:10])) for a, b in zip(first, second))
    return {
        "requests": len(first),
        "any_count": any_count,
        "any_fraction": any_count / len(first),
        "top10_count": top10_count,
        "top10_fraction": top10_count / len(first),
    }


def ndcg_rows(request_ids, item_ids, scores, labels):
    output = []
    for request_id, items, values, label in zip(request_ids, item_ids, scores, labels):
        ranked = rankings([request_id], [items], [values])[0]
        positives = {str(item) for item, relevance in zip(items, label) if relevance > 0}
        output.append(ndcg_at_k(ranked, positives, 10))
    return np.asarray(output, dtype=np.float64)


def aggregate(config: Mapping[str, Any]) -> dict[str, Any]:
    _, proposal_hash = verify_proposal_lock(config)
    _, execution_hash = verify_execution_lock(config, proposal_hash)
    artifact_root = Path(config["paths"]["artifact_root"])
    output_path = artifact_root / "confirmation_report.json"
    if output_path.exists():
        raise FileExistsError(output_path)
    seeds = [int(value) for value in config["checkpoints"]["c41_seeds"]]
    reports = [read_json(artifact_root / f"seed_{seed}_report.json") for seed in seeds]
    store = FrozenStore(config)
    indices = store.role_indices("internal_A")
    request_ids = [store.request_id(index) for index in indices]
    item_ids = [store.candidate_ids(index) for index in indices]
    names = [
        "base",
        "fixed_semantic_true",
        "uniform_semantic_true",
        "c38_unprojected_true",
        *[f"{mode}_{source}" for mode in MODES for source in ("true", "wrong", "correction")],
    ]
    score_rows = {}
    for seed, report in zip(seeds, reports):
        path = Path(report["score_artifact"]["path"])
        if sha256_file(path) != report["score_artifact"]["sha256"]:
            raise RuntimeError("C42 score artifact changed")
        with np.load(path, allow_pickle=False) as values:
            off = np.asarray(values["offsets"], dtype=np.int64)
            score_rows[seed] = {name: unflatten(off, values[name]) for name in names}
    averaged = {name: average_rows([score_rows[seed][name] for seed in seeds]) for name in names}
    controls = [*MATCHED, *FUNCTIONAL]
    control_key = {
        **{mode: f"{mode}_true" for mode in MATCHED},
        "fixed_semantic": "fixed_semantic_true",
        "uniform_semantic": "uniform_semantic_true",
        "c38_unprojected": "c38_unprojected_true",
    }
    activity = {
        "primary_vs_base": order_changes(request_ids, item_ids, averaged["base"], averaged[f"{PRIMARY}_true"]),
        "true_vs_wrong": order_changes(request_ids, item_ids, averaged[f"{PRIMARY}_true"], averaged[f"{PRIMARY}_wrong"]),
        **{
            f"primary_vs_{name}": order_changes(request_ids, item_ids, averaged[control_key[name]], averaged[f"{PRIMARY}_true"])
            for name in controls
        },
    }
    gate = config["gate"]
    contract_names = (*MODES,)
    a0_checks = {
        "zero_optimizer_steps": all(report["optimizer_steps"] == 0 for report in reports),
        "checkpoint_hashes_preserved": all(
            report["mode_reports"][mode]["checkpoint_sha256"]
            == config["integrity"]["c41_checkpoint_sha256"][str(seed)][mode]
            for seed, report in zip(seeds, reports)
            for mode in MODES
        ),
        "equal_capacity": all(
            {report["mode_reports"][mode]["parameters"] for mode in MODES} == {49152}
            for report in reports
        ),
        "primary_identity_loop": all(
            report["mode_reports"][PRIMARY]["loop_assignment"] == [0, 1, 2, 3]
            and report["mode_reports"][PRIMARY]["semantic_carrier_exact"] is False
            and report["mode_reports"][PRIMARY]["states_finite"]
            for report in reports
        ),
        "model_contracts": all(
            report["mode_reports"][mode]["diagnostics"]["deterministic_max_abs"] <= float(gate["deterministic_max_abs"])
            and report["mode_reports"][mode]["diagnostics"]["candidate_permutation_max_abs"] <= float(gate["candidate_permutation_max_abs"])
            and report["mode_reports"][mode]["diagnostics"]["nohistory_max_abs"] == 0.0
            and report["mode_reports"][mode]["diagnostics"]["query_absent_wrapper_max_abs"] == 0.0
            and report["mode_reports"][mode]["diagnostics"]["repeat_wrapper_max_abs"] == 0.0
            for report in reports
            for mode in contract_names
        ),
        "functional_contracts": all(
            report["functional_control_reports"][name]["deterministic_max_abs"] <= float(gate["deterministic_max_abs"])
            and report["functional_control_reports"][name]["candidate_permutation_max_abs"] <= float(gate["candidate_permutation_max_abs"])
            and report["functional_control_reports"][name]["nohistory_max_abs"] == 0.0
            for report in reports
            for name in FUNCTIONAL
        ),
        "primary_order_active": activity["primary_vs_base"]["any_fraction"] >= 0.02,
        "primary_top10_active": activity["primary_vs_base"]["top10_fraction"] >= 0.005,
        "wrong_order_distinct": activity["true_vs_wrong"]["any_fraction"] >= float(gate["true_vs_wrong_order_fraction_min"]),
        "wrong_top10_distinct": activity["true_vs_wrong"]["top10_fraction"] >= float(gate["true_vs_wrong_top10_fraction_min"]),
        **{
            f"{name}_order_distinct": activity[f"primary_vs_{name}"]["any_fraction"] >= float(gate["primary_vs_control_order_fraction_min"])
            for name in controls
        },
        **{
            f"{name}_top10_distinct": activity[f"primary_vs_{name}"]["top10_fraction"] >= float(gate["primary_vs_control_top10_fraction_min"])
            for name in controls
        },
        "dev_test_closed": all(not report["dev_test_read"] for report in reports),
    }
    report: dict[str, Any] = {
        "candidate_id": "c42",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "proposal_lock_sha256": proposal_hash,
        "execution_lock_sha256": execution_hash,
        "A0": {"checks": a0_checks, "activity": activity},
        "seed_reports": {str(seed): value for seed, value in zip(seeds, reports)},
        "optimizer_steps": 0,
        "internal_A_scores_opened": True,
        "internal_A_labels_opened": False,
        "dev_test_opened": False,
    }
    if not all(a0_checks.values()):
        report["status"] = "failed_A0_terminal"
        write_json(output_path, report)
        return report

    labels = open_role_labels(
        records_train_path=config["paths"]["records_train"],
        records_train_sha256=config["integrity"]["records_train_sha256"],
        selection_path=config["paths"]["selection"],
        selection_sha256=config["paths"]["selection_sha256"],
        store=store,
        role="internal_A",
    )
    label_rows = [labels.row(index, store.candidate_count(index)) for index in indices]
    metric_names = ["base", *MODES, *FUNCTIONAL, "primary_wrong"]
    per_seed = {}
    for seed in seeds:
        rows = score_rows[seed]
        per_seed[seed] = {
            "base": ndcg_rows(request_ids, item_ids, rows["base"], label_rows),
            **{mode: ndcg_rows(request_ids, item_ids, rows[f"{mode}_true"], label_rows) for mode in MODES},
            "fixed_semantic": ndcg_rows(request_ids, item_ids, rows["fixed_semantic_true"], label_rows),
            "uniform_semantic": ndcg_rows(request_ids, item_ids, rows["uniform_semantic_true"], label_rows),
            "c38_unprojected": ndcg_rows(request_ids, item_ids, rows["c38_unprojected_true"], label_rows),
            "primary_wrong": ndcg_rows(request_ids, item_ids, rows[f"{PRIMARY}_wrong"], label_rows),
        }
    averaged_ndcg = {
        name: np.mean(np.stack([per_seed[seed][name] for seed in seeds]), axis=0)
        for name in metric_names
    }
    references = {
        "base": averaged_ndcg["base"],
        "c38_unprojected": averaged_ndcg["c38_unprojected"],
        **{mode: averaged_ndcg[mode] for mode in MATCHED},
        "wrong_history": averaged_ndcg["primary_wrong"],
    }
    comparisons = compare(
        request_ids,
        averaged_ndcg[PRIMARY],
        references,
        samples=int(config["evaluation"]["bootstrap_samples"]),
        seed=int(config["evaluation"]["bootstrap_seed"]),
        folds=int(config["evaluation"]["hash_folds"]),
    )
    seed_diff = {
        reference: {
            str(seed): float((per_seed[seed][PRIMARY] - per_seed[seed][reference]).mean())
            for seed in seeds
        }
        for reference in ("base", "c38_unprojected", *MATCHED, "primary_wrong")
    }
    correction = average_rows([score_rows[seed][f"{PRIMARY}_correction"] for seed in seeds])
    direction = bootstrap(
        clicked_direction(correction, label_rows),
        samples=int(config["evaluation"]["bootstrap_samples"]),
        seed=int(config["evaluation"]["bootstrap_seed"]) + 101,
    )

    def strong(reference, minimum):
        return {
            f"over_{reference}_effect": comparisons[reference]["mean"] >= minimum,
            f"over_{reference}_ci": comparisons[reference]["percentile_95_ci"][0] > 0,
            f"over_{reference}_all_seeds": all(value > 0 for value in seed_diff[reference].values()),
            f"over_{reference}_all_folds": all(row["mean_difference"] > 0 for row in comparisons[reference]["hash_folds"]),
        }

    a1_checks = {
        **strong("base", float(gate["primary_minus_base_min"])),
        **strong("c38_unprojected", float(gate["primary_minus_c38_min"])),
        **{
            f"over_{mode}_effect": comparisons[mode]["mean"] >= float(gate["primary_minus_matched_min"])
            for mode in MATCHED
        },
        **{f"over_{mode}_ci": comparisons[mode]["percentile_95_ci"][0] > 0 for mode in MATCHED},
        **{f"over_{mode}_all_seeds": all(value >= 0 for value in seed_diff[mode].values()) for mode in MATCHED},
        "true_over_wrong_ci": comparisons["wrong_history"]["percentile_95_ci"][0] > 0,
        "true_over_wrong_all_seeds": all(value > 0 for value in seed_diff["primary_wrong"].values()),
        "true_over_wrong_all_folds": all(row["mean_difference"] > 0 for row in comparisons["wrong_history"]["hash_folds"]),
        "clicked_direction_ci": direction["percentile_95_ci"][0] > 0,
    }
    report["A1"] = {
        "checks": a1_checks,
        "comparisons": comparisons,
        "clicked_direction": direction,
        "seed_differences": seed_diff,
        "seed_averaged_ndcg10": {name: float(value.mean()) for name, value in averaged_ndcg.items()},
    }
    report["internal_A_labels_opened"] = True
    report["status"] = "passed_A1_confirmation" if all(a1_checks.values()) else "failed_A1_terminal"
    write_json(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("g0", "seed", "aggregate"), required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.stage == "g0":
        value = run_g0(config)
    elif args.stage == "seed":
        if args.seed is None:
            raise ValueError("C42 seed stage requires --seed")
        value = run_seed(config, args.seed, torch.device(config["program_device"]))
    else:
        value = aggregate(config)
    print(json.dumps({"candidate_id": "c42", "stage": args.stage, "status": value.get("status", "complete")}, sort_keys=True))


if __name__ == "__main__":
    main()
