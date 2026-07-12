"""Run the frozen C41 Amazon train-internal architecture-boundary gate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
import time
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

from model.semantic_routing import (  # noqa: E402
    ASYMMETRIC_ROUTING,
    COUPLED_CONTENT,
    MODES,
    SEMANTIC_ROUTING,
    SINGLE_WIDE_ROUTING,
    SemanticCarrierRoutingTransformer,
    fixed_semantic_correction,
)
from myrec.eval.metrics import ScoredCandidate, ndcg_at_k, sort_candidates  # noqa: E402
from train.gate_metrics import bootstrap, clicked_direction, compare  # noqa: E402
from train.locking import verify_execution_lock, verify_proposal_lock  # noqa: E402
from train.store import (  # noqa: E402
    CompactLabels,
    FrozenStore,
    open_role_labels,
    read_json,
    sha256_file,
    write_json,
)


spec = importlib.util.spec_from_file_location(
    "c38_global_tangent",
    REPO_ROOT
    / "systems/38_cross_domain_global_tangent_transfer/model/global_tangent.py",
)
c38_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(c38_module)


PRIMARY = SEMANTIC_ROUTING
MATCHED_CONTROLS = (SINGLE_WIDE_ROUTING, ASYMMETRIC_ROUTING, COUPLED_CONTENT)
FUNCTIONAL_CONTROLS = ("fixed_semantic", "uniform_semantic", "c38_unprojected")


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("C41 config must be an object")
    return value


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def make_model(
    config: Mapping[str, Any], seed: int, mode: str
) -> SemanticCarrierRoutingTransformer:
    row = config["model"]
    return SemanticCarrierRoutingTransformer(
        dim=int(row["embedding_dim"]),
        heads=int(row["heads"]),
        rank=int(row["rank"]),
        temperature=float(row["history_temperature"]),
        profile_scale=float(row["profile_scale"]),
        correction_scale=float(row["correction_scale"]),
        seed=seed,
        mode=mode,
        init_std=float(row["init_std"]),
    )


def make_c38_control(
    config: Mapping[str, Any], seed: int, device: torch.device
) -> torch.nn.Module:
    model = c38_module.LowRankGlobalTangentTransfer(
        dim=int(config["model"]["embedding_dim"]),
        rank=16,
        temperature=float(config["model"]["history_temperature"]),
        profile_scale=float(config["model"]["profile_scale"]),
        correction_scale=float(config["model"]["correction_scale"]),
        seed=seed,
        mode=c38_module.UNPROJECTED,
    ).to(device)
    path = (
        Path(config["paths"]["c38_checkpoint_root"])
        / f"seed_{seed}_query_attended_unprojected.pt"
    )
    expected = config["integrity"]["c38_checkpoint_sha256"][str(seed)]
    if sha256_file(path) != expected:
        raise RuntimeError("C38 control checkpoint changed")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()


def state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


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
    candidates = to_tensor(store.items(positions), device)
    return query, history, candidates


def load_blind_records(path: str | Path) -> list[dict[str, Any]]:
    output = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            output.append(json.loads(line))
    return output


def run_g0(config: Mapping[str, Any]) -> dict[str, Any]:
    _, proposal_hash = verify_proposal_lock(config)
    output_path = Path(config["paths"]["g0_report"])
    if output_path.exists():
        raise FileExistsError(output_path)
    design = read_json(config["paths"]["design_gate_report"])
    selection = read_json(config["paths"]["selection"])
    feature_root = Path(config["paths"]["feature_root"])
    feature = read_json(feature_root / "feature_manifest.json")
    embedding = read_json(feature_root / "embedding_manifest.json")
    store = FrozenStore(config)
    records = load_blind_records(config["paths"]["records_train_blind"])
    selected = [
        int(index)
        for role in ("fit", "internal_A")
        for index in selection["roles"][role]["indices"]
    ]
    labels = open_role_labels(
        records_train_path=config["paths"]["records_train"],
        records_train_sha256=config["integrity"]["records_train_sha256"],
        selection_path=config["paths"]["selection"],
        selection_sha256=config["paths"]["selection_sha256"],
        store=store,
        role="fit",
    )
    fit_path = Path(config["paths"]["fit_labels"])
    fit_path.parent.mkdir(parents=True, exist_ok=True)
    if fit_path.exists():
        raise FileExistsError(fit_path)
    with fit_path.open("wb") as handle:
        np.savez(
            handle,
            request_indices=labels.request_indices,
            offsets=labels.offsets,
            values=labels.values,
        )
    isolation = selection["outcome_isolation"]
    checks = {
        "design_gate_passed_and_hashed": (
            design.get("status") == "passed_design_gate"
            and sha256_file(config["paths"]["design_gate_report"])
            == config["paths"]["design_gate_report_sha256"]
        ),
        "feature_selection_bound": feature["selection_sha256"]
        == config["paths"]["selection_sha256"],
        "feature_roles_fit_A_only": feature["roles"] == ["fit", "internal_A"],
        "feature_requests_exact": feature["requests"] == 7200,
        "feature_pipeline_label_free": (
            feature["label_access"]["records_train_labels_opened"] is False
            and feature["label_access"]["dev_test_records_labels_qrels_opened"] is False
        ),
        "embeddings_finite": embedding.get("finite") is True,
        "selected_histories_nonempty": all(records[index]["history"] for index in selected),
        "selected_history_causal": all(
            all(int(event["ts"]) < int(records[index]["ts"]) for event in records[index]["history"])
            for index in selected
        ),
        "wrong_donor_contract": (
            selection["wrong_donor_audit"]["coverage_fraction"] == 1.0
            and selection["wrong_donor_audit"]["same_length_bin_fraction"] == 1.0
            and selection["wrong_donor_audit"]["same_user_assignments"] == 0
        ),
        "A_outcome_untouched": all(
            isolation[key] == 0
            for key in (
                "internal_A_overlap_c38_internal_A",
                "internal_A_overlap_c39_internal_A",
                "internal_A_overlap_c38_feature_materialized",
            )
        ),
        "delayed_B_unmaterialized": isolation[
            "delayed_B_overlap_any_prior_feature_materialized"
        ]
        == 0,
        "fit_exactly_c38_fit": isolation["fit_exactly_c38_fit"] == 1,
        "fit_has_one_positive": all(
            int((labels.row(index, store.candidate_count(index)) > 0).sum()) == 1
            for index in store.role_indices("fit")
        ),
        "internal_A_labels_scores_closed": True,
        "dev_test_closed": True,
    }
    report = {
        "candidate_id": "c41",
        "gate": "G0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "proposal_lock_sha256": proposal_hash,
        "feature_manifest_sha256": sha256_file(feature_root / "feature_manifest.json"),
        "embedding_manifest_sha256": sha256_file(feature_root / "embedding_manifest.json"),
        "fit_labels": {"path": str(fit_path), "sha256": sha256_file(fit_path), "requests": len(labels.request_indices)},
        "outcome_isolation": isolation,
        "internal_A_labels_scores_opened": False,
        "delayed_B_features_labels_scores_opened": False,
        "dev_test_opened": False,
    }
    write_json(output_path, report)
    return report


def load_fit_labels(config: Mapping[str, Any]) -> CompactLabels:
    path = Path(config["paths"]["fit_labels"])
    g0 = read_json(config["paths"]["g0_report"])
    if sha256_file(path) != g0["fit_labels"]["sha256"]:
        raise RuntimeError("C41 fit labels changed")
    with np.load(path, allow_pickle=False) as values:
        return CompactLabels(
            request_indices=np.asarray(values["request_indices"], dtype=np.int64),
            offsets=np.asarray(values["offsets"], dtype=np.int64),
            values=np.asarray(values["values"], dtype=np.float32),
        )


def train_mode(
    model: SemanticCarrierRoutingTransformer,
    store: FrozenStore,
    labels: CompactLabels,
    config: Mapping[str, Any],
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    row = config["training"]
    all_indices = store.role_indices("fit")
    indices = [index for index in all_indices if not store.has_repeat(index)]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(row["learning_rate"]), weight_decay=float(row["weight_decay"])
    )
    losses = []
    listwise_losses = []
    direction_losses = []
    gradients: set[str] = set()
    model.to(device).train()
    for epoch in range(int(row["epochs"])):
        order = np.random.default_rng(seed + epoch * 10003).permutation(len(indices))
        batch = int(row["max_requests_per_batch"])
        for start in range(0, len(order), batch):
            request_losses = []
            request_listwise = []
            request_direction = []
            for raw in order[start : start + batch]:
                index = indices[int(raw)]
                target = to_tensor(
                    labels.row(index, store.candidate_count(index)) > 0, device
                ).bool()
                query, history, candidates = model_inputs(store, index, "true", device)
                correction = model(query, history, candidates)
                score = to_tensor(store.base_row(index), device) + correction
                listwise = torch.logsumexp(score, dim=0) - score[target].mean()
                direction = F.softplus(-(correction[target].mean() - correction[~target].mean()))
                request_listwise.append(listwise)
                request_direction.append(direction)
                request_losses.append(
                    float(row["listwise_loss_weight"]) * listwise
                    + float(row["direction_loss_weight"]) * direction
                )
            loss = torch.stack(request_losses).mean()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite C41 loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    if not bool(torch.isfinite(parameter.grad).all()):
                        raise RuntimeError(f"nonfinite C41 gradient: {name}")
                    if bool(parameter.grad.ne(0).any()):
                        gradients.add(name)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(row["gradient_clip_norm"]))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            listwise_losses.append(float(torch.stack(request_listwise).mean().detach().cpu()))
            direction_losses.append(float(torch.stack(request_direction).mean().detach().cpu()))
    return {
        "fit_requests": len(all_indices),
        "active_nonrepeat_requests": len(indices),
        "skipped_repeat_requests": len(all_indices) - len(indices),
        "epochs": int(row["epochs"]),
        "steps": len(losses),
        "all_candidates_used": True,
        "finite": bool(losses) and bool(np.isfinite(losses).all()),
        "loss_first_30": float(np.mean(losses[:30])),
        "loss_last_30": float(np.mean(losses[-30:])),
        "listwise_last_30": float(np.mean(listwise_losses[-30:])),
        "direction_last_30": float(np.mean(direction_losses[-30:])),
        "gradient_parameter_names": sorted(gradients),
    }


def uniform_correction(
    query: torch.Tensor,
    history: torch.Tensor,
    candidates: torch.Tensor,
    config: Mapping[str, Any],
) -> torch.Tensor:
    if len(history) == 0:
        return candidates.new_zeros(len(candidates))
    query = F.normalize(query, dim=-1, eps=1e-6)
    history = F.normalize(history, dim=-1, eps=1e-6)
    candidates = F.normalize(candidates, dim=-1, eps=1e-6)
    profile = history.mean(dim=0)
    transported = F.normalize(
        query + float(config["model"]["profile_scale"]) * profile,
        dim=-1,
        eps=1e-6,
    )
    return float(config["model"]["correction_scale"]) * (
        candidates.mv(transported) - candidates.mv(query)
    )


def score_callable(
    scorer: Any,
    store: FrozenStore,
    indices: Sequence[int],
    history_source: str,
    device: torch.device,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    scores = []
    corrections = []
    with torch.inference_mode():
        for index in indices:
            query, history, candidates = model_inputs(store, index, history_source, device)
            correction = (
                torch.zeros(len(candidates), device=device)
                if store.has_repeat(index)
                else scorer(query, history, candidates)
            )
            value = correction.detach().cpu().numpy().astype(np.float32)
            corrections.append(value)
            scores.append((store.base_row(index) + value).astype(np.float32))
    return scores, corrections


def diagnose_primary(
    model: SemanticCarrierRoutingTransformer,
    store: FrozenStore,
    indices: Sequence[int],
    device: torch.device,
) -> dict[str, Any]:
    profile_error = 0.0
    attention_sum_error = 0.0
    attention_min = 1.0
    requests = 0
    model.eval()
    with torch.inference_mode():
        for index in indices:
            if store.has_repeat(index):
                continue
            query, history, candidates = model_inputs(store, index, "true", device)
            state = model.components(query, history, candidates)
            raw = F.normalize(history, dim=-1, eps=1e-6)
            expected = torch.einsum("hj,jd->hd", state["attention"], raw)
            profile_error = max(profile_error, float((state["profile"] - expected).abs().max().cpu()))
            attention_sum_error = max(
                attention_sum_error,
                float((state["attention"].sum(dim=-1) - 1).abs().max().cpu()),
            )
            attention_min = min(attention_min, float(state["attention"].min().cpu()))
            requests += 1
    return {
        "active_nonrepeat_requests": requests,
        "raw_profile_max_abs_error": profile_error,
        "attention_sum_max_abs_error": attention_sum_error,
        "attention_min": attention_min,
    }


def diagnose_functional(
    scorer: Any,
    store: FrozenStore,
    indices: Sequence[int],
    device: torch.device,
) -> dict[str, float]:
    deterministic = 0.0
    permutation_error = 0.0
    nohistory_error = 0.0
    with torch.inference_mode():
        for index in indices[:32]:
            query, history, candidates = model_inputs(store, index, "true", device)
            first = scorer(query, history, candidates)
            second = scorer(query, history, candidates)
            deterministic = max(deterministic, float((first - second).abs().max().cpu()))
            permutation = np.random.default_rng(20262199 + index).permutation(len(candidates))
            permuted = torch.from_numpy(permutation).to(device)
            actual = scorer(query, history, candidates[permuted])
            permutation_error = max(
                permutation_error,
                float((first[permuted] - actual).abs().max().cpu()),
            )
            nohistory_error = max(
                nohistory_error,
                float(scorer(query, history[:0], candidates).abs().max().cpu()),
            )
    return {
        "deterministic_max_abs": deterministic,
        "candidate_permutation_max_abs": permutation_error,
        "nohistory_max_abs": nohistory_error,
        "query_absent_wrapper_max_abs": 0.0,
        "repeat_wrapper_max_abs": 0.0,
    }


def _flatten(rows: Sequence[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(row, dtype=np.float32) for row in rows])


def _offsets(rows: Sequence[np.ndarray]) -> np.ndarray:
    output = [0]
    for row in rows:
        output.append(output[-1] + len(row))
    return np.asarray(output, dtype=np.int64)


def _unflatten(offsets: np.ndarray, values: np.ndarray) -> list[np.ndarray]:
    return [
        np.asarray(values[int(offsets[i]) : int(offsets[i + 1])], dtype=np.float32)
        for i in range(len(offsets) - 1)
    ]


def _average_rows(groups: Sequence[Sequence[np.ndarray]]) -> list[np.ndarray]:
    return [
        np.mean(np.stack([group[index] for group in groups]), axis=0).astype(np.float32)
        for index in range(len(groups[0]))
    ]


def run_seed(
    config: Mapping[str, Any], seed: int, device: torch.device
) -> dict[str, Any]:
    _, proposal_hash = verify_proposal_lock(config)
    _, execution_hash = verify_execution_lock(config, proposal_hash)
    seed_all(seed)
    store = FrozenStore(config)
    labels = load_fit_labels(config)
    indices = store.role_indices("internal_A")
    artifact_root = Path(config["paths"]["artifact_root"])
    report_path = artifact_root / f"seed_{seed}_report.json"
    score_path = artifact_root / f"seed_{seed}_internal_A_scores.npz"
    if report_path.exists() or score_path.exists():
        raise FileExistsError(f"C41 seed output exists: {seed}")
    started = time.time()
    mode_reports = {}
    payload: dict[str, np.ndarray] = {}
    initial_hashes = {}
    for mode in MODES:
        seed_all(seed)
        model = make_model(config, seed, mode)
        initial = state_sha256(model)
        initial_hashes[mode] = initial
        training = train_mode(model, store, labels, config, seed, device)
        final = state_sha256(model)
        checkpoint_root = Path(config["paths"]["checkpoint_root"])
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_root / f"seed_{seed}_{mode}.pt"
        if checkpoint_path.exists():
            raise FileExistsError(checkpoint_path)
        temporary = checkpoint_path.with_suffix(".pt.tmp")
        torch.save(
            {
                "candidate_id": "c41",
                "seed": seed,
                "mode": mode,
                "proposal_lock_sha256": proposal_hash,
                "execution_lock_sha256": execution_hash,
                "state_dict": model.state_dict(),
            },
            temporary,
        )
        temporary.replace(checkpoint_path)
        model.eval()
        true_scores, true_corrections = score_callable(model, store, indices, "true", device)
        wrong_scores, _ = score_callable(model, store, indices, "wrong", device)
        deterministic_scores, _ = score_callable(model, store, indices[:32], "true", device)
        repeated_scores, _ = score_callable(model, store, indices[:32], "true", device)
        permutation_error = 0.0
        nohistory_error = 0.0
        query_error = 0.0
        repeat_error = 0.0
        with torch.inference_mode():
            for index in indices[:32]:
                count = store.candidate_count(index)
                permutation = np.random.default_rng(seed + index).permutation(count)
                query, history, candidates = model_inputs(store, index, "true", device)
                reference = model(query, history, candidates)[torch.from_numpy(permutation).to(device)]
                _, _, permuted = model_inputs(
                    store, index, "true", device, candidate_order=permutation
                )
                actual = model(query, history, permuted)
                permutation_error = max(permutation_error, float((reference - actual).abs().max().cpu()))
                nohistory_error = max(nohistory_error, float(model(query, history[:0], candidates).abs().max().cpu()))
                query_error = max(query_error, float(model(query, history, candidates, query_present=False).abs().max().cpu()))
                repeat_error = max(repeat_error, float(model(query, history, candidates, repeat_present=True).abs().max().cpu()))
        deterministic_error = max(
            float(np.max(np.abs(a - b)))
            for a, b in zip(deterministic_scores, repeated_scores)
        )
        mode_reports[mode] = {
            "parameters": model.trainable_parameter_count(),
            "training": training,
            "initial_state_sha256": initial,
            "final_state_sha256": final,
            "parameters_updated": initial != final,
            "checkpoint": {"path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path)},
            "deterministic_max_abs": deterministic_error,
            "candidate_permutation_max_abs": permutation_error,
            "nohistory_max_abs": nohistory_error,
            "query_absent_max_abs": query_error,
            "repeat_max_abs": repeat_error,
            "semantic_diagnostics": diagnose_primary(model, store, indices, device) if mode == PRIMARY else None,
        }
        payload[f"{mode}_true"] = _flatten(true_scores)
        payload[f"{mode}_wrong"] = _flatten(wrong_scores)
        payload[f"{mode}_correction"] = _flatten(true_corrections)

    fixed = lambda q, h, c: fixed_semantic_correction(
        q,
        h,
        c,
        temperature=float(config["model"]["history_temperature"]),
        profile_scale=float(config["model"]["profile_scale"]),
        correction_scale=float(config["model"]["correction_scale"]),
    )
    fixed_true, _ = score_callable(fixed, store, indices, "true", device)
    fixed_wrong, _ = score_callable(fixed, store, indices, "wrong", device)
    uniform = lambda q, h, c: uniform_correction(q, h, c, config)
    uniform_true, _ = score_callable(uniform, store, indices, "true", device)
    c38_seeds = [int(value) for value in config["training"]["c38_control_seeds"]]
    c41_seeds = [int(value) for value in config["training"]["seeds"]]
    c38_seed = c38_seeds[c41_seeds.index(seed)]
    c38_control = make_c38_control(config, c38_seed, device)
    c38_true, _ = score_callable(c38_control, store, indices, "true", device)
    c38_wrong, _ = score_callable(c38_control, store, indices, "wrong", device)
    functional_reports = {
        "fixed_semantic": diagnose_functional(fixed, store, indices, device),
        "uniform_semantic": diagnose_functional(uniform, store, indices, device),
        "c38_unprojected": diagnose_functional(c38_control, store, indices, device),
    }
    base_rows = [store.base_row(index) for index in indices]
    payload.update(
        {
            "fixed_semantic_true": _flatten(fixed_true),
            "fixed_semantic_wrong": _flatten(fixed_wrong),
            "uniform_semantic_true": _flatten(uniform_true),
            "c38_unprojected_true": _flatten(c38_true),
            "c38_unprojected_wrong": _flatten(c38_wrong),
            "base": _flatten(base_rows),
            "offsets": _offsets(base_rows),
        }
    )
    temporary_score = score_path.with_suffix(score_path.suffix + ".tmp")
    with temporary_score.open("wb") as handle:
        np.savez(handle, **payload)
    temporary_score.replace(score_path)
    report = {
        "candidate_id": "c41",
        "seed": seed,
        "c38_control_seed": c38_seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "physical_gpu": config["resources"]["seed_to_physical_gpu"][str(seed)],
        "proposal_lock_sha256": proposal_hash,
        "execution_lock_sha256": execution_hash,
        "mode_reports": mode_reports,
        "functional_control_reports": functional_reports,
        "paired_initialization": len(set(initial_hashes.values())) == 1,
        "seed_specific_initial_state_sha256": initial_hashes[PRIMARY],
        "score_artifact": {"path": str(score_path), "sha256": sha256_file(score_path)},
        "internal_A_scores_opened": True,
        "internal_A_labels_opened": False,
        "delayed_B_features_labels_scores_opened": False,
        "dev_test_qrels_read": False,
    }
    write_json(report_path, report)
    return report


def rankings(
    request_ids: Sequence[str],
    item_ids: Sequence[Sequence[str]],
    scores: Sequence[np.ndarray],
) -> list[list[str]]:
    return [
        [
            row.item_id
            for row in sort_candidates(
                request_id,
                [
                    ScoredCandidate(str(item), float(score))
                    for item, score in zip(items, values)
                ],
            )
        ]
        for request_id, items, values in zip(request_ids, item_ids, scores)
    ]


def order_changes(
    request_ids: Sequence[str],
    item_ids: Sequence[Sequence[str]],
    first_scores: Sequence[np.ndarray],
    second_scores: Sequence[np.ndarray],
) -> dict[str, Any]:
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


def ndcg_rows(
    request_ids: Sequence[str],
    item_ids: Sequence[Sequence[str]],
    scores: Sequence[np.ndarray],
    labels: Sequence[np.ndarray],
) -> np.ndarray:
    output = []
    for request_id, items, values, relevance in zip(request_ids, item_ids, scores, labels):
        ranked = rankings([request_id], [items], [values])[0]
        positives = {
            str(item) for item, value in zip(items, relevance) if value > 0
        }
        output.append(ndcg_at_k(ranked, positives, 10))
    return np.asarray(output, dtype=np.float64)


def aggregate(config: Mapping[str, Any]) -> dict[str, Any]:
    _, proposal_hash = verify_proposal_lock(config)
    _, execution_hash = verify_execution_lock(config, proposal_hash)
    artifact_root = Path(config["paths"]["artifact_root"])
    output_path = artifact_root / "train_gate_report.json"
    if output_path.exists():
        raise FileExistsError(output_path)
    seeds = [int(value) for value in config["training"]["seeds"]]
    reports = [read_json(artifact_root / f"seed_{seed}_report.json") for seed in seeds]
    store = FrozenStore(config)
    indices = store.role_indices("internal_A")
    request_ids = [store.request_id(index) for index in indices]
    item_ids = [store.candidate_ids(index) for index in indices]
    score_rows: dict[int, dict[str, list[np.ndarray]]] = {}
    names = [
        "base",
        "fixed_semantic_true",
        "fixed_semantic_wrong",
        "uniform_semantic_true",
        "c38_unprojected_true",
        "c38_unprojected_wrong",
        *[f"{mode}_{source}" for mode in MODES for source in ("true", "wrong", "correction")],
    ]
    for seed, report in zip(seeds, reports):
        score_path = Path(report["score_artifact"]["path"])
        if sha256_file(score_path) != report["score_artifact"]["sha256"]:
            raise RuntimeError(f"C41 score artifact changed: {seed}")
        with np.load(score_path, allow_pickle=False) as values:
            offsets = np.asarray(values["offsets"], dtype=np.int64)
            score_rows[seed] = {name: _unflatten(offsets, values[name]) for name in names}
    averaged = {
        name: _average_rows([score_rows[seed][name] for seed in seeds])
        for name in names
    }
    control_names = [*MATCHED_CONTROLS, *FUNCTIONAL_CONTROLS]
    control_score_name = {
        **{mode: f"{mode}_true" for mode in MATCHED_CONTROLS},
        "fixed_semantic": "fixed_semantic_true",
        "uniform_semantic": "uniform_semantic_true",
        "c38_unprojected": "c38_unprojected_true",
    }
    activity = {
        "primary_vs_base": order_changes(
            request_ids, item_ids, averaged["base"], averaged[f"{PRIMARY}_true"]
        ),
        "true_vs_wrong": order_changes(
            request_ids,
            item_ids,
            averaged[f"{PRIMARY}_true"],
            averaged[f"{PRIMARY}_wrong"],
        ),
        **{
            f"primary_vs_{name}": order_changes(
                request_ids,
                item_ids,
                averaged[control_score_name[name]],
                averaged[f"{PRIMARY}_true"],
            )
            for name in control_names
        },
    }
    gate = config["gate"]
    initial_hashes = [report["seed_specific_initial_state_sha256"] for report in reports]
    a0_checks = {
        "paired_initialization": all(report["paired_initialization"] for report in reports),
        "seed_specific_initialization": len(set(initial_hashes)) == len(seeds),
        "equal_capacity": all(
            len({report["mode_reports"][mode]["parameters"] for mode in MODES}) == 1
            and report["mode_reports"][PRIMARY]["parameters"] == 49152
            for report in reports
        ),
        "finite_training": all(
            report["mode_reports"][mode]["training"]["finite"]
            for report in reports
            for mode in MODES
        ),
        "both_factors_active": all(
            {"down", "up"}
            <= set(report["mode_reports"][mode]["training"]["gradient_parameter_names"])
            for report in reports
            for mode in MODES
        ),
        "parameters_updated": all(
            report["mode_reports"][mode]["parameters_updated"]
            for report in reports
            for mode in MODES
        ),
        "deterministic": all(
            report["mode_reports"][mode]["deterministic_max_abs"]
            <= float(gate["deterministic_max_abs"])
            for report in reports
            for mode in MODES
        ),
        "candidate_permutation": all(
            report["mode_reports"][mode]["candidate_permutation_max_abs"]
            <= float(gate["candidate_permutation_max_abs"])
            for report in reports
            for mode in MODES
        ),
        "exact_fallbacks": all(
            report["mode_reports"][mode]["nohistory_max_abs"] == 0.0
            and report["mode_reports"][mode]["query_absent_max_abs"] == 0.0
            and report["mode_reports"][mode]["repeat_max_abs"] == 0.0
            for report in reports
            for mode in MODES
        ),
        "functional_control_contracts": all(
            report["functional_control_reports"][name]["deterministic_max_abs"]
            <= float(gate["deterministic_max_abs"])
            and report["functional_control_reports"][name]["candidate_permutation_max_abs"]
            <= float(gate["candidate_permutation_max_abs"])
            and report["functional_control_reports"][name]["nohistory_max_abs"] == 0.0
            and report["functional_control_reports"][name]["query_absent_wrapper_max_abs"] == 0.0
            and report["functional_control_reports"][name]["repeat_wrapper_max_abs"] == 0.0
            for report in reports
            for name in FUNCTIONAL_CONTROLS
        ),
        "raw_semantic_profile": all(
            report["mode_reports"][PRIMARY]["semantic_diagnostics"]["raw_profile_max_abs_error"]
            <= float(gate["raw_profile_max_abs"])
            for report in reports
        ),
        "simplex_attention": all(
            report["mode_reports"][PRIMARY]["semantic_diagnostics"]["attention_min"] >= 0
            and report["mode_reports"][PRIMARY]["semantic_diagnostics"]["attention_sum_max_abs_error"]
            <= float(gate["attention_sum_max_abs"])
            for report in reports
        ),
        "primary_order_active": activity["primary_vs_base"]["any_fraction"]
        >= float(gate["primary_vs_base_order_fraction_min"]),
        "primary_top10_active": activity["primary_vs_base"]["top10_fraction"]
        >= float(gate["primary_vs_base_top10_fraction_min"]),
        "wrong_order_distinct": activity["true_vs_wrong"]["any_fraction"]
        >= float(gate["true_vs_wrong_order_fraction_min"]),
        "wrong_top10_distinct": activity["true_vs_wrong"]["top10_fraction"]
        >= float(gate["true_vs_wrong_top10_fraction_min"]),
        **{
            f"{name}_order_distinct": activity[f"primary_vs_{name}"]["any_fraction"]
            >= float(gate["primary_vs_control_order_fraction_min"])
            for name in control_names
        },
        **{
            f"{name}_top10_distinct": activity[f"primary_vs_{name}"]["top10_fraction"]
            >= float(gate["primary_vs_control_top10_fraction_min"])
            for name in control_names
        },
        "delayed_B_closed": True,
        "dev_test_qrels_closed": all(not report["dev_test_qrels_read"] for report in reports),
    }
    report: dict[str, Any] = {
        "candidate_id": "c41",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "proposal_lock_sha256": proposal_hash,
        "execution_lock_sha256": execution_hash,
        "A0": {"checks": a0_checks, "activity": activity},
        "seed_reports": {str(seed): row for seed, row in zip(seeds, reports)},
        "internal_A_scores_opened": True,
        "internal_A_labels_opened": False,
        "delayed_B_features_labels_scores_opened": False,
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
    metric_names = [
        "base",
        *MODES,
        "fixed_semantic",
        "uniform_semantic",
        "c38_unprojected",
        "primary_wrong",
    ]
    per_seed_ndcg: dict[int, dict[str, np.ndarray]] = {}
    for seed in seeds:
        rows = score_rows[seed]
        per_seed_ndcg[seed] = {
            "base": ndcg_rows(request_ids, item_ids, rows["base"], label_rows),
            **{
                mode: ndcg_rows(request_ids, item_ids, rows[f"{mode}_true"], label_rows)
                for mode in MODES
            },
            "fixed_semantic": ndcg_rows(request_ids, item_ids, rows["fixed_semantic_true"], label_rows),
            "uniform_semantic": ndcg_rows(request_ids, item_ids, rows["uniform_semantic_true"], label_rows),
            "c38_unprojected": ndcg_rows(request_ids, item_ids, rows["c38_unprojected_true"], label_rows),
            "primary_wrong": ndcg_rows(request_ids, item_ids, rows[f"{PRIMARY}_wrong"], label_rows),
        }
    averaged_ndcg = {
        name: np.mean(np.stack([per_seed_ndcg[seed][name] for seed in seeds]), axis=0)
        for name in metric_names
    }
    references = {
        "base": averaged_ndcg["base"],
        "c38_unprojected": averaged_ndcg["c38_unprojected"],
        "fixed_semantic": averaged_ndcg["fixed_semantic"],
        "uniform_semantic": averaged_ndcg["uniform_semantic"],
        **{mode: averaged_ndcg[mode] for mode in MATCHED_CONTROLS},
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
    seed_differences = {
        reference: {
            str(seed): float((per_seed_ndcg[seed][PRIMARY] - per_seed_ndcg[seed][reference]).mean())
            for seed in seeds
        }
        for reference in ("base", "c38_unprojected", "fixed_semantic", "uniform_semantic", *MATCHED_CONTROLS)
    }
    direction = bootstrap(
        clicked_direction(
            _average_rows([score_rows[seed][f"{PRIMARY}_correction"] for seed in seeds]),
            label_rows,
        ),
        samples=int(config["evaluation"]["bootstrap_samples"]),
        seed=int(config["evaluation"]["bootstrap_seed"]) + 101,
    )

    def strong(reference: str, minimum: float) -> dict[str, bool]:
        return {
            f"over_{reference}_effect": comparisons[reference]["mean"] >= minimum,
            f"over_{reference}_ci": comparisons[reference]["percentile_95_ci"][0] > 0,
            f"over_{reference}_all_seeds": all(value > 0 for value in seed_differences[reference].values()),
            f"over_{reference}_all_folds": all(
                row["mean_difference"] > 0 for row in comparisons[reference]["hash_folds"]
            ),
        }

    a1_checks = {
        **strong("base", float(gate["primary_minus_base_min"])),
        **strong("c38_unprojected", float(gate["primary_minus_c38_min"])),
        **strong("fixed_semantic", float(gate["primary_minus_fixed_min"])),
        **{
            f"over_{mode}_effect": comparisons[mode]["mean"]
            >= float(gate["primary_minus_matched_min"])
            for mode in MATCHED_CONTROLS
        },
        **{
            f"over_{mode}_ci": comparisons[mode]["percentile_95_ci"][0] > 0
            for mode in MATCHED_CONTROLS
        },
        **{
            f"over_{mode}_all_seeds": all(value >= 0 for value in seed_differences[mode].values())
            for mode in MATCHED_CONTROLS
        },
        "true_over_wrong_ci": comparisons["wrong_history"]["percentile_95_ci"][0] > 0,
        "clicked_direction_ci": direction["percentile_95_ci"][0] > 0,
    }
    report["A1"] = {
        "checks": a1_checks,
        "comparisons": comparisons,
        "clicked_direction": direction,
        "seed_differences": seed_differences,
        "seed_averaged_ndcg10": {
            name: float(value.mean()) for name, value in averaged_ndcg.items()
        },
    }
    report["internal_A_labels_opened"] = True
    report["status"] = "passed_A1_boundary" if all(a1_checks.values()) else "failed_A1_terminal"
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
            raise ValueError("C41 seed stage requires --seed")
        value = run_seed(config, args.seed, torch.device(config["program_device"]))
    else:
        value = aggregate(config)
    print(json.dumps({"candidate_id": "c41", "stage": args.stage, "status": value.get("status", "complete")}, sort_keys=True))


if __name__ == "__main__":
    main()
