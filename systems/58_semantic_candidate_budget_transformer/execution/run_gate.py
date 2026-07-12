"""Run C58's locked label-free mechanics gate and conditional utility gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
C56_ROOT = REPO_ROOT / "systems/56_query_complement_token_competition_transformer"
for value in (str(C56_ROOT), str(REPO_ROOT / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)
from train.data import C56Store, to_device  # noqa: E402
if str(C56_ROOT) in sys.path:
    sys.path.remove(str(C56_ROOT))
sys.path.insert(0, str(SYSTEM_ROOT))

from execution.locking import (  # noqa: E402
    load_config,
    read_json,
    sha256_file,
    verify_execution,
    write_once,
)
from model.semantic_budget import MODES, SemanticCandidateBudgetTransformer  # noqa: E402
from myrec.eval.metrics import ScoredCandidate, ndcg_at_k, sort_candidates  # noqa: E402


PRIMARY = "candidate_budget"
SCORE_NAMES = (
    "base",
    "item_only",
    "primary",
    "wrong",
    "slot_budget_no_null",
    "history_softmax",
    "pooled_history",
    "raw_query",
)
MODE_TO_SCORE = {
    "candidate_budget": "primary",
    "slot_budget_no_null": "slot_budget_no_null",
    "history_softmax": "history_softmax",
    "pooled_history": "pooled_history",
    "raw_query": "raw_query",
}


def assert_sources(config: Mapping[str, Any]) -> None:
    for path_name, hash_name in (
        ("selection", "c56_v2_selection_sha256"),
        ("contextual_manifest", "c56_contextual_manifest_sha256"),
        ("c56_data_source", "c56_data_source_sha256"),
        ("c57_report", "c57_report_sha256"),
    ):
        if sha256_file(REPO_ROOT / config["paths"][path_name]) != config["integrity"][hash_name]:
            raise RuntimeError(f"C58 registered source changed: {path_name}")
    source = read_json(REPO_ROOT / config["paths"]["c57_report"])
    if source.get("status") != "failed_label_free_mechanics_terminal":
        raise RuntimeError("C58 C57 terminal boundary differs")
    if source.get("holdout_fit_labels_read") is not False:
        raise PermissionError("C58 inherited holdout labels are not closed")
    if source.get("C26_A_B_escrow_dev_test_qrels_opened") is not False:
        raise PermissionError("C58 inherited protected roles are not closed")


def assert_cuda(config: Mapping[str, Any], shard_id: int, device_name: str) -> None:
    physical = int(config["resources"]["physical_gpus"][shard_id])
    if device_name != "cuda:0" or os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical):
        raise RuntimeError("C58 GPU binding differs")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise RuntimeError("C58 deterministic CUBLAS workspace absent")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("C58 requires exactly one visible GPU")


def make_model(config: Mapping[str, Any]) -> SemanticCandidateBudgetTransformer:
    value = config["operator"]
    model = SemanticCandidateBudgetTransformer(
        null_logit=float(value["null_logit"]),
        epsilon=float(value["normalization_epsilon"]),
    )
    if model.parameter_count() != 0:
        raise RuntimeError("C58 fixed operator unexpectedly has parameters")
    return model


def max_difference(first: Sequence[np.ndarray], second: Sequence[np.ndarray]) -> float:
    if len(first) != len(second) or not first:
        raise ValueError("C58 comparison rows differ")
    return max(float(np.max(np.abs(a - b))) for a, b in zip(first, second))


def ranked(request_id: str, items: Sequence[str], scores: np.ndarray) -> list[str]:
    return [
        row.item_id
        for row in sort_candidates(
            request_id,
            [ScoredCandidate(str(item), float(score)) for item, score in zip(items, scores)],
        )
    ]


def reference(
    store: C56Store, holdout: Sequence[int], positions: Sequence[int], scores: Sequence[np.ndarray]
) -> dict[str, Any]:
    indices = [int(holdout[int(position)]) for position in positions]
    return {
        "request_ids": [store.data.request_ids[index] for index in indices],
        "item_ids": [store.data.candidate_ids(index) for index in indices],
        "scores": scores,
    }


def changes(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, Any]:
    any_count = top_count = 0
    for request_id, items, a, b in zip(
        first["request_ids"], first["item_ids"], first["scores"], second["scores"]
    ):
        ra, rb = ranked(request_id, items, a), ranked(request_id, items, b)
        any_count += int(ra != rb)
        top_count += int(set(ra[:10]) != set(rb[:10]))
    count = len(first["request_ids"])
    return {
        "requests": count,
        "any_count": any_count,
        "any_fraction": any_count / count,
        "top10_count": top_count,
        "top10_fraction": top_count / count,
    }


def score_one(
    model: SemanticCandidateBudgetTransformer,
    store: C56Store,
    index: int,
    device: torch.device,
    *,
    history_source: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    batch = store.collate([index], history_source=history_source)
    tensors = to_device(batch, device)
    with torch.inference_mode():
        output = model(**tensors)
    count = int(batch["candidate_mask"][0].sum())
    scores = {
        MODE_TO_SCORE[mode]: output.scores[mode][0, :count].cpu().numpy().astype(np.float32, copy=True)
        for mode in MODES
    }
    corrections = {
        MODE_TO_SCORE[mode]: output.corrections[mode][0, :count].cpu().numpy().astype(np.float32, copy=True)
        for mode in MODES
    }
    scores["base"] = np.asarray(batch["base_scores"][0, :count], dtype=np.float32).copy()
    scores["item_only"] = np.asarray(batch["item_only_scores"][0, :count], dtype=np.float32).copy()
    return scores, corrections


def deterministic_audit(
    model: SemanticCandidateBudgetTransformer,
    store: C56Store,
    index: int,
    device: torch.device,
) -> float:
    first, _ = score_one(model, store, index, device, history_source="true")
    second, _ = score_one(model, store, index, device, history_source="true")
    return max_difference([first["primary"]], [second["primary"]])


def permutation_audit(
    model: SemanticCandidateBudgetTransformer,
    store: C56Store,
    index: int,
    device: torch.device,
) -> float:
    tensors = to_device(store.collate([index]), device)
    count = tensors["candidate_mask"].shape[1]
    order = torch.arange(count - 1, -1, -1, device=device)
    inverse = torch.argsort(order)
    moved = dict(tensors)
    for name in (
        "candidate_tokens",
        "candidate_token_mask",
        "candidate_mask",
        "base_scores",
        "item_only_scores",
    ):
        moved[name] = tensors[name][:, order]
    with torch.inference_mode():
        first = model(**tensors).scores[PRIMARY]
        second = model(**moved).scores[PRIMARY][:, inverse]
    return float((first - second).abs().max().cpu())


def fallback_audit(
    model: SemanticCandidateBudgetTransformer,
    store: C56Store,
    device: torch.device,
) -> dict[str, float]:
    nohistory_scores: list[np.ndarray] = []
    nohistory_base: list[np.ndarray] = []
    for index in store.role("structural_nohistory"):
        scores, _ = score_one(model, store, index, device, history_source="true")
        nohistory_scores.append(scores["primary"])
        nohistory_base.append(scores["base"])
    repeat_scores: list[np.ndarray] = []
    repeat_anchor: list[np.ndarray] = []
    for index in store.role("structural_repeat"):
        scores, _ = score_one(model, store, index, device, history_source="true")
        repeat_scores.append(scores["primary"])
        repeat_anchor.append(scores["item_only"])
    return {
        "nohistory_max_abs_vs_base": max_difference(nohistory_scores, nohistory_base),
        "repeat_max_abs_vs_item_only": max_difference(repeat_scores, repeat_anchor),
    }


def flatten(rows: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    offsets = [0]
    for row in rows:
        offsets.append(offsets[-1] + len(row))
    return np.asarray(offsets, dtype=np.int64), np.concatenate(rows).astype(np.float32, copy=False)


def save_rows(
    path: Path,
    positions: Sequence[int],
    rows: Mapping[str, Sequence[np.ndarray]],
) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(path)
    offsets, _ = flatten(rows["base"])
    payload: dict[str, np.ndarray] = {
        "positions": np.asarray(positions, dtype=np.int64),
        "offsets": offsets,
    }
    for name in SCORE_NAMES:
        payload[name] = flatten(rows[name])[1]
    np.savez(path, **payload)
    return {"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256_file(path)}


def run_shard(config_path: str | Path, shard_id: int, device_name: str) -> dict[str, Any]:
    config = load_config(config_path)
    _, execution_hash = verify_execution(config)
    assert_sources(config)
    shards = int(config["resources"]["num_shards"])
    if shard_id < 0 or shard_id >= shards:
        raise ValueError("C58 shard differs")
    assert_cuda(config, shard_id, device_name)
    torch.use_deterministic_algorithms(True)
    store = C56Store(config, REPO_ROOT)
    holdout = store.role("holdout")
    expected_hash = store.selection["candidate_key_sha256"]["holdout"]
    actual_hash = store.candidate_hash(holdout)
    if actual_hash != expected_hash:
        raise RuntimeError("C58 holdout candidate hash differs")
    positions = list(range(shard_id, len(holdout), shards))
    device = torch.device(device_name)
    model = make_model(config).to(device).eval()
    rows: dict[str, list[np.ndarray]] = {name: [] for name in SCORE_NAMES}
    correction_scales: list[float] = []
    for position in positions:
        index = int(holdout[position])
        true, correction = score_one(model, store, index, device, history_source="true")
        wrong, _ = score_one(model, store, index, device, history_source="wrong")
        for name in SCORE_NAMES:
            if name == "wrong":
                rows[name].append(wrong["primary"])
            else:
                rows[name].append(true[name])
        correction_scales.append(float(np.std(correction["primary"], dtype=np.float64)))
    root = REPO_ROOT / config["paths"]["artifact_root"]
    root.mkdir(parents=True, exist_ok=True)
    artifact = save_rows(root / f"shard_{shard_id}_scores.npz", positions, rows)
    deterministic_error = deterministic_audit(model, store, int(holdout[positions[0]]), device)
    permutation_error = permutation_audit(model, store, int(holdout[positions[0]]), device)
    fallback = fallback_audit(model, store, device) if shard_id == 0 else None
    tolerance = float(config["evaluation"]["exact_fallback_tolerance"])
    checks = {
        "candidate_hash_asserted": actual_hash == expected_hash,
        "no_optimizer_or_labels": True,
        "zero_trainable_parameters": model.parameter_count() == 0,
        "finite_scores": all(np.isfinite(value).all() for values in rows.values() for value in values),
        "deterministic": deterministic_error <= float(config["evaluation"]["deterministic_tolerance"]),
        "candidate_permutation": permutation_error <= float(config["evaluation"]["candidate_permutation_tolerance"]),
        "fallback_contract": shard_id != 0 or (
            fallback is not None
            and fallback["nohistory_max_abs_vs_base"] <= tolerance
            and fallback["repeat_max_abs_vs_item_only"] <= tolerance
        ),
        "holdout_labels_closed": True,
        "C26_A_B_escrow_dev_test_qrels_closed": True,
    }
    value = {
        "candidate_id": "c58",
        "stage": "shard",
        "shard_id": shard_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "execution_lock_sha256": execution_hash,
        "physical_gpu": int(config["resources"]["physical_gpus"][shard_id]),
        "positions": len(positions),
        "position_first": positions[0],
        "position_last": positions[-1],
        "primary_correction_std": {
            "mean": float(np.mean(correction_scales)),
            "min": float(np.min(correction_scales)),
            "max": float(np.max(correction_scales)),
        },
        "fallback": fallback,
        "deterministic_max_abs_difference": deterministic_error,
        "candidate_permutation_max_abs_difference": permutation_error,
        "checks": checks,
        "score_artifact": artifact,
        "holdout_candidate_key_sha256": actual_hash,
        "fit_train_labels_read": False,
        "fit_holdout_labels_read": False,
        "C26_A_B_escrow_dev_test_qrels_opened": False,
    }
    write_once(root / f"shard_{shard_id}_report.json", value)
    return value


def unflatten(offsets: np.ndarray, values: np.ndarray) -> list[np.ndarray]:
    return [
        np.asarray(values[int(offsets[row]) : int(offsets[row + 1])], dtype=np.float32).copy()
        for row in range(len(offsets) - 1)
    ]


def load_shard(report: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, list[np.ndarray]]]:
    path = REPO_ROOT / report["score_artifact"]["path"]
    if sha256_file(path) != report["score_artifact"]["sha256"]:
        raise RuntimeError("C58 score artifact changed")
    with np.load(path, allow_pickle=False) as source:
        positions = np.asarray(source["positions"], dtype=np.int64)
        offsets = np.asarray(source["offsets"], dtype=np.int64)
        rows = {name: unflatten(offsets, source[name]) for name in SCORE_NAMES}
    return positions, rows


def collect_rows(
    reports: Sequence[Mapping[str, Any]], count: int
) -> dict[str, list[np.ndarray]]:
    output: dict[str, list[np.ndarray | None]] = {
        name: [None] * count for name in SCORE_NAMES
    }
    seen: set[int] = set()
    for report in reports:
        positions, rows = load_shard(report)
        for local, raw_position in enumerate(positions):
            position = int(raw_position)
            if position in seen or position < 0 or position >= count:
                raise RuntimeError("C58 shard position coverage differs")
            seen.add(position)
            for name in SCORE_NAMES:
                output[name][position] = rows[name][local]
    if seen != set(range(count)):
        raise RuntimeError("C58 shard position coverage incomplete")
    return {name: [value for value in values if value is not None] for name, values in output.items()}


def aggregate_a0(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    _, execution_hash = verify_execution(config)
    assert_sources(config)
    store = C56Store(config, REPO_ROOT)
    holdout = store.role("holdout")
    actual_hash = store.candidate_hash(holdout)
    if actual_hash != store.selection["candidate_key_sha256"]["holdout"]:
        raise RuntimeError("C58 A0 candidate hash differs")
    root = REPO_ROOT / config["paths"]["artifact_root"]
    reports = [
        read_json(root / f"shard_{shard}_report.json")
        for shard in range(int(config["resources"]["num_shards"]))
    ]
    rows = collect_rows(reports, len(holdout))
    positions = list(range(len(holdout)))
    refs = {name: reference(store, holdout, positions, values) for name, values in rows.items()}
    order_changes = {
        "primary_vs_base": changes(refs["base"], refs["primary"]),
        "primary_vs_wrong": changes(refs["primary"], refs["wrong"]),
        "primary_vs_history_axis": changes(refs["primary"], refs["history_softmax"]),
    }
    ev = config["evaluation"]

    def above(row: Mapping[str, Any], prefix: str) -> bool:
        return row["any_fraction"] >= float(ev[f"{prefix}_order_change_fraction_min"]) and row[
            "top10_fraction"
        ] >= float(ev[f"{prefix}_top10_change_fraction_min"])

    owner_fallback = reports[0]["fallback"]
    tolerance = float(ev["exact_fallback_tolerance"])
    checks = {
        "all_shard_execution_checks": all(all(report["checks"].values()) for report in reports),
        "candidate_hash_asserted": all(
            report["holdout_candidate_key_sha256"] == actual_hash for report in reports
        ),
        "complete_disjoint_shard_coverage": sum(report["positions"] for report in reports) == len(holdout),
        "base_activity": above(order_changes["primary_vs_base"], "active"),
        "wrong_history_load_bearing": above(order_changes["primary_vs_wrong"], "wrong"),
        "candidate_axis_load_bearing": above(order_changes["primary_vs_history_axis"], "axis"),
        "exact_nohistory_base": owner_fallback["nohistory_max_abs_vs_base"] <= tolerance,
        "exact_repeat_item_only": owner_fallback["repeat_max_abs_vs_item_only"] <= tolerance,
        "fit_train_and_holdout_labels_closed": True,
        "C26_A_B_escrow_dev_test_qrels_closed": True,
    }
    passed = all(checks.values())
    value = {
        "candidate_id": "c58",
        "gate": "A0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed_label_free_mechanics" if passed else "failed_label_free_mechanics_terminal",
        "execution_lock_sha256": execution_hash,
        "holdout_candidate_key_sha256": actual_hash,
        "holdout_requests": len(holdout),
        "changes": order_changes,
        "fallback": owner_fallback,
        "shard_diagnostics": {
            str(report["shard_id"]): {
                "primary_correction_std": report["primary_correction_std"],
                "deterministic_max_abs_difference": report["deterministic_max_abs_difference"],
                "candidate_permutation_max_abs_difference": report[
                    "candidate_permutation_max_abs_difference"
                ],
            }
            for report in reports
        },
        "checks": checks,
        "fit_train_labels_read": False,
        "fit_holdout_labels_read": False,
        "C26_A_B_escrow_dev_test_qrels_opened": False,
    }
    write_once(root / "a0_report.json", value)
    if not passed:
        terminal = dict(value)
        terminal.update(
            {
                "gate_id": config["gate_id"],
                "decision": "close_fixed_semantic_candidate_budget_on_mechanics",
                "claims": {"architecture_signal": False, "fresh_result": False, "novelty": False},
            }
        )
        write_once(REPO_ROOT / config["paths"]["promoted_report"], terminal)
    return value


def paired_interval(values: np.ndarray, *, samples: int, seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 1000):
        count = min(1000, samples - start)
        positions = rng.integers(0, len(values), size=(count, len(values)))
        draws[start : start + count] = values[positions].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "percentile_95_ci": [float(value) for value in np.quantile(draws, [0.025, 0.975])],
        "requests": len(values),
        "samples": samples,
        "seed": seed,
    }


def ndcg_rows(
    store: C56Store, indices: Sequence[int], scores: Mapping[str, Sequence[np.ndarray]]
) -> dict[str, np.ndarray]:
    output = {name: [] for name in scores}
    for row, index in enumerate(indices):
        request_id = store.data.request_ids[index]
        items = store.data.candidate_ids(index)
        labels = store.label(index)
        positive = {item for item, label in zip(items, labels) if label > 0}
        for name, values in scores.items():
            output[name].append(ndcg_at_k(ranked(request_id, items, values[row]), positive, 10))
    return {name: np.asarray(value, dtype=np.float64) for name, value in output.items()}


def fold_means(values: np.ndarray, request_ids: Sequence[str], folds: int) -> list[float]:
    group = np.asarray(
        [int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big") % folds for value in request_ids]
    )
    return [float(values[group == fold].mean()) for fold in range(folds)]


def aggregate_a1(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    _, execution_hash = verify_execution(config)
    assert_sources(config)
    root = REPO_ROOT / config["paths"]["artifact_root"]
    a0 = read_json(root / "a0_report.json")
    if a0.get("status") != "passed_label_free_mechanics":
        raise PermissionError("C58 A1 requires passed A0")
    store = C56Store(config, REPO_ROOT)
    holdout = store.role("holdout")
    actual_hash = store.candidate_hash(holdout)
    if actual_hash != a0["holdout_candidate_key_sha256"]:
        raise RuntimeError("C58 A1 candidate hash differs")
    reports = [
        read_json(root / f"shard_{shard}_report.json")
        for shard in range(int(config["resources"]["num_shards"]))
    ]
    scores = collect_rows(reports, len(holdout))
    metric = ndcg_rows(store, holdout, scores)
    compare_names = (
        "base",
        "wrong",
        "slot_budget_no_null",
        "history_softmax",
        "pooled_history",
        "raw_query",
    )
    ev = config["evaluation"]
    samples, base_seed = int(ev["bootstrap_samples"]), int(ev["bootstrap_seed"])
    comparisons = {
        f"primary_minus_{name}": paired_interval(
            metric["primary"] - metric[name], samples=samples, seed=base_seed + offset
        )
        for offset, name in enumerate(compare_names)
    }
    request_ids = [store.data.request_ids[index] for index in holdout]
    folds = {
        name: fold_means(
            metric["primary"] - metric[name], request_ids, int(ev["hash_folds"])
        )
        for name in compare_names
    }
    controls = ("slot_budget_no_null", "history_softmax", "pooled_history", "raw_query")
    checks = {
        "A0_passed": True,
        "candidate_hash_asserted": actual_hash == a0["holdout_candidate_key_sha256"],
        "gain_over_base": comparisons["primary_minus_base"]["mean"]
        >= float(ev["ndcg_primary_minus_base_min"])
        and comparisons["primary_minus_base"]["percentile_95_ci"][0] > 0,
        "gain_over_wrong": comparisons["primary_minus_wrong"]["mean"]
        >= float(ev["ndcg_primary_minus_wrong_min"])
        and comparisons["primary_minus_wrong"]["percentile_95_ci"][0] > 0,
        "gain_over_controls": all(
            comparisons[f"primary_minus_{name}"]["mean"]
            >= float(ev["ndcg_primary_minus_each_control_min"])
            and comparisons[f"primary_minus_{name}"]["percentile_95_ci"][0] > 0
            for name in controls
        ),
        "all_fold_directions_positive": all(all(value > 0 for value in row) for row in folds.values()),
        "fit_train_labels_closed": True,
        "C26_A_B_escrow_dev_test_qrels_closed": True,
    }
    passed = all(checks.values())
    value = {
        "candidate_id": "c58",
        "gate_id": config["gate_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed_fixed_semantic_candidate_budget_foundation"
        if passed
        else "failed_fixed_semantic_candidate_budget_terminal",
        "decision": "authorize_fresh_dual_domain_trainable_successor"
        if passed
        else "close_fixed_semantic_candidate_budget_family",
        "execution_lock_sha256": execution_hash,
        "holdout_candidate_key_sha256": actual_hash,
        "holdout_requests": len(holdout),
        "mean_ndcg10": {name: float(value.mean()) for name, value in metric.items()},
        "comparisons": comparisons,
        "fold_directions": folds,
        "checks": checks,
        "claims": {"architecture_signal": passed, "fresh_result": False, "novelty": False},
        "fit_train_labels_read": False,
        "fit_holdout_labels_read": True,
        "C26_A_B_escrow_dev_test_qrels_opened": False,
    }
    write_once(root / "formulation_gate_report.json", value)
    write_once(REPO_ROOT / config["paths"]["promoted_report"], value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("shard", "a0", "a1"), required=True)
    parser.add_argument("--shard-id", type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.stage == "shard":
        if args.shard_id is None:
            raise ValueError("C58 shard stage requires --shard-id")
        value = run_shard(args.config, args.shard_id, args.device)
    elif args.stage == "a0":
        value = aggregate_a0(args.config)
    else:
        value = aggregate_a1(args.config)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
