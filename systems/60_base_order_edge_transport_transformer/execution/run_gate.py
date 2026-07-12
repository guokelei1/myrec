"""Run C60's exposed-role mechanics and formulation gates."""

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
from train.data import C56Store  # noqa: E402
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
from model.edge_transport import BaseOrderEdgeTransportTransformer  # noqa: E402
from myrec.eval.metrics import ScoredCandidate, ndcg_at_k, sort_candidates  # noqa: E402


C59_SCORE_NAMES = (
    "base",
    "item_only",
    "primary",
    "wrong",
    "slot_budget_no_null",
    "history_softmax",
    "pooled_history",
    "raw_query",
)
C60_SCORE_NAMES = (
    "base",
    "primary",
    "wrong",
    "signed_adjacent",
    "hard_adjacent",
    "history_axis_adjacent",
    "raw_query_adjacent",
    "direct_additive",
)


def assert_sources(config: Mapping[str, Any]) -> None:
    for path_name, hash_name in (
        ("selection", "c56_v2_selection_sha256"),
        ("contextual_manifest", "c56_contextual_manifest_sha256"),
        ("c56_data_source", "c56_data_source_sha256"),
        ("c59_a0_report", "c59_a0_report_sha256"),
        ("c59_execution_lock", "c59_execution_lock_sha256"),
        ("c59_report", "c59_report_sha256"),
        ("c28_report", "c28_report_sha256"),
    ):
        if sha256_file(REPO_ROOT / config["paths"][path_name]) != config["integrity"][hash_name]:
            raise RuntimeError(f"C60 registered source changed: {path_name}")
    a0 = read_json(REPO_ROOT / config["paths"]["c59_a0_report"])
    if a0.get("status") != "passed_label_free_mechanics":
        raise RuntimeError("C60 C59 mechanics boundary differs")
    source = read_json(REPO_ROOT / config["paths"]["c59_report"])
    if source.get("status") != "failed_exact_semantic_candidate_budget_terminal":
        raise RuntimeError("C60 C59 utility boundary differs")
    if source.get("fit_holdout_labels_read") is not True:
        raise RuntimeError("C60 exposed-role declaration differs")
    if source.get("C26_A_B_escrow_dev_test_qrels_opened") is not False:
        raise PermissionError("C60 inherited protected roles are not closed")
    root = REPO_ROOT / config["paths"]["c59_artifact_root"]
    for shard, expected in config["integrity"]["c59_shard_report_sha256"].items():
        if sha256_file(root / f"shard_{shard}_report.json") != expected:
            raise RuntimeError(f"C60 C59 shard report changed: {shard}")


def assert_cuda(config: Mapping[str, Any], device_name: str) -> None:
    physical = int(config["resources"]["physical_gpu"])
    if device_name != "cuda:0" or os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical):
        raise RuntimeError("C60 GPU binding differs")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise RuntimeError("C60 deterministic CUBLAS workspace absent")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("C60 requires exactly one visible GPU")


def unflatten(offsets: np.ndarray, values: np.ndarray) -> list[np.ndarray]:
    return [
        np.asarray(values[int(offsets[row]) : int(offsets[row + 1])], dtype=np.float64).copy()
        for row in range(len(offsets) - 1)
    ]


def load_c59_rows(config: Mapping[str, Any], count: int) -> dict[str, list[np.ndarray]]:
    root = REPO_ROOT / config["paths"]["c59_artifact_root"]
    output: dict[str, list[np.ndarray | None]] = {
        name: [None] * count for name in C59_SCORE_NAMES
    }
    seen: set[int] = set()
    for shard in range(4):
        report = read_json(root / f"shard_{shard}_report.json")
        path = REPO_ROOT / report["score_artifact"]["path"]
        if sha256_file(path) != report["score_artifact"]["sha256"]:
            raise RuntimeError("C60 inherited C59 score artifact changed")
        with np.load(path, allow_pickle=False) as source:
            positions = np.asarray(source["positions"], dtype=np.int64)
            offsets = np.asarray(source["offsets"], dtype=np.int64)
            rows = {name: unflatten(offsets, source[name]) for name in C59_SCORE_NAMES}
        for local, raw_position in enumerate(positions):
            position = int(raw_position)
            if position in seen or position < 0 or position >= count:
                raise RuntimeError("C60 inherited shard coverage differs")
            seen.add(position)
            for name in C59_SCORE_NAMES:
                output[name][position] = rows[name][local]
    if seen != set(range(count)):
        raise RuntimeError("C60 inherited shard coverage incomplete")
    return {name: [value for value in values if value is not None] for name, values in output.items()}


def ranked(request_id: str, items: Sequence[str], scores: np.ndarray) -> list[str]:
    return [
        row.item_id
        for row in sort_candidates(
            request_id,
            [ScoredCandidate(str(item), float(score)) for item, score in zip(items, scores)],
        )
    ]


def canonical_order(request_id: str, items: Sequence[str], base: np.ndarray) -> np.ndarray:
    order = ranked(request_id, items, base)
    positions = {str(item): index for index, item in enumerate(items)}
    if len(positions) != len(items):
        raise ValueError("C60 duplicate candidate item")
    return np.asarray([positions[item] for item in order], dtype=np.int64)


def apply_transport(
    model: BaseOrderEdgeTransportTransformer,
    *,
    request_id: str,
    items: Sequence[str],
    base: np.ndarray,
    evidence: np.ndarray,
    mode: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    order = canonical_order(request_id, items, base)
    count = len(base)
    with torch.inference_mode():
        output = model(
            base_scores=torch.as_tensor(base, dtype=torch.float64, device=device)[None],
            evidence=torch.as_tensor(evidence, dtype=torch.float64, device=device)[None],
            candidate_mask=torch.ones((1, count), dtype=torch.bool, device=device),
            canonical_order=torch.as_tensor(order, dtype=torch.long, device=device)[None],
            mode=mode,
        )
    return (
        output.scores[0].cpu().numpy().copy(),
        output.correction[0].cpu().numpy().copy(),
        output.transport[0, : max(0, count - 1)].cpu().numpy().copy(),
        output.base_gap[0, : max(0, count - 1)].cpu().numpy().copy(),
    )


def reference(
    store: C56Store, indices: Sequence[int], scores: Sequence[np.ndarray]
) -> dict[str, Any]:
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


def flatten(rows: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    offsets = [0]
    for row in rows:
        offsets.append(offsets[-1] + len(row))
    return np.asarray(offsets, dtype=np.int64), np.concatenate(rows).astype(np.float64, copy=False)


def save_rows(path: Path, rows: Mapping[str, Sequence[np.ndarray]]) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(path)
    offsets, _ = flatten(rows["base"])
    payload = {"offsets": offsets}
    for name in C60_SCORE_NAMES:
        payload[name] = flatten(rows[name])[1]
    np.savez(path, **payload)
    return {"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256_file(path)}


def load_rows(path: Path, expected: str) -> dict[str, list[np.ndarray]]:
    if sha256_file(path) != expected:
        raise RuntimeError("C60 score artifact changed")
    with np.load(path, allow_pickle=False) as source:
        offsets = np.asarray(source["offsets"], dtype=np.int64)
        return {name: unflatten(offsets, source[name]) for name in C60_SCORE_NAMES}


def permutation_audit(
    model: BaseOrderEdgeTransportTransformer,
    request_id: str,
    items: Sequence[str],
    base: np.ndarray,
    evidence: np.ndarray,
    device: torch.device,
) -> float:
    first = apply_transport(
        model, request_id=request_id, items=items, base=base, evidence=evidence,
        mode="one_sided", device=device,
    )[0]
    order = np.arange(len(base) - 1, -1, -1)
    second = apply_transport(
        model,
        request_id=request_id,
        items=[items[index] for index in order],
        base=base[order],
        evidence=evidence[order],
        mode="one_sided",
        device=device,
    )[0]
    return float(np.max(np.abs(first - second[np.argsort(order)])))


def run_a0(config_path: str | Path, device_name: str) -> dict[str, Any]:
    config = load_config(config_path)
    _, execution_hash = verify_execution(config)
    assert_sources(config)
    assert_cuda(config, device_name)
    torch.use_deterministic_algorithms(True)
    store = C56Store(config, REPO_ROOT)
    holdout = store.role("holdout")
    actual_hash = store.candidate_hash(holdout)
    expected_hash = store.selection["candidate_key_sha256"]["holdout"]
    if actual_hash != expected_hash:
        raise RuntimeError("C60 candidate hash differs")
    inherited = load_c59_rows(config, len(holdout))
    model = BaseOrderEdgeTransportTransformer().to(torch.device(device_name)).eval()
    if model.parameter_count() != 0:
        raise RuntimeError("C60 fixed formulation has parameters")
    device = torch.device(device_name)
    rows: dict[str, list[np.ndarray]] = {name: [] for name in C60_SCORE_NAMES}
    conservation_error = capacity_error = zero_error = 0.0
    active_edges = total_edges = 0
    for row, index in enumerate(holdout):
        request_id = store.data.request_ids[index]
        items = store.data.candidate_ids(index)
        base = inherited["base"][row]
        evidence = inherited["primary"][row] - base
        outputs = {}
        outputs["primary"], correction, transport, gap = apply_transport(
            model, request_id=request_id, items=items, base=base, evidence=evidence,
            mode="one_sided", device=device,
        )
        outputs["wrong"] = apply_transport(
            model, request_id=request_id, items=items, base=base,
            evidence=inherited["wrong"][row] - base, mode="one_sided", device=device,
        )[0]
        outputs["signed_adjacent"] = apply_transport(
            model, request_id=request_id, items=items, base=base, evidence=evidence,
            mode="signed", device=device,
        )[0]
        outputs["hard_adjacent"] = apply_transport(
            model, request_id=request_id, items=items, base=base, evidence=evidence,
            mode="hard", device=device,
        )[0]
        outputs["history_axis_adjacent"] = apply_transport(
            model, request_id=request_id, items=items, base=base,
            evidence=inherited["history_softmax"][row] - base,
            mode="one_sided", device=device,
        )[0]
        outputs["raw_query_adjacent"] = apply_transport(
            model, request_id=request_id, items=items, base=base,
            evidence=inherited["raw_query"][row] - base,
            mode="one_sided", device=device,
        )[0]
        outputs["direct_additive"] = inherited["primary"][row]
        rows["base"].append(base.copy())
        for name, value in outputs.items():
            rows[name].append(value)
        conservation_error = max(conservation_error, float(abs(correction.sum())))
        if len(transport):
            capacity_error = max(capacity_error, float(np.max(np.abs(transport) - gap)))
            active_edges += int(np.count_nonzero(transport))
            total_edges += len(transport)
        zero = apply_transport(
            model, request_id=request_id, items=items, base=base,
            evidence=np.zeros_like(base), mode="one_sided", device=device,
        )[0]
        zero_error = max(zero_error, float(np.max(np.abs(zero - base))))
    root = REPO_ROOT / config["paths"]["artifact_root"]
    root.mkdir(parents=True, exist_ok=True)
    artifact = save_rows(root / "scores.npz", rows)
    refs = {name: reference(store, holdout, value) for name, value in rows.items()}
    order_changes = {
        "primary_vs_base": changes(refs["base"], refs["primary"]),
        "primary_vs_wrong": changes(refs["primary"], refs["wrong"]),
        "primary_vs_signed": changes(refs["primary"], refs["signed_adjacent"]),
        "primary_vs_hard": changes(refs["primary"], refs["hard_adjacent"]),
        "primary_vs_history_axis": changes(refs["primary"], refs["history_axis_adjacent"]),
    }
    first_index = holdout[0]
    first_items = store.data.candidate_ids(first_index)
    first_base = inherited["base"][0]
    first_evidence = inherited["primary"][0] - first_base
    deterministic_first = apply_transport(
        model, request_id=store.data.request_ids[first_index], items=first_items,
        base=first_base, evidence=first_evidence, mode="one_sided", device=device,
    )[0]
    deterministic_second = apply_transport(
        model, request_id=store.data.request_ids[first_index], items=first_items,
        base=first_base, evidence=first_evidence, mode="one_sided", device=device,
    )[0]
    deterministic_error = float(np.max(np.abs(deterministic_first - deterministic_second)))
    permutation_error = permutation_audit(
        model, store.data.request_ids[first_index], first_items, first_base, first_evidence, device
    )
    ev = config["evaluation"]

    def above(row: Mapping[str, Any], prefix: str) -> bool:
        return row["any_fraction"] >= float(ev[f"{prefix}_order_change_fraction_min"]) and row[
            "top10_fraction"
        ] >= float(ev[f"{prefix}_top10_change_fraction_min"])

    checks = {
        "candidate_hash_asserted": actual_hash == expected_hash,
        "zero_trainable_parameters": model.parameter_count() == 0,
        "finite_scores": all(np.isfinite(value).all() for values in rows.values() for value in values),
        "deterministic": deterministic_error <= float(ev["deterministic_tolerance"]),
        "candidate_permutation": permutation_error <= float(ev["candidate_permutation_tolerance"]),
        "exact_zero_evidence_base": zero_error == 0.0,
        "score_conservation": conservation_error <= float(ev["conservation_tolerance"]),
        "edge_capacity": capacity_error <= float(ev["capacity_tolerance"]),
        "nonzero_edge_transport": active_edges > 0 and total_edges > 0,
        "base_activity": above(order_changes["primary_vs_base"], "active"),
        "wrong_history_load_bearing": above(order_changes["primary_vs_wrong"], "wrong"),
        "c60_labels_not_read_in_A0": True,
        "fresh_C26_A_B_escrow_dev_test_qrels_closed": True,
    }
    passed = all(checks.values())
    value = {
        "candidate_id": "c60",
        "gate": "A0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed_exposed_label_free_mechanics" if passed else "failed_exposed_mechanics_terminal",
        "execution_lock_sha256": execution_hash,
        "holdout_candidate_key_sha256": actual_hash,
        "holdout_requests": len(holdout),
        "changes": order_changes,
        "diagnostics": {
            "active_edges": active_edges,
            "total_edges": total_edges,
            "active_edge_fraction": active_edges / total_edges,
            "max_abs_conservation_error": conservation_error,
            "max_capacity_excess": max(0.0, capacity_error),
            "zero_evidence_max_abs_error": zero_error,
            "deterministic_max_abs_error": deterministic_error,
            "candidate_permutation_max_abs_error": permutation_error,
        },
        "checks": checks,
        "score_artifact": artifact,
        "c59_holdout_labels_previously_exposed": True,
        "c60_label_values_read": False,
        "fresh_C26_A_B_escrow_dev_test_qrels_opened": False,
    }
    write_once(root / "a0_report.json", value)
    if not passed:
        terminal = dict(value)
        terminal.update(
            {
                "gate_id": config["gate_id"],
                "decision": "close_base_order_edge_transport_on_mechanics",
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


def run_a1(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    _, execution_hash = verify_execution(config)
    assert_sources(config)
    root = REPO_ROOT / config["paths"]["artifact_root"]
    a0 = read_json(root / "a0_report.json")
    if a0.get("status") != "passed_exposed_label_free_mechanics":
        raise PermissionError("C60 A1 requires passed A0")
    store = C56Store(config, REPO_ROOT)
    holdout = store.role("holdout")
    actual_hash = store.candidate_hash(holdout)
    if actual_hash != a0["holdout_candidate_key_sha256"]:
        raise RuntimeError("C60 A1 candidate hash differs")
    scores = load_rows(REPO_ROOT / a0["score_artifact"]["path"], a0["score_artifact"]["sha256"])
    metric = ndcg_rows(store, holdout, scores)
    compare_names = (
        "base",
        "wrong",
        "signed_adjacent",
        "hard_adjacent",
        "history_axis_adjacent",
        "raw_query_adjacent",
        "direct_additive",
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
    controls = tuple(name for name in compare_names if name not in {"base", "wrong"})
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
        "fresh_C26_A_B_escrow_dev_test_qrels_closed": True,
    }
    passed = all(checks.values())
    value = {
        "candidate_id": "c60",
        "gate_id": config["gate_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed_exposed_formulation" if passed else "failed_exposed_formulation_terminal",
        "decision": "authorize_fresh_trainable_compare_exchange_proposal"
        if passed
        else "close_fixed_base_order_edge_transport",
        "execution_lock_sha256": execution_hash,
        "holdout_candidate_key_sha256": actual_hash,
        "holdout_requests": len(holdout),
        "mean_ndcg10": {name: float(value.mean()) for name, value in metric.items()},
        "comparisons": comparisons,
        "fold_directions": folds,
        "checks": checks,
        "claims": {
            "architecture_signal": passed,
            "fresh_result": False,
            "novelty": False,
        },
        "c59_holdout_labels_previously_exposed": True,
        "c60_label_values_read": True,
        "fresh_C26_A_B_escrow_dev_test_qrels_opened": False,
    }
    write_once(root / "formulation_gate_report.json", value)
    write_once(REPO_ROOT / config["paths"]["promoted_report"], value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("a0", "a1"), required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    value = run_a0(args.config, args.device) if args.stage == "a0" else run_a1(args.config)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
