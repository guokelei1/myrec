"""Shared qrels-gated evaluator for N12 SwiGLU stage bundles."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from myrec.baselines.motivation_v12_contracts import sanitize_record_for_model
from myrec.eval.history_response import gain_ndcg_at_k
from myrec.eval.target_aware_surfaces import build_target_aware_surface_memberships
from myrec.mechanism.attention_edge_runtime import (
    DEEP_DIVE_MANIFEST_PATH,
    FIXED_BLOCKS,
    SUPPORTED_METHODS,
    _load_frozen_baseline,
)
from myrec.mechanism.deep_dive_native_evaluator import benjamini_hochberg, cluster_mean_inference
from myrec.mechanism.mlp_stage_runtime import (
    N12_MANIFEST_PATH,
    N12_MANIFEST_SHA256,
    mlp_stage_runtime_implementation_identity,
    _load_n12_manifest,
)
from myrec.mechanism.mlp_stage_scoring import ACTIVE_STAGE_CONDITIONS, MLP_STAGE_CONDITIONS
from myrec.mechanism.patch_evaluator import _target_margins
from myrec.mechanism.representation_evaluator import STRICT_TRANSFER_SURFACE, _audit_candidate_and_request_manifests, _load_dev_qrels
from myrec.mechanism.representation_probe import normalize_query, normalized_query_fold
from myrec.mechanism.scalar_condition_bundle import audit_scalar_partial
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl


ENDPOINTS = ("target_margin", "ndcg@10")


def evaluate_mlp_stage_bundles(
    standardized_dir: str | Path,
    bundle_dirs: Mapping[str, Mapping[int, str | Path]],
    output_dir: str | Path,
    analysis_run_id: str,
    *,
    dev_eval_log_path: str | Path = "reports/dev_eval_log.jsonl",
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    if set(bundle_dirs) != set(SUPPORTED_METHODS) or any(
        set(map(int, values)) != set(FIXED_BLOCKS) for values in bundle_dirs.values()
    ):
        raise ValueError("N12 evaluator requires Q2/Q3 and blocks 13/20/27")
    standardized_dir = Path(standardized_dir)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"N12 evaluation output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = standardized_dir / "records_dev.jsonl"
    raw_records = list(iter_jsonl(records_path))
    records = [sanitize_record_for_model(row) for row in raw_records]
    if len(records) != 8000:
        raise ValueError("N12 evaluator requires all 8000 dev requests")
    candidates = _audit_candidate_and_request_manifests(
        standardized_dir / "candidate_manifest.json",
        standardized_dir / "request_manifest.json",
        records,
        raw_records,
    )
    manifest = _load_n12_manifest(N12_MANIFEST_PATH)
    bundles = {
        method: {
            block: _audit_bundle(bundle_dirs[method][block], records, method, block)
            for block in FIXED_BLOCKS
        }
        for method in SUPPORTED_METHODS
    }
    implementation_digest = _common_digest(bundles)
    eligibility = np.ones(len(records), dtype=bool)
    pre_qrels = {
        "schema_version": 1,
        "analysis_type": "n12_mlp_stage_pre_qrels_integrity",
        "analysis_run_id": analysis_run_id,
        "status": "passed",
        "qrels_read": False,
        "checks": {
            "six_complete_bundles": True,
            "all_finite_coverage": True,
            "identity_conditions_at_most_1e-5": True,
            "qrels_blind_scoring": True,
        },
        "implementation_digest": implementation_digest,
    }
    pre_qrels_path = output_dir / "pre_qrels_audit.json"
    _write_json(pre_qrels_path, pre_qrels)

    parent = json.loads(DEEP_DIVE_MANIFEST_PATH.read_text(encoding="utf-8"))
    qrels_path = standardized_dir / "qrels_dev.jsonl"
    if sha256_file(qrels_path) != parent["frozen_inputs"]["qrels_dev_sha256"]:
        raise ValueError("N12 qrels hash mismatch")
    gains = _load_dev_qrels(qrels_path, candidates)
    request_ids = [record.request_id for record in records]
    clusters = np.asarray([normalize_query(record.query) for record in records], dtype=np.str_)
    folds = np.asarray([normalized_query_fold(record.query) for record in records], dtype=np.int8)
    memberships = build_target_aware_surface_memberships(records_path, candidates, gains)
    strict = np.asarray([request_id in memberships[STRICT_TRANSFER_SURFACE] for request_id in request_ids], dtype=bool)
    family_rows: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    per_request: dict[str, np.ndarray] = {}
    for method in SUPPORTED_METHODS:
        results[method] = {}
        for block in FIXED_BLOCKS:
            scores = bundles[method][block]["scores"]
            block_results: dict[str, Any] = {}
            for condition in ACTIVE_STAGE_CONDITIONS:
                condition_results: dict[str, Any] = {}
                reference = "baseline_null" if condition.startswith("null_") else "baseline_full"
                for endpoint in ENDPOINTS:
                    values = _metric_delta(records, candidates, gains, scores[condition], scores[reference])[endpoint]
                    rows = []
                    for fold_name, fold_mask in (("all", np.ones(len(records), dtype=bool)), ("0", folds == 0), ("1", folds == 1)):
                        mask = strict & eligibility & fold_mask & np.isfinite(values)
                        rows.append({"surface": STRICT_TRANSFER_SURFACE, "eligibility": "all_fixed_full_null_stage_requests", "normalized_query_fold": fold_name, **cluster_mean_inference(values[mask], clusters[mask])})
                    all_row = next(row for row in rows if row["normalized_query_fold"] == "all")
                    family_rows.append({"method_id": method, "block_zero_based": block, "condition": condition, "endpoint": endpoint, "reference": reference, "two_sided_p": float(all_row["two_sided_p"])})
                    condition_results[endpoint] = {"registered": rows, "descriptive_full_population": {"surface": STRICT_TRANSFER_SURFACE, "eligibility": "all_fixed_full_null_stage_requests", "normalized_query_fold": "all", **cluster_mean_inference(values[strict & np.isfinite(values)], clusters[strict & np.isfinite(values)])}}
                    per_request[f"{method}__b{block}__{condition}__{endpoint}"] = values
                block_results[condition] = condition_results
            results[method][str(block)] = block_results
    q_values = benjamini_hochberg([row["two_sided_p"] for row in family_rows])
    for row, q_value in zip(family_rows, q_values):
        row["bh_q"] = float(q_value)
    per_request_path = output_dir / "per_request_contrasts.npz"
    np.savez(per_request_path, **per_request, request_ids=np.asarray(request_ids, dtype=np.str_), normalized_queries=clusters, folds=folds, strict_mask=strict)
    metrics = {
        "schema_version": 1,
        "analysis_type": "transformer_deep_dive_n12_mlp_stage_operator",
        "analysis_run_id": analysis_run_id,
        "primary_surface": STRICT_TRANSFER_SURFACE,
        "registered_eligibility": "all_fixed_full_null_stage_requests",
        "implementation_digest": implementation_digest,
        "strict_transfer_requests": int(strict.sum()),
        "bootstrap": {"cluster": "normalized_query", "samples": 5000, "seed": 20260715},
        "multiple_testing": {"family": "model_x_block_x_condition_x_endpoint", "family_size": len(family_rows), "method": "benjamini_hochberg"},
        "family_rows": family_rows,
        "results": results,
        "pre_qrels_audit_path": str(pre_qrels_path),
        "pre_qrels_audit_sha256": sha256_file(pre_qrels_path),
        "qrels_read": True,
        "qrels_opened_only_after_score_integrity": True,
        "qrels_dev_sha256": sha256_file(qrels_path),
        "per_request_contrasts_path": str(per_request_path),
        "per_request_contrasts_sha256": sha256_file(per_request_path),
        "command": list(command or []),
        "status": "completed",
    }
    metrics_path = output_dir / "metrics.json"
    _write_json(metrics_path, metrics)
    _append_jsonl(Path(dev_eval_log_path), {"analysis_type": metrics["analysis_type"], "run_id": analysis_run_id, "method_ids": list(SUPPORTED_METHODS), "split": "dev", "qrels_sha256": metrics["qrels_dev_sha256"], "metrics_path": str(metrics_path), "metrics_sha256": sha256_file(metrics_path)})
    return metrics


def _audit_bundle(root: str | Path, records: Sequence[Any], method: str, block: int) -> dict[str, Any]:
    root = Path(root)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("status") != "completed" or metadata.get("result_eligible") is not True or metadata.get("qrels_read") is not False or metadata.get("source_test_opened") is not False or metadata.get("complete_finite_score_coverage") is not True or metadata.get("identity_passed") is not True or float(metadata.get("maximum_identity_delta", math.inf)) > 1.0e-5 or metadata.get("method_id") != method or int(metadata.get("block_zero_based", -1)) != block or tuple(metadata.get("score_conditions", ())) != MLP_STAGE_CONDITIONS:
        raise ValueError(f"N12 bundle metadata failed integrity: {root}")
    scores_path = root / "scores.jsonl"
    if metadata.get("scores_sha256") != sha256_file(scores_path):
        raise ValueError("N12 score hash differs")
    observed = audit_scalar_partial(scores_path, records, MLP_STAGE_CONDITIONS)
    if observed["completed_requests"] != len(records):
        raise ValueError("N12 bundle has incomplete coverage")
    width = len(records[0].candidates)
    scores = {name: np.zeros((len(records), width), dtype=np.float64) for name in MLP_STAGE_CONDITIONS}
    for ordinal, block_row in enumerate(iter_jsonl(scores_path)):
        if int(block_row.get("block_zero_based", -1)) != block:
            raise ValueError("N12 block drift")
        for candidate_ordinal, row in enumerate(block_row["rows"]):
            for name in MLP_STAGE_CONDITIONS:
                scores[name][ordinal, candidate_ordinal] = float(row["conditions"][name])
    for identity, baseline in (("full_gate_identity", "baseline_full"), ("null_gate_identity", "baseline_null")):
        if float(np.max(np.abs(scores[identity] - scores[baseline]))) > 1.0e-5:
            raise ValueError("N12 identity condition exceeds tolerance")
    implementation = mlp_stage_runtime_implementation_identity()
    if metadata.get("implementation_identity", {}).get("digest") != implementation["digest"]:
        raise ValueError("N12 implementation digest drift")
    return {"metadata": metadata, "scores": scores}


def _metric_delta(records: Sequence[Any], candidates: Mapping[str, Sequence[str]], gains: Mapping[str, Mapping[str, float]], left: np.ndarray, right: np.ndarray) -> dict[str, np.ndarray]:
    request_ids = [record.request_id for record in records]
    left_map = _score_map(request_ids, candidates, left)
    right_map = _score_map(request_ids, candidates, right)
    return {"target_margin": _target_margins(request_ids, candidates, gains, left_map) - _target_margins(request_ids, candidates, gains, right_map), "ndcg@10": _ndcg_values(request_ids, candidates, gains, left) - _ndcg_values(request_ids, candidates, gains, right)}


def _score_map(request_ids: Sequence[str], candidates: Mapping[str, Sequence[str]], scores: np.ndarray) -> dict[str, dict[str, float]]:
    return {request_id: {item_id: float(scores[ordinal, candidate_ordinal]) for candidate_ordinal, item_id in enumerate(candidates[request_id])} for ordinal, request_id in enumerate(request_ids)}


def _ndcg_values(request_ids: Sequence[str], candidates: Mapping[str, Sequence[str]], gains: Mapping[str, Mapping[str, float]], scores: np.ndarray) -> np.ndarray:
    result = np.zeros(len(request_ids), dtype=np.float64)
    for ordinal, request_id in enumerate(request_ids):
        item_ids = list(candidates[request_id])
        result[ordinal] = gain_ndcg_at_k(request_id, item_ids, [float(value) for value in scores[ordinal]], [float(gains[request_id].get(item_id, 0.0)) for item_id in item_ids], 10)
    return result


def _common_digest(bundles: Mapping[str, Mapping[int, Mapping[str, Any]]]) -> str:
    digests = {bundles[method][block]["metadata"]["implementation_identity"]["digest"] for method in SUPPORTED_METHODS for block in FIXED_BLOCKS}
    if len(digests) != 1:
        raise ValueError("N12 bundles do not share one implementation digest")
    return next(iter(digests))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")

