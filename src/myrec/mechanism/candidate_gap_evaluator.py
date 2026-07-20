"""Qrels-free shared evaluator for N10 candidate-gap geometry bundles."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from myrec.baselines.motivation_v12_contracts import sanitize_record_for_model
from myrec.mechanism.attention_edge_evaluator import _append_jsonl
from myrec.mechanism.candidate_gap_scoring import (
    CANDIDATE_GAP_CONDITIONS,
    CANDIDATE_GAP_MODES,
    CANDIDATE_GAP_NODES,
)
from myrec.mechanism.deep_dive_native_evaluator import cluster_mean_inference
from myrec.mechanism.scalar_condition_bundle import audit_scalar_partial
from myrec.mechanism.representation_probe import normalize_query, normalized_query_fold
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl


GEOMETRY_ENDPOINTS = (
    "mean_common_score_shift",
    "candidate_relative_l2_shift",
    "pairwise_order_flip_rate",
    "mean_absolute_score_shift",
)


def evaluate_candidate_gap_bundle(
    standardized_dir: str | Path,
    bundle_dir: str | Path,
    output_dir: str | Path,
    analysis_run_id: str,
    *,
    dev_eval_log_path: str | Path = "reports/dev_eval_log.jsonl",
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    standardized_dir = Path(standardized_dir)
    bundle_dir = Path(bundle_dir)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"candidate-gap evaluator output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [sanitize_record_for_model(row) for row in iter_jsonl(standardized_dir / "records_dev.jsonl")]
    if len(records) != 8000:
        raise ValueError("candidate-gap evaluator requires all 8000 requests")
    metadata = json.loads((bundle_dir / "metadata.json").read_text(encoding="utf-8"))
    expected = {
        "status": "completed",
        "result_eligible": True,
        "qrels_read": False,
        "source_test_opened": False,
        "complete_finite_score_coverage": True,
        "identity_passed": True,
        "score_conditions": list(CANDIDATE_GAP_CONDITIONS),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"candidate-gap metadata differs: {key}")
    from myrec.mechanism.candidate_gap_runtime import candidate_gap_implementation_identity

    implementation = candidate_gap_implementation_identity()
    if metadata.get("implementation_identity", {}).get("digest") != implementation["digest"]:
        raise ValueError("candidate-gap implementation digest differs")
    if float(metadata.get("maximum_identity_delta", math.inf)) > 1.0e-5:
        raise ValueError("candidate-gap identity gate failed")
    scores_path = bundle_dir / "scores.jsonl"
    if metadata.get("scores_sha256") != sha256_file(scores_path):
        raise ValueError("candidate-gap scores hash differs")
    observed = audit_scalar_partial(scores_path, records, CANDIDATE_GAP_CONDITIONS)
    if observed["completed_requests"] != len(records) or observed["completed_score_rows"] != 160753:
        raise ValueError("candidate-gap score coverage differs")

    request_ids = [record.request_id for record in records]
    request_ordinals = {request_id: ordinal for ordinal, request_id in enumerate(request_ids)}
    clusters = np.asarray([normalize_query(record.query) for record in records], dtype=np.str_)
    folds = np.asarray([normalized_query_fold(record.query) for record in records], dtype=np.int8)
    score_arrays = {condition: [None] * len(records) for condition in CANDIDATE_GAP_CONDITIONS}
    for row in iter_jsonl(scores_path):
        ordinal = request_ordinals[str(row["request_id"])]
        values = {condition: [] for condition in CANDIDATE_GAP_CONDITIONS}
        for candidate in row["rows"]:
            for condition in CANDIDATE_GAP_CONDITIONS:
                values[condition].append(float(candidate["conditions"][condition]))
        for condition in CANDIDATE_GAP_CONDITIONS:
            score_arrays[condition][ordinal] = np.asarray(values[condition], dtype=np.float64)
    if any(value[i] is None for value in score_arrays.values() for i in range(len(records))):
        raise ValueError("candidate-gap request score array has a missing request")

    rows: list[dict[str, Any]] = []
    per_request: dict[str, np.ndarray] = {}
    baseline = score_arrays["baseline_full"]
    for condition in CANDIDATE_GAP_CONDITIONS[2:]:
        common = np.asarray([float(np.mean(score_arrays[condition][i] - baseline[i])) for i in range(len(records))])
        relative_l2 = np.asarray([
            float(np.linalg.norm((score_arrays[condition][i] - baseline[i]) - common[i]) / math.sqrt(len(baseline[i])))
            for i in range(len(records))
        ])
        flips = np.asarray([
            _pairwise_flip_rate(baseline[i], score_arrays[condition][i])
            for i in range(len(records))
        ])
        absolute = np.asarray([
            float(np.mean(np.abs(score_arrays[condition][i] - baseline[i])))
            for i in range(len(records))
        ])
        endpoint_values = {
            "mean_common_score_shift": common,
            "candidate_relative_l2_shift": relative_l2,
            "pairwise_order_flip_rate": flips,
            "mean_absolute_score_shift": absolute,
        }
        for endpoint, values in endpoint_values.items():
            inference = []
            for fold_name, fold_mask in (
                ("all", np.ones(len(records), dtype=bool)),
                ("0", folds == 0),
                ("1", folds == 1),
            ):
                mask = fold_mask & np.isfinite(values)
                inference.append({
                    "normalized_query_fold": fold_name,
                    **cluster_mean_inference(values[mask], clusters[mask]),
                })
            rows.append({
                "condition": condition,
                "endpoint": endpoint,
                "mean": inference[0]["mean"],
                "ci95": inference[0]["ci95"],
                "evidence_mode": "registered_n10_candidate_gap_geometry",
            })
            per_request[f"{condition}__{endpoint}"] = values

    per_request_path = output_dir / "per_request_geometry.npz"
    np.savez(
        per_request_path,
        **per_request,
        request_ids=np.asarray(request_ids, dtype=np.str_),
        normalized_queries=clusters,
        folds=folds,
    )
    metrics = {
        "schema_version": 1,
        "analysis_type": "transformer_n10_candidate_gap_geometry",
        "analysis_run_id": analysis_run_id,
        "method_id": metadata["method_id"],
        "nodes": list(CANDIDATE_GAP_NODES),
        "modes": list(CANDIDATE_GAP_MODES),
        "conditions": list(CANDIDATE_GAP_CONDITIONS),
        "implementation_digest": implementation["digest"],
        "geometry_endpoints": list(GEOMETRY_ENDPOINTS),
        "rows": rows,
        "strict_transfer_surface": "not_applicable_qrels_blind_geometry",
        "qrels_read": False,
        "source_test_opened": False,
        "complete_finite_score_coverage": True,
        "identity_passed": True,
        "bootstrap": {"cluster": "normalized_query", "samples": 5000, "seed": 20260715},
        "per_request_geometry_path": str(per_request_path),
        "per_request_geometry_sha256": sha256_file(per_request_path),
        "bundle_metadata_sha256": sha256_file(bundle_dir / "metadata.json"),
        "bundle_scores_sha256": sha256_file(scores_path),
        "command": list(command or []),
        "status": "completed",
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _append_jsonl(
        Path(dev_eval_log_path),
        {
            "analysis_type": metrics["analysis_type"],
            "run_id": analysis_run_id,
            "method_ids": [metadata["method_id"]],
            "split": "internal-dev-qrels-blind",
            "qrels_read": False,
            "metrics_path": str(metrics_path),
            "metrics_sha256": sha256_file(metrics_path),
        },
    )
    return metrics


def _pairwise_flip_rate(baseline: np.ndarray, condition: np.ndarray) -> float:
    if baseline.shape != condition.shape or baseline.ndim != 1:
        raise ValueError("pairwise score vectors are misaligned")
    if len(baseline) < 2:
        return 0.0
    base_diff = baseline[:, None] - baseline[None, :]
    cond_diff = condition[:, None] - condition[None, :]
    upper = np.triu(np.ones(base_diff.shape, dtype=bool), k=1)
    eligible = upper & (np.abs(base_diff) > 1.0e-12)
    if not eligible.any():
        return 0.0
    return float(np.mean(np.sign(base_diff[eligible]) != np.sign(cond_diff[eligible])))
