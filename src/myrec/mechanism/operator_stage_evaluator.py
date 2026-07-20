"""Shared qrels-gated evaluator for N15/N16 operator bundles."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from myrec.baselines.motivation_v12_contracts import sanitize_record_for_model
from myrec.eval.target_aware_surfaces import build_target_aware_surface_memberships
from myrec.mechanism.attention_edge_runtime import (
    DEEP_DIVE_MANIFEST_PATH,
    SUPPORTED_METHODS,
    _load_manifest,
)
from myrec.mechanism.deep_dive_native_evaluator import benjamini_hochberg, cluster_mean_inference
from myrec.mechanism.operator_stage_runtime import (
    N15_N16_MANIFEST_PATH,
    N15_N16_MANIFEST_SHA256,
    operator_stage_runtime_implementation_identity,
)
from myrec.mechanism.operator_stage_scoring import RMSNORM_CONDITIONS, RESIDUAL_CONDITIONS
from myrec.mechanism.routing_boundary_runtime import (
    routing_boundary_runtime_implementation_identity,
)
from myrec.mechanism.swiglu_formation_runtime import (
    swiglu_formation_runtime_implementation_identity,
)
from myrec.mechanism.swiglu_formation_scoring import SWIGLU_CONDITIONS
from myrec.mechanism.routing_boundary_scoring import GQA_CONDITIONS, HEAD_NORM_CONDITIONS
from myrec.mechanism.patch_evaluator import _target_margins
from myrec.mechanism.qkv_projection_evaluator import _metric_delta, _ndcg_values, _score_map, _write_json, _append_jsonl
from myrec.mechanism.representation_evaluator import (
    STRICT_TRANSFER_SURFACE,
    _audit_candidate_and_request_manifests,
    _load_dev_qrels,
)
from myrec.mechanism.representation_probe import normalize_query, normalized_query_fold
from myrec.mechanism.scalar_condition_bundle import audit_scalar_partial
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl


def evaluate_operator_bundles(
    standardized_dir: str | Path,
    bundle_dirs: Mapping[str, str | Path],
    output_dir: str | Path,
    analysis_run_id: str,
    *,
    kind: str,
    dev_eval_log_path: str | Path = "reports/dev_eval_log.jsonl",
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    if set(bundle_dirs) != set(SUPPORTED_METHODS):
        raise ValueError("operator evaluator requires Q2 and Q3 bundles")
    if kind not in {
        "n15_residual_composition",
        "n16_rmsnorm",
        "n17_head_norm",
        "n18_gqa_grouping",
        "n25_swiglu_formation",
    }:
        raise ValueError(f"unsupported operator evaluator kind={kind}")
    conditions = {
        "n15_residual_composition": RESIDUAL_CONDITIONS,
        "n16_rmsnorm": RMSNORM_CONDITIONS,
        "n17_head_norm": HEAD_NORM_CONDITIONS,
        "n18_gqa_grouping": GQA_CONDITIONS,
        "n25_swiglu_formation": SWIGLU_CONDITIONS,
    }[kind]
    standardized_dir = Path(standardized_dir)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"operator evaluation output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_records = list(iter_jsonl(standardized_dir / "records_dev.jsonl"))
    records = [sanitize_record_for_model(row) for row in raw_records]
    if len(records) != 8000:
        raise ValueError("operator evaluator requires all 8000 dev requests")
    candidates = _audit_candidate_and_request_manifests(
        standardized_dir / "candidate_manifest.json",
        standardized_dir / "request_manifest.json",
        records,
        raw_records,
    )
    parent_manifest = _load_manifest(DEEP_DIVE_MANIFEST_PATH)
    bundles = {
        method_id: _audit_bundle(bundle_dirs[method_id], records, method_id, conditions, kind)
        for method_id in SUPPORTED_METHODS
    }
    if not np.array_equal(
        bundles[SUPPORTED_METHODS[0]]["eligibility"],
        bundles[SUPPORTED_METHODS[1]]["eligibility"],
    ):
        raise ValueError("operator eligibility differs across models")
    eligibility = bundles[SUPPORTED_METHODS[0]]["eligibility"]
    if int(eligibility.sum()) != 7254:
        raise ValueError("operator expected 7254 eligible requests")
    identities = {
        (
            bundle["metadata"].get("block_zero_based"),
            bundle["metadata"].get("branch"),
            bundle["metadata"].get("scope"),
            bundle["metadata"].get("component"),
        )
        for bundle in bundles.values()
    }
    if len(identities) != 1:
        raise ValueError("Q2/Q3 operator scope differs")
    implementation = _implementation_for_kind(kind)
    pre_qrels = {
        "schema_version": 1,
        "analysis_type": f"{kind}_pre_qrels_integrity",
        "analysis_run_id": analysis_run_id,
        "status": "passed",
        "qrels_read": False,
        "checks": {
            "two_complete_bundles": True,
            "all_finite_coverage": True,
            "identity_conditions_at_most_1e-5": True,
            "eligibility_exactly_7254": True,
            "qrels_blind_scoring": True,
        },
        "implementation_digest": implementation["digest"],
    }
    pre_qrels_path = output_dir / "pre_qrels_audit.json"
    _write_json(pre_qrels_path, pre_qrels)
    qrels_path = standardized_dir / "qrels_dev.jsonl"
    if sha256_file(qrels_path) != parent_manifest["frozen_inputs"]["qrels_dev_sha256"]:
        raise ValueError("operator qrels hash mismatch")
    gains = _load_dev_qrels(qrels_path, candidates)
    request_ids = [record.request_id for record in records]
    clusters = np.asarray([normalize_query(record.query) for record in records], dtype=np.str_)
    folds = np.asarray([normalized_query_fold(record.query) for record in records], dtype=np.int8)
    memberships = build_target_aware_surface_memberships(
        standardized_dir / "records_dev.jsonl", candidates, gains
    )
    strict = np.asarray(
        [request_id in memberships[STRICT_TRANSFER_SURFACE] for request_id in request_ids],
        dtype=bool,
    )
    active = [
        name[len("full_") :]
        for name in conditions
        if name.startswith("full_") and not name.endswith("identity")
    ]
    family_rows: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    per_request: dict[str, np.ndarray] = {}
    for method_id in SUPPORTED_METHODS:
        scores = bundles[method_id]["scores"]
        method_results: dict[str, Any] = {}
        for suffix in active:
            full_name = f"full_{suffix}"
            null_name = f"null_{suffix}"
            full_delta = _metric_delta(records, candidates, gains, scores[full_name], scores["baseline_full"])
            null_delta = _metric_delta(records, candidates, gains, scores[null_name], scores["baseline_null"])
            baseline_gap = _metric_delta(records, candidates, gains, scores["baseline_full"], scores["baseline_null"])
            operator_gap = _metric_delta(records, candidates, gains, scores[full_name], scores[null_name])
            endpoint_results: dict[str, Any] = {}
            for endpoint in ("target_margin", "ndcg@10"):
                contrasts = {
                    "full_operator_delta": full_delta[endpoint],
                    "null_operator_delta": null_delta[endpoint],
                    "transfer_gap_change": operator_gap[endpoint] - baseline_gap[endpoint],
                }
                contrast_results: dict[str, Any] = {}
                for contrast_name, values in contrasts.items():
                    registered_rows = []
                    for fold_name, fold_mask in (("all", np.ones(len(records), dtype=bool)), ("0", folds == 0), ("1", folds == 1)):
                        mask = strict & eligibility & fold_mask & np.isfinite(values)
                        inferred = cluster_mean_inference(values[mask], clusters[mask])
                        registered_rows.append({
                            "surface": STRICT_TRANSFER_SURFACE,
                            "eligibility": "frozen_content_control_eligible",
                            "normalized_query_fold": fold_name,
                            **inferred,
                        })
                    all_row = next(row for row in registered_rows if row["normalized_query_fold"] == "all")
                    family_rows.append({
                        "method_id": method_id,
                        "mode": suffix,
                        "endpoint": endpoint,
                        "contrast": contrast_name,
                        "two_sided_p": float(all_row["two_sided_p"]),
                    })
                    contrast_results[contrast_name] = {
                        "registered": registered_rows,
                        "descriptive_full_population": {
                            "surface": STRICT_TRANSFER_SURFACE,
                            "eligibility": "all_requests_with_ineligible_copied_baseline",
                            "normalized_query_fold": "all",
                            **cluster_mean_inference(values[strict & np.isfinite(values)], clusters[strict & np.isfinite(values)]),
                        },
                    }
                    per_request[f"{method_id}__{suffix}__{endpoint}__{contrast_name}"] = values
                endpoint_results[endpoint] = contrast_results
            method_results[suffix] = endpoint_results
        results[method_id] = method_results
    q_values = benjamini_hochberg([row["two_sided_p"] for row in family_rows])
    for row, q_value in zip(family_rows, q_values):
        row["bh_q"] = float(q_value)
    per_request_path = output_dir / "per_request_contrasts.npz"
    np.savez(
        per_request_path,
        **per_request,
        request_ids=np.asarray(request_ids, dtype=np.str_),
        normalized_queries=clusters,
        folds=folds,
        strict_mask=strict,
        frozen_eligible_mask=eligibility,
    )
    metrics = {
        "schema_version": 1,
        "analysis_type": kind,
        "analysis_run_id": analysis_run_id,
        "primary_surface": STRICT_TRANSFER_SURFACE,
        "registered_eligibility": "frozen_content_control_eligible",
        "implementation_digest": implementation["digest"],
        "eligible_requests": int(eligibility.sum()),
        "strict_transfer_requests": int(strict.sum()),
        "strict_transfer_eligible_requests": int((strict & eligibility).sum()),
        "bootstrap": {"cluster": "normalized_query", "samples": 5000, "seed": 20260715},
        "multiple_testing": {"family": "model_x_mode_x_endpoint_x_contrast", "family_size": len(family_rows), "method": "benjamini_hochberg"},
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
    _append_jsonl(Path(dev_eval_log_path), {
        "analysis_type": kind,
        "run_id": analysis_run_id,
        "method_ids": list(SUPPORTED_METHODS),
        "split": "dev",
        "qrels_sha256": metrics["qrels_dev_sha256"],
        "metrics_path": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
    })
    return metrics


def _audit_bundle(
    root: str | Path,
    records: Sequence[Any],
    method_id: str,
    conditions: Sequence[str],
    kind: str,
) -> dict[str, Any]:
    root = Path(root)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if (
        metadata.get("status") != "completed"
        or metadata.get("result_eligible") is not True
        or metadata.get("qrels_read") is not False
        or metadata.get("source_test_opened") is not False
        or metadata.get("complete_finite_score_coverage") is not True
        or metadata.get("identity_passed") is not True
        or float(metadata.get("maximum_identity_delta", math.inf)) > 1.0e-5
        or metadata.get("method_id") != method_id
        or metadata.get("analysis_stage") != kind
        or tuple(metadata.get("score_conditions", ())) != tuple(conditions)
    ):
        raise ValueError(f"operator bundle metadata failed integrity: {root}")
    scores_path = root / "scores.jsonl"
    if metadata.get("scores_sha256") != sha256_file(scores_path):
        raise ValueError("operator score hash differs")
    observed = audit_scalar_partial(scores_path, records, conditions)
    if observed["completed_requests"] != len(records):
        raise ValueError("operator bundle has incomplete coverage")
    width = len(records[0].candidates)
    scores = {name: np.zeros((len(records), width), dtype=np.float64) for name in conditions}
    eligibility = np.zeros(len(records), dtype=bool)
    for ordinal, block_row in enumerate(iter_jsonl(scores_path)):
        eligibility[ordinal] = bool(block_row.get("content_control_eligible"))
        for candidate_ordinal, row in enumerate(block_row["rows"]):
            for name in conditions:
                scores[name][ordinal, candidate_ordinal] = float(row["conditions"][name])
    if int(eligibility.sum()) != 7254:
        raise ValueError("operator bundle eligibility count drift")
    for path_kind in ("full", "null"):
        identity_names = (
            [f"{path_kind}_{operator}_identity" for operator in ("gate_pre", "up", "silu_gate", "product")]
            if kind == "n25_swiglu_formation"
            else [f"{path_kind}_{_identity_prefix(kind)}_identity"]
        )
        baseline = f"baseline_{path_kind}"
        for identity in identity_names:
            if float(np.max(np.abs(scores[identity] - scores[baseline]))) > 1.0e-5:
                raise ValueError(f"operator {identity} condition exceeds tolerance")
    implementation = _implementation_for_kind(kind)
    if metadata.get("implementation_identity", {}).get("digest") != implementation["digest"]:
        raise ValueError("operator implementation digest drift")
    return {"root": root, "metadata": metadata, "scores": scores, "eligibility": eligibility}


def _identity_prefix(kind: str) -> str:
    return {
        "n15_residual_composition": "residual",
        "n16_rmsnorm": "norm",
        "n17_head_norm": "head_norm",
        "n18_gqa_grouping": "gqa",
        "n25_swiglu_formation": "swiglu",
    }[kind]


def _implementation_for_kind(kind: str) -> dict[str, Any]:
    if kind in {"n17_head_norm", "n18_gqa_grouping"}:
        return routing_boundary_runtime_implementation_identity()
    if kind == "n25_swiglu_formation":
        return swiglu_formation_runtime_implementation_identity()
    return operator_stage_runtime_implementation_identity()
