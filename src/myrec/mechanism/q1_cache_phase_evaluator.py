"""Shared qrels-gated evaluator for complete N20 Q1 cache bundles."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from myrec.baselines.motivation_v12_contracts import sanitize_record_for_model
from myrec.eval.history_response import gain_ndcg_at_k
from myrec.eval.target_aware_surfaces import build_target_aware_surface_memberships
from myrec.mechanism.attention_edge_runtime import DEEP_DIVE_MANIFEST_PATH, _load_manifest
from myrec.mechanism.deep_dive_native_evaluator import benjamini_hochberg, cluster_mean_inference
from myrec.mechanism.patch_evaluator import _target_margins
from myrec.mechanism.q1_cache_phase_runtime import (
    N20_CONDITIONS,
    q1_cache_phase_runtime_implementation_identity,
)
from myrec.mechanism.representation_evaluator import (
    STRICT_TRANSFER_SURFACE,
    _audit_candidate_and_request_manifests,
    _load_dev_qrels,
)
from myrec.mechanism.scalar_condition_bundle import audit_scalar_partial
from myrec.mechanism.representation_probe import normalize_query, normalized_query_fold
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl


def evaluate_q1_cache_phase_bundle(
    standardized_dir: str | Path,
    bundle_dir: str | Path,
    output_dir: str | Path,
    analysis_run_id: str,
    *,
    dev_eval_log_path: str | Path = "reports/dev_eval_log.jsonl",
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Audit N20 completely before opening the frozen development qrels."""

    standardized_dir = Path(standardized_dir)
    bundle_dir = Path(bundle_dir)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"N20 evaluator output is not empty: {output_dir}")
    raw_records = list(iter_jsonl(standardized_dir / "records_dev.jsonl"))
    records = [sanitize_record_for_model(row) for row in raw_records]
    if len(records) != 8000:
        raise ValueError("N20 evaluator requires all 8000 dev requests")
    candidates = _audit_candidate_and_request_manifests(
        standardized_dir / "candidate_manifest.json",
        standardized_dir / "request_manifest.json",
        records,
        raw_records,
    )
    audited = _audit_bundle(bundle_dir, records)
    implementation = q1_cache_phase_runtime_implementation_identity()
    pre_qrels = {
        "schema_version": 1,
        "analysis_type": "n20_q1_cache_phase_pre_qrels_integrity",
        "analysis_run_id": analysis_run_id,
        "status": "passed",
        "qrels_read": False,
        "checks": {
            "complete_finite_score_coverage": True,
            "identity_conditions_at_most_1e-5": True,
            "native_score_matches_frozen_baselines": True,
            "wrong_user_cache_position_integrity": True,
            "qrels_blind_scoring": True,
        },
        "implementation_digest": implementation["digest"],
        "bundle_metadata_sha256": sha256_file(bundle_dir / "metadata.json"),
        "bundle_scores_sha256": sha256_file(bundle_dir / "scores.jsonl"),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    pre_qrels_path = output_dir / "pre_qrels_audit.json"
    _write_json(pre_qrels_path, pre_qrels)

    parent_manifest = _load_manifest(DEEP_DIVE_MANIFEST_PATH)
    qrels_path = standardized_dir / "qrels_dev.jsonl"
    if sha256_file(qrels_path) != parent_manifest["frozen_inputs"]["qrels_dev_sha256"]:
        raise ValueError("N20 qrels hash mismatch")
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
    eligibility = audited["eligibility"]
    baseline_full = audited["scores"]["baseline_full"]
    baseline_null = audited["scores"]["baseline_null"]
    mode_specs = {
        "cache_rebuild": ("full_cache_rebuild", "null_cache_rebuild"),
        "zero_prefix": ("full_zero_prefix", "null_zero_prefix"),
        "no_cache_rebuild": ("full_no_cache_rebuild", "null_no_cache_rebuild"),
        "wrong_user_prefix": ("full_wrong_user_prefix", "baseline_null"),
    }
    family_rows: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    per_request: dict[str, np.ndarray] = {}
    for mode, (full_name, null_name) in mode_specs.items():
        full_values = audited["scores"][full_name]
        null_values = audited["scores"][null_name]
        contrasts = {
            "full_operator_delta": _metric_delta(records, candidates, gains, full_values, baseline_full),
            "null_operator_delta": _metric_delta(records, candidates, gains, null_values, baseline_null),
            "transfer_gap_change": _metric_gap_change(
                records, candidates, gains, full_values, null_values, baseline_full, baseline_null
            ),
        }
        mode_results: dict[str, Any] = {}
        for endpoint in ("target_margin", "ndcg@10"):
            endpoint_results: dict[str, Any] = {}
            for contrast_name, values in contrasts.items():
                rows = []
                for fold_name, fold_mask in (
                    ("all", np.ones(len(records), dtype=bool)),
                    ("0", folds == 0),
                    ("1", folds == 1),
                ):
                    mask = strict & eligibility & fold_mask & np.isfinite(values[endpoint])
                    inferred = cluster_mean_inference(values[endpoint][mask], clusters[mask])
                    rows.append({
                        "surface": STRICT_TRANSFER_SURFACE,
                        "eligibility": "frozen_wrong_user_or_content_neutral_eligible",
                        "normalized_query_fold": fold_name,
                        **inferred,
                    })
                all_row = next(row for row in rows if row["normalized_query_fold"] == "all")
                family_rows.append({
                    "mode": mode,
                    "endpoint": endpoint,
                    "contrast": contrast_name,
                    "two_sided_p": float(all_row["two_sided_p"]),
                })
                endpoint_results[contrast_name] = {"registered": rows}
                per_request[f"{mode}__{endpoint}__{contrast_name}"] = values[endpoint]
            mode_results[endpoint] = endpoint_results
        results[mode] = mode_results

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
        "analysis_type": "transformer_deep_dive_n20_q1_cache_phase",
        "analysis_run_id": analysis_run_id,
        "primary_surface": STRICT_TRANSFER_SURFACE,
        "registered_eligibility": "frozen_wrong_user_eligible",
        "implementation_digest": implementation["digest"],
        "eligible_requests": int(eligibility.sum()),
        "strict_transfer_requests": int(strict.sum()),
        "strict_transfer_eligible_requests": int((strict & eligibility).sum()),
        "bootstrap": {"cluster": "normalized_query", "samples": 5000, "seed": 20260715},
        "multiple_testing": {"family": "mode_x_endpoint_x_contrast", "family_size": len(family_rows), "method": "benjamini_hochberg"},
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
    metrics["metrics_path"] = str(metrics_path)
    _write_json(metrics_path, metrics)
    _append_dev_eval_log(Path(dev_eval_log_path), metrics, analysis_run_id, metrics_path)
    return metrics


def _audit_bundle(root: Path, records: Sequence[Any]) -> dict[str, Any]:
    metadata = _read_json(root / "metadata.json")
    if (
        metadata.get("status") != "completed"
        or metadata.get("result_eligible") is not True
        or metadata.get("qrels_read") is not False
        or metadata.get("source_test_opened") is not False
        or metadata.get("complete_finite_score_coverage") is not True
        or metadata.get("identity_passed") is not True
        or metadata.get("method_id") != "q1_instructrec_generalqwen"
        or tuple(metadata.get("score_conditions", ())) != N20_CONDITIONS
        or float(metadata.get("maximum_identity_delta", math.inf)) > 1.0e-5
        or float(metadata.get("maximum_frozen_baseline_delta", math.inf)) > 1.0e-5
    ):
        raise ValueError("N20 bundle metadata failed integrity")
    scores_path = root / "scores.jsonl"
    if metadata.get("scores_sha256") != sha256_file(scores_path):
        raise ValueError("N20 scores hash differs")
    observed = audit_scalar_partial(scores_path, records, N20_CONDITIONS)
    if observed["completed_requests"] != len(records):
        raise ValueError("N20 bundle has incomplete coverage")
    scores = {name: {} for name in N20_CONDITIONS}
    eligibility = np.zeros(len(records), dtype=bool)
    for ordinal, block in enumerate(iter_jsonl(scores_path)):
        eligibility[ordinal] = bool(block.get("content_control_eligible"))
        if eligibility[ordinal]:
            audit = block.get("phase_audit", {})
            if audit.get("token_position_integrity") is not True or audit.get("cache_key_integrity") is not True:
                raise ValueError("N20 phase integrity flag failed")
        request_id = records[ordinal].request_id
        for row in block["rows"]:
            item_id = str(row["candidate_item_id"])
            for name in N20_CONDITIONS:
                scores[name].setdefault(request_id, {})[item_id] = float(row["conditions"][name])
    if int(eligibility.sum()) != 7254:
        raise ValueError("N20 eligibility count drift")
    for full_name, base in (("full_cache_identity", "baseline_full"), ("null_cache_identity", "baseline_null")):
        delta = _max_map_delta(scores[full_name], scores[base])
        if delta > 1.0e-5:
            raise ValueError(f"N20 {full_name} exceeds identity tolerance: {delta}")
    implementation = q1_cache_phase_runtime_implementation_identity()
    if metadata.get("implementation_identity", {}).get("digest") != implementation["digest"]:
        raise ValueError("N20 implementation digest drift")
    return {"metadata": metadata, "scores": scores, "eligibility": eligibility}


def _metric_delta(records: Sequence[Any], candidates: Mapping[str, Sequence[str]], gains: Mapping[str, Mapping[str, float]], left: Mapping[str, Mapping[str, float]], right: Mapping[str, Mapping[str, float]]) -> dict[str, np.ndarray]:
    request_ids = [record.request_id for record in records]
    left_margin = _target_margins(request_ids, candidates, gains, left)
    right_margin = _target_margins(request_ids, candidates, gains, right)
    left_ndcg = _ndcg(request_ids, candidates, gains, left)
    right_ndcg = _ndcg(request_ids, candidates, gains, right)
    return {"target_margin": left_margin - right_margin, "ndcg@10": left_ndcg - right_ndcg}


def _metric_gap_change(records: Sequence[Any], candidates: Mapping[str, Sequence[str]], gains: Mapping[str, Mapping[str, float]], full: Mapping[str, Mapping[str, float]], null: Mapping[str, Mapping[str, float]], baseline_full: Mapping[str, Mapping[str, float]], baseline_null: Mapping[str, Mapping[str, float]]) -> dict[str, np.ndarray]:
    current = _metric_delta(records, candidates, gains, full, null)
    baseline = _metric_delta(records, candidates, gains, baseline_full, baseline_null)
    return {name: current[name] - baseline[name] for name in current}


def _ndcg(request_ids: Sequence[str], candidates: Mapping[str, Sequence[str]], gains: Mapping[str, Mapping[str, float]], scores: Mapping[str, Mapping[str, float]]) -> np.ndarray:
    values = np.zeros(len(request_ids), dtype=np.float64)
    for ordinal, request_id in enumerate(request_ids):
        item_ids = list(candidates[request_id])
        values[ordinal] = gain_ndcg_at_k(
            request_id,
            item_ids,
            [float(scores[request_id][item_id]) for item_id in item_ids],
            [float(gains[request_id].get(item_id, 0.0)) for item_id in item_ids],
            10,
        )
    return values


def _max_map_delta(left: Mapping[str, Mapping[str, float]], right: Mapping[str, Mapping[str, float]]) -> float:
    maximum = 0.0
    for request_id, values in left.items():
        for item_id, value in values.items():
            maximum = max(maximum, abs(float(value) - float(right[request_id][item_id])))
    return maximum


def _append_dev_eval_log(path: Path, metrics: Mapping[str, Any], analysis_run_id: str, metrics_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "analysis_type": metrics["analysis_type"],
        "run_id": analysis_run_id,
        "method_ids": ["q1_instructrec_generalqwen"],
        "split": "dev",
        "qrels_sha256": metrics["qrels_dev_sha256"],
        "metrics_path": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
