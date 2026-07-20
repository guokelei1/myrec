"""Qrels-gated evaluator for one fixed N19 Q3 LoRA branch bundle."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from myrec.baselines.motivation_v12_contracts import sanitize_record_for_model
from myrec.eval.target_aware_surfaces import build_target_aware_surface_memberships
from myrec.mechanism.attention_edge_runtime import DEEP_DIVE_MANIFEST_PATH, _load_manifest
from myrec.mechanism.deep_dive_native_evaluator import benjamini_hochberg, cluster_mean_inference
from myrec.mechanism.q3_lora_branch_scoring import LORA_BRANCH_CONDITIONS
from myrec.mechanism.representation_evaluator import (
    STRICT_TRANSFER_SURFACE,
    _audit_candidate_and_request_manifests,
    _load_dev_qrels,
)
from myrec.mechanism.representation_probe import normalize_query, normalized_query_fold
from myrec.mechanism.scalar_condition_bundle import audit_scalar_partial
from myrec.mechanism.qkv_projection_evaluator import _metric_delta, _write_json, _append_jsonl
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl


def evaluate_q3_lora_branch_bundle(
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
        raise FileExistsError(f"N19 evaluation output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_records = list(iter_jsonl(standardized_dir / "records_dev.jsonl"))
    records = [sanitize_record_for_model(row) for row in raw_records]
    if len(records) != 8000:
        raise ValueError("N19 evaluator requires all 8000 requests")
    candidates = _audit_candidate_and_request_manifests(
        standardized_dir / "candidate_manifest.json",
        standardized_dir / "request_manifest.json",
        records,
        raw_records,
    )
    metadata = json.loads((bundle_dir / "metadata.json").read_text(encoding="utf-8"))
    _audit_metadata(metadata)
    scores_path = bundle_dir / "scores.jsonl"
    if metadata.get("scores_sha256") != sha256_file(scores_path):
        raise ValueError("N19 score hash mismatch")
    observed = audit_scalar_partial(scores_path, records, LORA_BRANCH_CONDITIONS)
    if observed["completed_requests"] != len(records):
        raise ValueError("N19 request coverage mismatch")
    width = len(records[0].candidates)
    scores = {name: np.zeros((len(records), width), dtype=np.float64) for name in LORA_BRANCH_CONDITIONS}
    eligibility = np.zeros(len(records), dtype=bool)
    for ordinal, block_row in enumerate(iter_jsonl(scores_path)):
        eligibility[ordinal] = bool(block_row.get("content_control_eligible"))
        for candidate_ordinal, row in enumerate(block_row["rows"]):
            for name in LORA_BRANCH_CONDITIONS:
                scores[name][ordinal, candidate_ordinal] = float(row["conditions"][name])
    if int(eligibility.sum()) != 7254:
        raise ValueError("N19 eligibility count drift")
    for path_kind in ("full", "null"):
        identity = f"{path_kind}_lora_identity"
        baseline = f"baseline_{path_kind}"
        if float(np.max(np.abs(scores[identity] - scores[baseline]))) > 1.0e-5:
            raise ValueError(f"N19 {identity} identity gate failed")

    parent_manifest = _load_manifest(DEEP_DIVE_MANIFEST_PATH)
    qrels_path = standardized_dir / "qrels_dev.jsonl"
    if sha256_file(qrels_path) != parent_manifest["frozen_inputs"]["qrels_dev_sha256"]:
        raise ValueError("N19 qrels hash mismatch")
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
        name[len("full_lora_") :]
        for name in LORA_BRANCH_CONDITIONS
        if name.startswith("full_lora_") and not name.endswith("identity")
    ]
    family_rows: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    per_request: dict[str, np.ndarray] = {}
    for mode in active:
        full_name = f"full_lora_{mode}"
        null_name = f"null_lora_{mode}"
        full_delta = _metric_delta(records, candidates, gains, scores[full_name], scores["baseline_full"])
        null_delta = _metric_delta(records, candidates, gains, scores[null_name], scores["baseline_null"])
        baseline_gap = _metric_delta(records, candidates, gains, scores["baseline_full"], scores["baseline_null"])
        operator_gap = _metric_delta(records, candidates, gains, scores[full_name], scores[null_name])
        mode_result: dict[str, Any] = {}
        for endpoint in ("target_margin", "ndcg@10"):
            contrasts = {
                "full_operator_delta": full_delta[endpoint],
                "null_operator_delta": null_delta[endpoint],
                "transfer_gap_change": operator_gap[endpoint] - baseline_gap[endpoint],
            }
            endpoint_result: dict[str, Any] = {}
            for contrast, values in contrasts.items():
                registered = []
                for fold_name, fold_mask in (("all", np.ones(len(records), dtype=bool)), ("0", folds == 0), ("1", folds == 1)):
                    mask = strict & eligibility & fold_mask & np.isfinite(values)
                    registered.append({
                        "surface": STRICT_TRANSFER_SURFACE,
                        "eligibility": "frozen_content_control_eligible",
                        "normalized_query_fold": fold_name,
                        **cluster_mean_inference(values[mask], clusters[mask]),
                    })
                endpoint_result[contrast] = {"registered": registered}
                row = next(item for item in registered if item["normalized_query_fold"] == "all")
                family_rows.append({
                    "mode": mode,
                    "endpoint": endpoint,
                    "contrast": contrast,
                    "two_sided_p": float(row["two_sided_p"]),
                })
                per_request[f"{mode}__{endpoint}__{contrast}"] = values
            mode_result[endpoint] = endpoint_result
        results[mode] = mode_result
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
        "analysis_type": "n19_lora_branch",
        "analysis_run_id": analysis_run_id,
        "method_id": "q3_tallrec_generalqwen",
        "block_zero_based": metadata["block_zero_based"],
        "component": metadata["component"],
        "primary_surface": STRICT_TRANSFER_SURFACE,
        "eligible_requests": int(eligibility.sum()),
        "strict_transfer_requests": int(strict.sum()),
        "strict_transfer_eligible_requests": int((strict & eligibility).sum()),
        "bootstrap": {"cluster": "normalized_query", "samples": 5000, "seed": 20260715},
        "multiple_testing": {"family_size": len(family_rows), "method": "benjamini_hochberg"},
        "family_rows": family_rows,
        "results": results,
        "input_bundle": {"path": str(bundle_dir), "metadata_sha256": sha256_file(bundle_dir / "metadata.json"), "scores_sha256": sha256_file(scores_path)},
        "qrels_read": True,
        "qrels_opened_only_after_score_integrity": True,
        "qrels_dev_sha256": sha256_file(qrels_path),
        "per_request_contrasts_path": str(per_request_path),
        "per_request_contrasts_sha256": sha256_file(per_request_path),
        "source_test_opened": False,
        "status": "completed",
        "claim_boundary": {"highest_authorized_claim": "q3_adapter_branch_diagnostic", "architecture_authorized": False},
        "command": list(command or []),
    }
    metrics_path = output_dir / "metrics.json"
    _write_json(metrics_path, metrics)
    _append_jsonl(Path(dev_eval_log_path), {
        "analysis_type": metrics["analysis_type"],
        "run_id": analysis_run_id,
        "method_ids": ["q3_tallrec_generalqwen"],
        "split": "dev",
        "qrels_sha256": metrics["qrels_dev_sha256"],
        "metrics_path": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
    })
    return metrics


def _audit_metadata(metadata: dict[str, Any]) -> None:
    expected = {
        "analysis_stage": "n19_lora_branch",
        "status": "completed",
        "result_eligible": True,
        "qrels_read": False,
        "source_test_opened": False,
        "complete_finite_score_coverage": True,
        "identity_passed": True,
        "method_id": "q3_tallrec_generalqwen",
        "score_conditions": list(LORA_BRANCH_CONDITIONS),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"N19 metadata mismatch: {key}")
    if float(metadata.get("maximum_identity_delta", math.inf)) > 1.0e-5:
        raise ValueError("N19 identity gate failed")
