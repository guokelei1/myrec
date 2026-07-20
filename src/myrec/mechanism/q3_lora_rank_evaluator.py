"""Shared qrels-gated evaluator for N10 Q3 LoRA rank paths."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from myrec.baselines.motivation_v12_contracts import sanitize_record_for_model
from myrec.mechanism.attention_edge_evaluator import _append_jsonl, _load_bundle_content_control_eligibility, _load_bundle_frozen_baseline
from myrec.mechanism.deep_dive_native_evaluator import benjamini_hochberg, cluster_mean_inference
from myrec.mechanism.fold_qrels import audit_fold_qrels
from myrec.mechanism.postblock_sweep_evaluator import _load_fold_qrels, _ndcg, _strict_transfer_mask, _target_margins
from myrec.mechanism.q3_lora_rank_scoring import LORA_PATH_CONDITIONS
from myrec.mechanism.representation_evaluator import _audit_candidate_and_request_manifests
from myrec.mechanism.representation_probe import normalize_query, normalized_query_fold
from myrec.mechanism.scalar_condition_bundle import audit_scalar_partial
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl


RANK_CONTRASTS = (
    "no_adapter_identity_minus_full",
    *(f"outer_product_rank_{rank}_minus_full" for rank in range(8)),
)
ENDPOINTS = ("target_margin", "ndcg@10")
FAMILY_SIZE = len(RANK_CONTRASTS) * len(ENDPOINTS)


def evaluate_q3_lora_rank_bundle(
    standardized_dir: str | Path,
    qrels_split_dir: str | Path,
    bundle_dir: str | Path,
    output_dir: str | Path,
    analysis_run_id: str,
    *,
    dev_eval_log_path: str | Path = "reports/dev_eval_log.jsonl",
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    standardized_dir = Path(standardized_dir)
    root = Path(bundle_dir)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"N10 rank evaluation output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_records = list(iter_jsonl(standardized_dir / "records_dev.jsonl"))
    all_records = [sanitize_record_for_model(row) for row in raw_records]
    if len(all_records) != 8000:
        raise ValueError("N10 rank evaluator requires 8000 requests")
    all_candidates = _audit_candidate_and_request_manifests(
        standardized_dir / "candidate_manifest.json",
        standardized_dir / "request_manifest.json",
        all_records,
        raw_records,
    )
    records = [record for record in all_records if normalized_query_fold(record.query) == 1]
    candidates = {record.request_id: all_candidates[record.request_id] for record in records}
    qrels_path, qrels_manifest = audit_fold_qrels(standardized_dir, qrels_split_dir, 1)
    gains = _load_fold_qrels(qrels_path, candidates)
    strict = _strict_transfer_mask(records, candidates, gains)
    request_ids = [record.request_id for record in records]
    clusters = np.asarray([normalize_query(record.query) for record in records], dtype=np.str_)
    metadata = _read_json(root / "metadata.json")
    expected = {
        "analysis_stage": "transformer_n10_q3_lora_rank_path",
        "status": "completed",
        "result_eligible": True,
        "identity_passed": True,
        "complete_finite_score_coverage": True,
        "qrels_read": False,
        "source_test_opened": False,
        "method_id": "q3_tallrec_generalqwen",
        "score_conditions": list(LORA_PATH_CONDITIONS),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"N10 rank metadata mismatch: {key}")
    for key in ("maximum_full_baseline_delta", "maximum_null_baseline_delta"):
        if float(metadata.get(key, math.inf)) > 1.0e-5:
            raise ValueError(f"N10 rank identity gate failed: {key}")
    scores_path = root / "scores.jsonl"
    if metadata.get("scores_sha256") != sha256_file(scores_path):
        raise ValueError("N10 rank scores hash mismatch")
    observed = audit_scalar_partial(scores_path, records, LORA_PATH_CONDITIONS)
    if observed["completed_requests"] != len(records):
        raise ValueError("N10 rank request coverage mismatch")
    eligible_map = _load_bundle_content_control_eligibility(
        metadata, records, label="N10 Q3 LoRA rank", identity_key="content_neutral_control"
    )
    full_fallback = _load_bundle_frozen_baseline(metadata, records, label="N10 rank full fallback")
    null_fallback = _load_bundle_frozen_baseline(metadata, records, label="N10 rank null fallback", identity_key="frozen_null_baseline")
    scores = {condition: {} for condition in LORA_PATH_CONDITIONS}
    for block_row in iter_jsonl(scores_path):
        request_id = str(block_row["request_id"])
        if not eligible_map[request_id]:
            for row in block_row["rows"]:
                key = (request_id, str(row["candidate_item_id"]))
                for condition in LORA_PATH_CONDITIONS:
                    expected_value = null_fallback[key] if condition == "baseline_null" else full_fallback[key]
                    if float(row["conditions"][condition]) != float(expected_value):
                        raise ValueError(f"N10 rank ineligible fallback drift: {request_id}:{condition}")
        for condition in LORA_PATH_CONDITIONS:
            scores[condition][request_id] = {
                str(row["candidate_item_id"]): float(row["conditions"][condition]) for row in block_row["rows"]
            }
    margins = {condition: _target_margins(request_ids, candidates, gains, scores[condition]) for condition in LORA_PATH_CONDITIONS}
    ndcgs = {condition: _ndcg(request_ids, candidates, gains, scores[condition]) for condition in LORA_PATH_CONDITIONS}
    frozen_eligible = np.asarray([eligible_map[request_id] for request_id in request_ids], dtype=bool)
    family_rows: list[dict[str, Any]] = []
    results: dict[str, Any] = {"registered_contrasts": {}, "descriptive_effects": {}}
    per_request: dict[str, np.ndarray] = {}
    for condition in ("a_only", "b_only", "no_adapter_identity", *(f"outer_product_rank_{rank}" for rank in range(8))):
        results["descriptive_effects"][condition] = {}
        for endpoint, arrays in (("target_margin", margins), ("ndcg@10", ndcgs)):
            values = arrays[condition] - arrays["baseline_full"]
            mask = strict & frozen_eligible & np.isfinite(values)
            results["descriptive_effects"][condition][endpoint] = {
                **cluster_mean_inference(values[mask], clusters[mask]),
                "surface": "strict_transfer",
                "eligibility": "frozen_content_neutral_eligible",
            }
            per_request[f"{condition}__{endpoint}"] = values
    for endpoint, arrays in (("target_margin", margins), ("ndcg@10", ndcgs)):
        results["registered_contrasts"][endpoint] = {}
        for contrast in RANK_CONTRASTS:
            condition = "no_adapter_identity" if contrast.startswith("no_adapter") else contrast.split("_minus_full")[0]
            values = arrays[condition] - arrays["baseline_full"]
            mask = strict & frozen_eligible & np.isfinite(values)
            inference = {**cluster_mean_inference(values[mask], clusters[mask]), "contrast": contrast, "endpoint": endpoint, "surface": "strict_transfer", "eligibility": "frozen_content_neutral_eligible", "status": "completed"}
            results["registered_contrasts"][endpoint][contrast] = inference
            family_rows.append({"contrast": contrast, "endpoint": endpoint, "two_sided_p": float(inference["two_sided_p"])})
            per_request[f"registered__{contrast}__{endpoint}"] = values
    if len(family_rows) != FAMILY_SIZE:
        raise AssertionError("N10 rank family size differs")
    q_values = benjamini_hochberg([row["two_sided_p"] for row in family_rows])
    for row, q_value in zip(family_rows, q_values):
        row["bh_q"] = float(q_value)
        results["registered_contrasts"][row["endpoint"]][row["contrast"]]["bh_q"] = float(q_value)
    per_request_path = output_dir / "per_request_contrasts.npz"
    np.savez(per_request_path, **per_request, request_ids=np.asarray(request_ids, dtype=np.str_), normalized_queries=clusters, strict_mask=strict, frozen_eligible_mask=frozen_eligible)
    metrics = {
        "schema_version": 1,
        "analysis_type": "transformer_n10_q3_lora_rank_path",
        "analysis_run_id": analysis_run_id,
        "method_id": "q3_tallrec_generalqwen",
        "conditions": list(LORA_PATH_CONDITIONS),
        "registered_contrasts": list(RANK_CONTRASTS),
        "strict_transfer_requests": int(strict.sum()),
        "strict_transfer_eligible_requests": int((strict & frozen_eligible).sum()),
        "eligible_requests": int(frozen_eligible.sum()),
        "family_policy": {"units": FAMILY_SIZE, "multiple_testing": "benjamini_hochberg"},
        "bootstrap": {"cluster": "normalized_query", "samples": 5000, "seed": 20260715},
        "family_rows": family_rows,
        "results": results,
        "input_bundle": {"path": str(root), "metadata_sha256": sha256_file(root / "metadata.json"), "scores_sha256": sha256_file(scores_path)},
        "qrels_read": True,
        "qrels_opened_only_after_score_integrity": True,
        "qrels_fold_opened": 1,
        "qrels_fold_sha256": sha256_file(qrels_path),
        "qrels_source_sha256": qrels_manifest["source_qrels_sha256"],
        "per_request_contrasts_path": str(per_request_path),
        "per_request_contrasts_sha256": sha256_file(per_request_path),
        "source_test_opened": False,
        "status": "completed",
        "claim_boundary": {"highest_authorized_claim": "q3_lora_parameterization_path_diagnostic", "rank_path_is_not_architecture_necessity": True, "transfer_architecture_authorized": False},
        "command": list(command or []),
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _append_jsonl(Path(dev_eval_log_path), {"analysis_type": metrics["analysis_type"], "run_id": analysis_run_id, "method_ids": ["q3_tallrec_generalqwen"], "split": "dev_fold1", "qrels_sha256": metrics["qrels_fold_sha256"], "metrics_path": str(metrics_path), "metrics_sha256": sha256_file(metrics_path)})
    return metrics


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value

