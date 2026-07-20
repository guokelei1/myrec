"""Shared qrels-gated evaluator for N9 history formation/transport paths."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from myrec.baselines.motivation_v12_contracts import sanitize_record_for_model
from myrec.mechanism.attention_edge_evaluator import (
    _append_jsonl,
    _load_bundle_frozen_baseline,
    _load_bundle_content_control_eligibility,
)
from myrec.mechanism.deep_dive_native_evaluator import benjamini_hochberg, cluster_mean_inference
from myrec.mechanism.fold_qrels import audit_fold_qrels
from myrec.mechanism.history_path_scoring import N9_SCORE_CONDITIONS
from myrec.mechanism.postblock_sweep_evaluator import _load_fold_qrels, _ndcg, _strict_transfer_mask, _target_margins
from myrec.mechanism.representation_evaluator import _audit_candidate_and_request_manifests
from myrec.mechanism.representation_probe import normalize_query, normalized_query_fold
from myrec.mechanism.scalar_condition_bundle import audit_scalar_partial
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl


METHODS = ("q2_recranker_generalqwen", "q3_tallrec_generalqwen")
BLOCKS = (13, 20, 27)
ENDPOINTS = ("target_margin", "ndcg@10")
REGISTERED_CONTRASTS = (
    "formation_transport_joint_minus_additive_logits",
    "formation_transport_joint_minus_full",
)
FAMILY_SIZE = len(METHODS) * len(BLOCKS) * len(ENDPOINTS) * len(REGISTERED_CONTRASTS)


def evaluate_history_path_bundles(
    standardized_dir: str | Path,
    qrels_split_dir: str | Path,
    model_bundles: Mapping[str, Mapping[int, str | Path]],
    output_dir: str | Path,
    analysis_run_id: str,
    *,
    dev_eval_log_path: str | Path = "reports/dev_eval_log.jsonl",
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Evaluate exactly six completed N9 bundles after score integrity checks."""

    if set(model_bundles) != set(METHODS) or any(set(map(int, blocks)) != set(BLOCKS) for blocks in model_bundles.values()):
        raise ValueError("N9 evaluator requires Q2/Q3 x blocks 13/20/27")
    standardized_dir = Path(standardized_dir)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"N9 evaluation output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_records = list(iter_jsonl(standardized_dir / "records_dev.jsonl"))
    all_records = [sanitize_record_for_model(row) for row in raw_records]
    if len(all_records) != 8000:
        raise ValueError("N9 evaluator requires frozen 8000-request dev")
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

    bundle_scores: dict[str, dict[int, dict[str, dict[str, dict[str, float]]]]] = {}
    eligibility: dict[str, dict[int, np.ndarray]] = {}
    identities: dict[str, dict[str, Any]] = {}
    implementation_digests: set[str] = set()
    for method_id in METHODS:
        bundle_scores[method_id] = {}
        eligibility[method_id] = {}
        identities[method_id] = {}
        for block in BLOCKS:
            root = Path(model_bundles[method_id][block])
            metadata = _read_json(root / "metadata.json")
            expected = {
                "analysis_stage": "transformer_n9_history_path",
                "status": "completed",
                "result_eligible": True,
                "identity_passed": True,
                "complete_finite_score_coverage": True,
                "qrels_read": False,
                "source_test_opened": False,
                "method_id": method_id,
                "block_zero_based": block,
                "normalized_query_fold": 1,
                "score_conditions": list(N9_SCORE_CONDITIONS),
            }
            for key, value in expected.items():
                if metadata.get(key) != value:
                    raise ValueError(f"N9 metadata mismatch: {method_id}:b{block}:{key}")
            for key in ("maximum_identity_delta", "maximum_full_baseline_delta", "maximum_null_baseline_delta"):
                if float(metadata.get(key, math.inf)) > 1.0e-5:
                    raise ValueError(f"N9 identity gate failed: {method_id}:b{block}:{key}")
            scores_path = root / "scores.jsonl"
            if metadata.get("scores_sha256") != sha256_file(scores_path):
                raise ValueError(f"N9 scores hash mismatch: {method_id}:b{block}")
            implementation = str(metadata.get("implementation_identity", {}).get("digest") or "")
            if not implementation:
                raise ValueError("N9 implementation digest is missing")
            implementation_digests.add(implementation)
            observed = audit_scalar_partial(scores_path, records, N9_SCORE_CONDITIONS)
            if observed["completed_requests"] != len(records):
                raise ValueError(f"N9 request coverage mismatch: {method_id}:b{block}")
            eligible_map = _load_bundle_content_control_eligibility(
                metadata,
                records,
                label=f"N9 {method_id}:b{block}",
                identity_key="content_neutral_control",
            )
            eligibility[method_id][block] = np.asarray(
                [eligible_map[record.request_id] for record in records], dtype=bool
            )
            full_fallback = _load_bundle_frozen_baseline(
                metadata, records, label=f"N9 {method_id}:b{block} full fallback"
            )
            null_fallback = _load_bundle_frozen_baseline(
                metadata,
                records,
                label=f"N9 {method_id}:b{block} null fallback",
                identity_key="frozen_null_baseline",
            )
            scores = {condition: {} for condition in N9_SCORE_CONDITIONS}
            for block_row in iter_jsonl(scores_path):
                request_id = str(block_row["request_id"])
                if not eligible_map[request_id]:
                    for row in block_row["rows"]:
                        key = (request_id, str(row["candidate_item_id"]))
                        values = row["conditions"]
                        for condition in N9_SCORE_CONDITIONS:
                            expected = null_fallback[key] if condition == "baseline_null" else full_fallback[key]
                            if float(values[condition]) != float(expected):
                                raise ValueError(f"N9 ineligible fallback drift: {method_id}:b{block}:{request_id}:{condition}")
                for condition in N9_SCORE_CONDITIONS:
                    scores[condition][request_id] = {
                        str(row["candidate_item_id"]): float(row["conditions"][condition])
                        for row in block_row["rows"]
                    }
            bundle_scores[method_id][block] = scores
            identities[method_id][str(block)] = {
                "path": str(root),
                "metadata_sha256": sha256_file(root / "metadata.json"),
                "scores_sha256": sha256_file(scores_path),
            }
    if len(implementation_digests) != 1:
        raise ValueError("N9 bundles do not share one implementation digest")
    for method_id in METHODS:
        if not np.array_equal(eligibility[method_id][13], eligibility[method_id][20]) or not np.array_equal(eligibility[method_id][13], eligibility[method_id][27]):
            raise ValueError(f"N9 eligibility differs across blocks: {method_id}")
        if int(eligibility[method_id][13].sum()) != 7254:
            raise ValueError(f"N9 eligible request count differs: {method_id}")
    if not np.array_equal(eligibility[METHODS[0]][13], eligibility[METHODS[1]][13]):
        raise ValueError("N9 eligibility differs across Q2/Q3")
    frozen_eligible = eligibility[METHODS[0]][13]

    family_rows: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    per_request: dict[str, np.ndarray] = {}
    for method_id in METHODS:
        results[method_id] = {}
        for block in BLOCKS:
            scores = bundle_scores[method_id][block]
            margins = {condition: _target_margins(request_ids, candidates, gains, scores[condition]) for condition in N9_SCORE_CONDITIONS}
            ndcgs = {condition: _ndcg(request_ids, candidates, gains, scores[condition]) for condition in N9_SCORE_CONDITIONS}
            endpoint_values = {
                "target_margin": {
                    "formation_transport_joint_minus_additive_logits": margins["formation_transport_joint"] - margins["formation_logits_mask"] - margins["transport_logits_mask"] + margins["baseline_full"],
                    "formation_transport_joint_minus_full": margins["formation_transport_joint"] - margins["baseline_full"],
                },
                "ndcg@10": {
                    "formation_transport_joint_minus_additive_logits": ndcgs["formation_transport_joint"] - ndcgs["formation_logits_mask"] - ndcgs["transport_logits_mask"] + ndcgs["baseline_full"],
                    "formation_transport_joint_minus_full": ndcgs["formation_transport_joint"] - ndcgs["baseline_full"],
                },
            }
            block_result: dict[str, Any] = {"registered_contrasts": {}, "descriptive_effects": {}}
            for condition in (
                "formation_logits_mask",
                "formation_value_zero",
                "transport_logits_mask",
                "transport_value_zero",
                "formation_transport_joint",
            ):
                block_result["descriptive_effects"][condition] = {}
                for endpoint, arrays in (("target_margin", margins), ("ndcg@10", ndcgs)):
                    values = arrays[condition] - arrays["baseline_full"]
                    mask = strict & frozen_eligible & np.isfinite(values)
                    block_result["descriptive_effects"][condition][endpoint] = {
                        **cluster_mean_inference(values[mask], clusters[mask]),
                        "surface": "strict_transfer",
                        "eligibility": "frozen_content_neutral_eligible",
                    }
                    per_request[f"{method_id}__b{block}__{condition}__{endpoint}"] = values
            for endpoint in ENDPOINTS:
                block_result["registered_contrasts"][endpoint] = {}
                for contrast in REGISTERED_CONTRASTS:
                    values = endpoint_values[endpoint][contrast]
                    mask = strict & frozen_eligible & np.isfinite(values)
                    inference = {
                        **cluster_mean_inference(values[mask], clusters[mask]),
                        "surface": "strict_transfer",
                        "eligibility": "frozen_content_neutral_eligible",
                        "contrast": contrast,
                        "endpoint": endpoint,
                        "status": "completed",
                    }
                    block_result["registered_contrasts"][endpoint][contrast] = inference
                    family_rows.append({
                        "method_id": method_id,
                        "block_zero_based": block,
                        "endpoint": endpoint,
                        "contrast": contrast,
                        "two_sided_p": float(inference["two_sided_p"]),
                    })
                    per_request[f"{method_id}__b{block}__{contrast}__{endpoint}"] = values
            results[method_id][str(block)] = block_result
    if len(family_rows) != FAMILY_SIZE:
        raise AssertionError(f"N9 family size differs: {len(family_rows)} != {FAMILY_SIZE}")
    q_values = benjamini_hochberg([row["two_sided_p"] for row in family_rows])
    for row, q_value in zip(family_rows, q_values):
        row["bh_q"] = float(q_value)
        results[row["method_id"]][str(row["block_zero_based"])] ["registered_contrasts"][row["endpoint"]][row["contrast"]]["bh_q"] = float(q_value)

    per_request_path = output_dir / "per_request_contrasts.npz"
    np.savez(
        per_request_path,
        **per_request,
        request_ids=np.asarray(request_ids, dtype=np.str_),
        normalized_queries=clusters,
        strict_mask=strict,
        frozen_eligible_mask=frozen_eligible,
    )
    metrics = {
        "schema_version": 1,
        "analysis_type": "transformer_n9_history_path",
        "analysis_run_id": analysis_run_id,
        "methods": list(METHODS),
        "blocks": list(BLOCKS),
        "conditions": list(N9_SCORE_CONDITIONS),
        "normalized_query_fold": 1,
        "strict_transfer_requests": int(strict.sum()),
        "strict_transfer_eligible_requests": int((strict & frozen_eligible).sum()),
        "eligible_requests": int(frozen_eligible.sum()),
        "bootstrap": {"cluster": "normalized_query", "samples": 5000, "seed": 20260715},
        "family_policy": {"units": FAMILY_SIZE, "multiple_testing": "benjamini_hochberg"},
        "family_rows": family_rows,
        "results": results,
        "input_bundles": identities,
        "implementation_digest": next(iter(implementation_digests)),
        "qrels_read": True,
        "qrels_opened_only_after_score_integrity": True,
        "qrels_fold_opened": 1,
        "qrels_fold_sha256": sha256_file(qrels_path),
        "qrels_source_sha256": qrels_manifest["source_qrels_sha256"],
        "per_request_contrasts_path": str(per_request_path),
        "per_request_contrasts_sha256": sha256_file(per_request_path),
        "source_test_opened": False,
        "status": "completed",
        "claim_boundary": {
            "highest_authorized_claim": "history_formation_transport_candidate_readout_path_diagnostic",
            "edge_sensitivity_is_not_operator_necessity": True,
            "transfer_architecture_authorized": False,
        },
        "command": list(command or []),
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _append_jsonl(
        Path(dev_eval_log_path),
        {
            "analysis_type": metrics["analysis_type"],
            "run_id": analysis_run_id,
            "method_ids": list(METHODS),
            "split": "dev_fold1",
            "qrels_sha256": metrics["qrels_fold_sha256"],
            "metrics_path": str(metrics_path),
            "metrics_sha256": sha256_file(metrics_path),
        },
    )
    return metrics


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value
