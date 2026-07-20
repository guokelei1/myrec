"""Shared qrels-gated evaluator for D5 fixed-length contextual controls."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from myrec.baselines.motivation_v12_contracts import sanitize_record_for_model
from myrec.eval.target_aware_surfaces import build_target_aware_surface_memberships
from myrec.mechanism.attention_edge_evaluator import (
    _audit_ineligible_frozen_conditions,
    _append_jsonl,
    _load_bundle_content_control_eligibility,
    _load_bundle_frozen_baseline,
    _ndcg_values,
    _write_json,
)
from myrec.mechanism.attention_edge_runtime import SUPPORTED_METHODS
from myrec.mechanism.contextual_control_scoring import CONTEXTUAL_SCORE_CONDITIONS
from myrec.mechanism.deep_dive_native_evaluator import (
    benjamini_hochberg,
    cluster_mean_inference,
)
from myrec.mechanism.patch_evaluator import _target_margins
from myrec.mechanism.representation_evaluator import (
    STRICT_TRANSFER_SURFACE,
    _audit_candidate_and_request_manifests,
    _load_dev_qrels,
)
from myrec.mechanism.representation_probe import (
    load_m2_probe_manifest,
    normalize_query,
    normalized_query_fold,
)
from myrec.mechanism.scalar_condition_bundle import audit_scalar_partial
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl


ACTIVE_CONDITIONS = ("history_content_neutral", "history_attention_null")
REGISTERED_FAMILY_SIZE = 8


@dataclass(frozen=True)
class ContextualBundle:
    root: Path
    metadata: dict[str, Any]
    scores: dict[str, dict[str, dict[str, float]]]
    eligibility: np.ndarray


def evaluate_contextual_control_bundles(
    standardized_dir: str | Path,
    bundle_dirs: Mapping[str, str | Path],
    output_dir: str | Path,
    analysis_run_id: str,
    *,
    dev_eval_log_path: str | Path = "reports/dev_eval_log.jsonl",
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    if set(bundle_dirs) != set(SUPPORTED_METHODS):
        raise ValueError("contextual evaluator requires Q2 and Q3 bundles")
    standardized_dir = Path(standardized_dir)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"contextual evaluation output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = standardized_dir / "records_dev.jsonl"
    qrels_path = standardized_dir / "qrels_dev.jsonl"
    raw_records = list(iter_jsonl(records_path))
    records = [sanitize_record_for_model(row) for row in raw_records]
    if len(records) != 8000:
        raise ValueError("contextual evaluator requires all 8000 dev requests")
    candidates = _audit_candidate_and_request_manifests(
        standardized_dir / "candidate_manifest.json",
        standardized_dir / "request_manifest.json",
        records,
        raw_records,
    )
    bundles = {
        method_id: _audit_bundle(bundle_dirs[method_id], records, method_id)
        for method_id in SUPPORTED_METHODS
    }
    eligibility = bundles[SUPPORTED_METHODS[0]].eligibility
    if not np.array_equal(
        eligibility, bundles[SUPPORTED_METHODS[1]].eligibility
    ) or int(eligibility.sum()) != 7254:
        raise ValueError("contextual frozen eligibility differs across models")
    implementation_digest = _common_implementation_digest(bundles)
    pre_qrels = {
        "schema_version": 1,
        "analysis_type": "d5_contextual_controls_pre_qrels_integrity",
        "analysis_run_id": analysis_run_id,
        "status": "passed",
        "qrels_read": False,
        "checks": {
            "both_model_bundles_present": True,
            "all_requests_and_candidates_complete_finite": True,
            "unmodified_identity_at_most_1e-5": True,
            "candidate_and_request_manifests_reconstructed": True,
            "eligible_requests_exactly_7254": True,
            "eligibility_matches_bound_frozen_control_rows": True,
            "ineligible_conditions_equal_bound_frozen_baseline": True,
            "both_bundles_share_one_implementation_digest": True,
        },
        "implementation_digest": implementation_digest,
        "bundles": {
            method_id: {
                "path": str(bundle.root),
                "metadata_sha256": sha256_file(bundle.root / "metadata.json"),
                "scores_sha256": sha256_file(bundle.root / "scores.jsonl"),
                "maximum_identity_delta": bundle.metadata["maximum_identity_delta"],
            }
            for method_id, bundle in bundles.items()
        },
    }
    pre_qrels_path = output_dir / "pre_qrels_audit.json"
    _write_json(pre_qrels_path, pre_qrels)
    qrels_sha256 = sha256_file(qrels_path)
    if qrels_sha256 != load_m2_probe_manifest()["frozen_inputs"]["qrels_dev_sha256"]:
        raise ValueError("contextual evaluator qrels hash mismatch")
    gains = _load_dev_qrels(qrels_path, candidates)
    memberships = build_target_aware_surface_memberships(records_path, candidates, gains)
    request_ids = [record.request_id for record in records]
    clusters = np.asarray([normalize_query(record.query) for record in records], dtype=np.str_)
    folds = np.asarray([normalized_query_fold(record.query) for record in records], dtype=np.int8)
    strict = np.asarray(
        [request_id in memberships[STRICT_TRANSFER_SURFACE] for request_id in request_ids],
        dtype=bool,
    )
    results: dict[str, Any] = {}
    family_rows = []
    per_request = {}
    for method_id in SUPPORTED_METHODS:
        bundle = bundles[method_id]
        margins = {
            condition: _target_margins(
                request_ids, candidates, gains, bundle.scores[condition]
            )
            for condition in CONTEXTUAL_SCORE_CONDITIONS
        }
        ndcg = {
            condition: _ndcg_values(
                request_ids, candidates, gains, bundle.scores[condition]
            )
            for condition in CONTEXTUAL_SCORE_CONDITIONS
        }
        method_results = {}
        for condition in ACTIVE_CONDITIONS:
            condition_results = {}
            for endpoint, values in (
                ("target_margin", margins[condition] - margins["baseline_full"]),
                ("ndcg@10", ndcg[condition] - ndcg["baseline_full"]),
            ):
                rows = []
                for fold_name, fold_mask in (
                    ("all", np.ones(len(records), dtype=bool)),
                    ("0", folds == 0),
                    ("1", folds == 1),
                ):
                    mask = strict & eligibility & fold_mask & np.isfinite(values)
                    rows.append(
                        {
                            "surface": STRICT_TRANSFER_SURFACE,
                            "eligibility": "frozen_content_control_eligible",
                            "normalized_query_fold": fold_name,
                            **cluster_mean_inference(values[mask], clusters[mask]),
                        }
                    )
                all_row = next(row for row in rows if row["normalized_query_fold"] == "all")
                family_rows.append(
                    {
                        "method_id": method_id,
                        "condition": condition,
                        "endpoint": endpoint,
                        "two_sided_p": float(all_row["two_sided_p"]),
                    }
                )
                full_mask = strict & np.isfinite(values)
                condition_results[endpoint] = {
                    "registered": rows,
                    "descriptive_full_population": {
                        "surface": STRICT_TRANSFER_SURFACE,
                        "eligibility": "all_requests_with_ineligible_copied_baseline",
                        "normalized_query_fold": "all",
                        **cluster_mean_inference(values[full_mask], clusters[full_mask]),
                    },
                }
                per_request[f"{method_id}__{condition}__{endpoint}"] = values
            method_results[condition] = condition_results
        results[method_id] = method_results
    if len(family_rows) != REGISTERED_FAMILY_SIZE:
        raise AssertionError("contextual registered family size is not 8")
    for family_row, q_value in zip(
        family_rows,
        benjamini_hochberg([row["two_sided_p"] for row in family_rows]),
    ):
        family_row["bh_q"] = float(q_value)
        rows = results[family_row["method_id"]][family_row["condition"]][
            family_row["endpoint"]
        ]["registered"]
        next(row for row in rows if row["normalized_query_fold"] == "all")[
            "bh_q"
        ] = float(q_value)
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
        "analysis_type": "transformer_deep_dive_d5_contextual_controls",
        "analysis_run_id": analysis_run_id,
        "primary_surface": STRICT_TRANSFER_SURFACE,
        "implementation_digest": implementation_digest,
        "eligible_requests": int(eligibility.sum()),
        "strict_transfer_requests": int(strict.sum()),
        "strict_transfer_eligible_requests": int((strict & eligibility).sum()),
        "multiple_testing": {
            "family": "model_x_contextual_control_x_endpoint",
            "family_size": REGISTERED_FAMILY_SIZE,
            "method": "benjamini_hochberg",
        },
        "family_rows": family_rows,
        "results": results,
        "pre_qrels_audit_path": str(pre_qrels_path),
        "pre_qrels_audit_sha256": sha256_file(pre_qrels_path),
        "qrels_read": True,
        "qrels_opened_only_after_score_integrity": True,
        "qrels_dev_sha256": qrels_sha256,
        "per_request_contrasts_path": str(per_request_path),
        "per_request_contrasts_sha256": sha256_file(per_request_path),
        "command": list(command or []),
        "status": "completed",
    }
    metrics_path = output_dir / "metrics.json"
    _write_json(metrics_path, metrics)
    _append_jsonl(
        Path(dev_eval_log_path),
        {
            "analysis_type": metrics["analysis_type"],
            "run_id": analysis_run_id,
            "method_ids": list(SUPPORTED_METHODS),
            "split": "dev",
            "qrels_sha256": qrels_sha256,
            "metrics_path": str(metrics_path),
            "metrics_sha256": sha256_file(metrics_path),
        },
    )
    return metrics


def _audit_bundle(
    root: str | Path, records: Sequence[Any], method_id: str
) -> ContextualBundle:
    root = Path(root)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if (
        metadata.get("analysis_stage")
        != "transformer_deep_dive_d5_contextual_controls"
        or metadata.get("status") != "completed"
        or metadata.get("result_eligible") is not True
        or metadata.get("qrels_read") is not False
        or metadata.get("source_test_opened") is not False
        or metadata.get("complete_finite_score_coverage") is not True
        or metadata.get("identity_passed") is not True
        or float(metadata.get("maximum_identity_delta", math.inf)) > 1.0e-5
        or metadata.get("method_id") != method_id
        or tuple(metadata.get("score_conditions", ())) != CONTEXTUAL_SCORE_CONDITIONS
        or metadata.get("ineligible_scoring")
        != "copy_frozen_baseline_score"
    ):
        raise ValueError(f"contextual bundle metadata failed integrity: {root}")
    scores_path = root / "scores.jsonl"
    if metadata.get("scores_sha256") != sha256_file(scores_path):
        raise ValueError("contextual bundle score hash differs")
    observed = audit_scalar_partial(
        scores_path, records, CONTEXTUAL_SCORE_CONDITIONS
    )
    if observed["completed_requests"] != 8000 or observed[
        "completed_score_rows"
    ] != 160753:
        raise ValueError("contextual bundle coverage is incomplete")
    scores: dict[str, dict[str, dict[str, float]]] = {
        condition: {} for condition in CONTEXTUAL_SCORE_CONDITIONS
    }
    frozen_baseline = _load_bundle_frozen_baseline(
        metadata,
        records,
        label="D5 contextual control",
    )
    frozen_eligibility = _load_bundle_content_control_eligibility(
        metadata,
        records,
        label="D5 contextual control",
    )
    eligibility = []
    for block_row in iter_jsonl(scores_path):
        eligible_value = block_row.get("content_control_eligible")
        if not isinstance(eligible_value, bool):
            raise ValueError(
                "D5 contextual content-control eligibility is not boolean"
            )
        request_id = str(block_row["request_id"])
        if eligible_value is not frozen_eligibility[request_id]:
            raise ValueError(
                "D5 contextual eligibility differs from frozen controls"
            )
        eligibility.append(eligible_value)
        if not eligible_value:
            _audit_ineligible_frozen_conditions(
                request_id,
                block_row["rows"],
                CONTEXTUAL_SCORE_CONDITIONS,
                frozen_baseline,
                label="D5 contextual control",
            )
        for row in block_row["rows"]:
            request_id = str(row["request_id"])
            item_id = str(row["candidate_item_id"])
            for condition in CONTEXTUAL_SCORE_CONDITIONS:
                request = scores[condition].setdefault(request_id, {})
                request[item_id] = float(row["conditions"][condition])
    return ContextualBundle(
        root=root,
        metadata=metadata,
        scores=scores,
        eligibility=np.asarray(eligibility, dtype=bool),
    )


def _common_implementation_digest(bundles):
    metadata_rows = [bundle.metadata for bundle in bundles.values()]
    digests = {
        str(metadata.get("implementation_identity", {}).get("digest") or "")
        for metadata in metadata_rows
    }
    if len(digests) != 1 or not next(iter(digests), ""):
        raise ValueError(
            "D5 contextual bundles use different implementation digests"
        )
    digest = next(iter(digests))
    if any(
        metadata.get("run_contract", {}).get("implementation_digest") != digest
        for metadata in metadata_rows
    ):
        raise ValueError("D5 contextual implementation differs from run contract")
    return digest
