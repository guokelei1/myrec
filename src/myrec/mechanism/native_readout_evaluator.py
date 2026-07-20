"""Shared qrels-gated evaluator for the registered D6 Q2 native readout."""

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
    _append_jsonl,
    _ndcg_values,
    _write_json,
)
from myrec.mechanism.deep_dive_native_evaluator import (
    benjamini_hochberg,
    cluster_mean_inference,
)
from myrec.mechanism.native_readout_runtime import (
    Q2_METHOD_ID,
    Q2_NATIVE_READOUT_CONDITIONS,
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
from myrec.mechanism.readout_decomposition import condition_decomposition_report
from myrec.mechanism.scalar_condition_bundle import audit_scalar_partial
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl


Q2_READOUT_COMPARISONS = {
    "same_minus_null": "baseline_null",
    "same_minus_full": "baseline_full",
    "same_minus_cross": None,
}
Q2_READOUT_ENDPOINTS = ("target_margin", "ndcg@10")
Q2_READOUT_FAMILY_SIZE = 12


@dataclass(frozen=True)
class Q2NativeReadoutBundle:
    root: Path
    metadata: dict[str, Any]
    scores: dict[str, dict[str, dict[str, float]]]


def evaluate_q2_native_readout(
    standardized_dir: str | Path,
    bundle_dir: str | Path,
    output_dir: str | Path,
    analysis_run_id: str,
    *,
    dev_eval_log_path: str | Path = "reports/dev_eval_log.jsonl",
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Evaluate 2 nodes x 3 comparisons x 2 endpoints after integrity gates."""

    standardized_dir = Path(standardized_dir)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"D6 Q2 evaluation output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = standardized_dir / "records_dev.jsonl"
    qrels_path = standardized_dir / "qrels_dev.jsonl"
    raw_records = list(iter_jsonl(records_path))
    records = [sanitize_record_for_model(row) for row in raw_records]
    if len(records) != 8000:
        raise ValueError("D6 Q2 evaluator requires all 8000 dev requests")
    candidates = _audit_candidate_and_request_manifests(
        standardized_dir / "candidate_manifest.json",
        standardized_dir / "request_manifest.json",
        records,
        raw_records,
    )
    bundle = _audit_q2_native_readout_bundle(bundle_dir, records)
    request_ids = [record.request_id for record in records]
    readout_decomposition = condition_decomposition_report(
        bundle.scores,
        request_ids,
        candidates,
        {
            f"{prefix}__{comparison}": (
                f"{prefix}_same_full_to_null",
                (
                    f"{prefix}_cross_full_to_null"
                    if reference is None
                    else reference
                ),
            )
            for prefix in ("input", "output")
            for comparison, reference in Q2_READOUT_COMPARISONS.items()
        },
    )
    pre_qrels = {
        "schema_version": 1,
        "analysis_type": "d6_q2_native_readout_pre_qrels_integrity",
        "analysis_run_id": analysis_run_id,
        "status": "passed",
        "qrels_read": False,
        "checks": {
            "all_requests_and_candidates_complete_finite": True,
            "all_four_identity_conditions_at_most_1e-5": True,
            "full_and_null_recompute_within_frozen_bf16_algebra_bound": True,
            "native_tied_readout_algebra_within_frozen_bf16_bound": True,
            "request_common_candidate_relative_recomposition_exact": True,
            "candidate_and_request_manifests_reconstructed": True,
        },
        "bundle": {
            "path": str(bundle.root),
            "metadata_sha256": sha256_file(bundle.root / "metadata.json"),
            "scores_sha256": sha256_file(bundle.root / "scores.jsonl"),
            "maximum_identity_delta": bundle.metadata["maximum_identity_delta"],
            "maximum_algebra_delta": bundle.metadata["maximum_algebra_delta"],
        },
    }
    pre_qrels_path = output_dir / "pre_qrels_audit.json"
    _write_json(pre_qrels_path, pre_qrels)

    frozen = load_m2_probe_manifest()["frozen_inputs"]
    qrels_sha256 = sha256_file(qrels_path)
    if qrels_sha256 != frozen["qrels_dev_sha256"]:
        raise ValueError("D6 Q2 evaluator qrels hash mismatch")
    gains = _load_dev_qrels(qrels_path, candidates)
    memberships = build_target_aware_surface_memberships(records_path, candidates, gains)
    clusters = np.asarray([normalize_query(record.query) for record in records], dtype=np.str_)
    folds = np.asarray([normalized_query_fold(record.query) for record in records], dtype=np.int8)
    strict = np.asarray(
        [request_id in memberships[STRICT_TRANSFER_SURFACE] for request_id in request_ids],
        dtype=bool,
    )
    condition_scores = {
        name: bundle.scores[name] for name in Q2_NATIVE_READOUT_CONDITIONS
    }
    endpoints = {
        name: {
            "target_margin": _target_margins(
                request_ids, candidates, gains, condition_scores[name]
            ),
            "ndcg@10": _ndcg_values(
                request_ids, candidates, gains, condition_scores[name]
            ),
        }
        for name in Q2_NATIVE_READOUT_CONDITIONS
    }
    results: dict[str, Any] = {}
    family_rows: list[dict[str, Any]] = []
    per_request: dict[str, np.ndarray] = {}
    for prefix in ("input", "output"):
        same = f"{prefix}_same_full_to_null"
        cross = f"{prefix}_cross_full_to_null"
        node_results: dict[str, Any] = {}
        for comparison, fixed_reference in Q2_READOUT_COMPARISONS.items():
            reference = cross if fixed_reference is None else fixed_reference
            comparison_results: dict[str, Any] = {}
            for endpoint in Q2_READOUT_ENDPOINTS:
                values = q2_readout_contrast_values(
                    endpoints,
                    prefix=prefix,
                    comparison=comparison,
                    endpoint=endpoint,
                )
                inference_rows = []
                for fold_name, fold_mask in (
                    ("all", np.ones(len(records), dtype=bool)),
                    ("0", folds == 0),
                    ("1", folds == 1),
                ):
                    mask = strict & fold_mask & np.isfinite(values)
                    inference_rows.append(
                        {
                            "surface": STRICT_TRANSFER_SURFACE,
                            "normalized_query_fold": fold_name,
                            **cluster_mean_inference(values[mask], clusters[mask]),
                        }
                    )
                all_row = next(
                    row for row in inference_rows if row["normalized_query_fold"] == "all"
                )
                family_rows.append(
                    {
                        "node": f"final_rmsnorm_{prefix}",
                        "comparison": comparison,
                        "endpoint": endpoint,
                        "two_sided_p": float(all_row["two_sided_p"]),
                    }
                )
                comparison_results[endpoint] = inference_rows
                per_request[f"{prefix}__{comparison}__{endpoint}"] = values
            node_results[comparison] = comparison_results
        results[f"final_rmsnorm_{prefix}"] = node_results
    if len(family_rows) != Q2_READOUT_FAMILY_SIZE:
        raise AssertionError("D6 Q2 registered family size is not 12")
    q_values = benjamini_hochberg([row["two_sided_p"] for row in family_rows])
    for family_row, q_value in zip(family_rows, q_values):
        family_row["bh_q"] = float(q_value)
        all_row = next(
            row
            for row in results[family_row["node"]][family_row["comparison"]][
                family_row["endpoint"]
            ]
            if row["normalized_query_fold"] == "all"
        )
        all_row["bh_q"] = float(q_value)

    per_request_path = output_dir / "per_request_contrasts.npz"
    np.savez(
        per_request_path,
        **per_request,
        request_ids=np.asarray(request_ids, dtype=np.str_),
        normalized_queries=clusters,
        folds=folds,
        strict_mask=strict,
    )
    metrics = {
        "schema_version": 1,
        "analysis_type": "transformer_deep_dive_d6_q2_native_readout",
        "analysis_run_id": analysis_run_id,
        "method_id": Q2_METHOD_ID,
        "primary_surface": STRICT_TRANSFER_SURFACE,
        "strict_transfer_requests": int(strict.sum()),
        "bootstrap": {"cluster": "normalized_query", "samples": 5000, "seed": 20260715},
        "multiple_testing": {
            "family": "node_x_comparison_x_endpoint",
            "family_size": Q2_READOUT_FAMILY_SIZE,
            "method": "benjamini_hochberg",
        },
        "family_rows": family_rows,
        "results": results,
        "readout_decomposition": readout_decomposition,
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
            "method_ids": [Q2_METHOD_ID],
            "split": "dev",
            "qrels_sha256": qrels_sha256,
            "metrics_path": str(metrics_path),
            "metrics_sha256": sha256_file(metrics_path),
        },
    )
    return metrics


def _audit_q2_native_readout_bundle(
    root: str | Path, records: Sequence[Any]
) -> Q2NativeReadoutBundle:
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
        or metadata.get("low_precision_algebra_passed") is not True
        or float(metadata.get("maximum_baseline_low_precision_ratio", math.inf)) > 1.0
        or float(metadata.get("maximum_algebra_low_precision_ratio", math.inf)) > 1.0
        or not math.isfinite(float(metadata.get("maximum_algebra_delta", math.inf)))
        or metadata.get("method_id") != Q2_METHOD_ID
        or tuple(metadata.get("score_conditions", ())) != Q2_NATIVE_READOUT_CONDITIONS
    ):
        raise ValueError(f"D6 Q2 bundle metadata failed integrity: {root}")
    observed = audit_scalar_partial(
        root / "scores.jsonl", records, Q2_NATIVE_READOUT_CONDITIONS
    )
    if observed["completed_requests"] != len(records) or observed[
        "completed_score_rows"
    ] != 160753:
        raise ValueError("D6 Q2 bundle has incomplete request/candidate coverage")
    scores: dict[str, dict[str, dict[str, float]]] = {
        condition: {} for condition in Q2_NATIVE_READOUT_CONDITIONS
    }
    for block_row in iter_jsonl(root / "scores.jsonl"):
        for row in block_row["rows"]:
            request_id = str(row["request_id"])
            item_id = str(row["candidate_item_id"])
            for condition in Q2_NATIVE_READOUT_CONDITIONS:
                scores[condition].setdefault(request_id, {})[item_id] = float(
                    row["conditions"][condition]
                )
    return Q2NativeReadoutBundle(root=root, metadata=metadata, scores=scores)


def q2_readout_contrast_values(
    endpoints: Mapping[str, Mapping[str, np.ndarray]],
    *,
    prefix: str,
    comparison: str,
    endpoint: str,
) -> np.ndarray:
    """Return the preregistered same-minus-reference per-request vector."""

    if prefix not in {"input", "output"}:
        raise ValueError("Q2 readout contrast prefix is invalid")
    if comparison not in Q2_READOUT_COMPARISONS:
        raise ValueError("Q2 readout comparison is invalid")
    if endpoint not in Q2_READOUT_ENDPOINTS:
        raise ValueError("Q2 readout endpoint is invalid")
    same = f"{prefix}_same_full_to_null"
    fixed_reference = Q2_READOUT_COMPARISONS[comparison]
    reference = f"{prefix}_cross_full_to_null" if fixed_reference is None else fixed_reference
    left = np.asarray(endpoints[same][endpoint], dtype=np.float64)
    right = np.asarray(endpoints[reference][endpoint], dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("Q2 readout contrast arrays are misaligned")
    return left - right
