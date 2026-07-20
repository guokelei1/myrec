"""Shared qrels-gated evaluator for registered D5 RoPE bundles."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from myrec.mechanism.rope_scoring import COMMON_OFFSET_BOUND_RATIO_TOLERANCE

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
from myrec.mechanism.attention_edge_runtime import FIXED_BLOCKS, SUPPORTED_METHODS
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
from myrec.mechanism.rope_scoring import ROPE_SCORE_CONDITIONS
from myrec.mechanism.scalar_condition_bundle import audit_scalar_partial
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl


ROPE_CONTRASTS = {
    "readout_q": (
        "readout_q_distance_compression",
        "readout_q_distance_expansion",
    ),
    "history_k": (
        "history_k_distance_compression",
        "history_k_distance_expansion",
    ),
    "paired_qk": (
        "paired_qk_distance_compression",
        "paired_qk_distance_expansion",
    ),
}
REGISTERED_FAMILY_SIZE = 36


@dataclass(frozen=True)
class RoPEBundle:
    root: Path
    metadata: dict[str, Any]
    scores: dict[str, dict[str, dict[str, float]]]
    eligibility: np.ndarray


def evaluate_rope_bundles(
    standardized_dir: str | Path,
    bundle_dirs: Mapping[str, Mapping[int, str | Path]],
    output_dir: str | Path,
    analysis_run_id: str,
    *,
    dev_eval_log_path: str | Path = "reports/dev_eval_log.jsonl",
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    if set(bundle_dirs) != set(SUPPORTED_METHODS) or any(
        set(map(int, blocks)) != set(FIXED_BLOCKS)
        for blocks in bundle_dirs.values()
    ):
        raise ValueError("D5 evaluator requires Q2/Q3 and blocks 13/20/27")
    standardized_dir = Path(standardized_dir)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"D5 evaluation output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = standardized_dir / "records_dev.jsonl"
    qrels_path = standardized_dir / "qrels_dev.jsonl"
    raw_records = list(iter_jsonl(records_path))
    records = [sanitize_record_for_model(row) for row in raw_records]
    if len(records) != 8000:
        raise ValueError("D5 evaluator requires all 8000 dev requests")
    candidates = _audit_candidate_and_request_manifests(
        standardized_dir / "candidate_manifest.json",
        standardized_dir / "request_manifest.json",
        records,
        raw_records,
    )
    bundles = {
        method_id: {
            block: _audit_rope_bundle(
                bundle_dirs[method_id][block], records, method_id, block
            )
            for block in FIXED_BLOCKS
        }
        for method_id in SUPPORTED_METHODS
    }
    invariants = (
        "method_id",
        "checkpoint_id",
        "config_sha256",
        "records_sha256",
        "candidate_manifest_sha256",
        "request_manifest_sha256",
        "dataset_manifest_sha256",
        "deep_dive_manifest_sha256",
    )
    for method_id in SUPPORTED_METHODS:
        reference = bundles[method_id][13]
        for block in FIXED_BLOCKS[1:]:
            current = bundles[method_id][block]
            if any(
                current.metadata.get(key) != reference.metadata.get(key)
                for key in invariants
            ):
                raise ValueError(f"D5 bundle invariants differ at {method_id} b{block}")
            if not np.array_equal(current.eligibility, reference.eligibility):
                raise ValueError("D5 eligibility differs across blocks")
    eligibility = bundles[SUPPORTED_METHODS[0]][13].eligibility
    if not np.array_equal(
        eligibility, bundles[SUPPORTED_METHODS[1]][13].eligibility
    ) or int(eligibility.sum()) != 7254:
        raise ValueError("D5 eligibility differs across models or frozen count")
    implementation_digest = _common_implementation_digest(bundles)
    pre_qrels = {
        "schema_version": 1,
        "analysis_type": "d5_rope_pre_qrels_integrity",
        "analysis_run_id": analysis_run_id,
        "status": "passed",
        "qrels_read": False,
        "checks": {
            "all_six_registered_bundles_present": True,
            "all_requests_and_candidates_complete_finite": True,
            "zero_phase_identity_at_most_1e-5": True,
            "common_offset_within_registered_low_precision_bound": True,
            "candidate_and_request_manifests_reconstructed": True,
            "eligible_requests_exactly_7254": True,
            "eligibility_matches_bound_frozen_control_rows": True,
            "ineligible_conditions_equal_bound_frozen_baseline": True,
            "all_six_bundles_share_one_implementation_digest": True,
        },
        "implementation_digest": implementation_digest,
        "bundles": {
            method_id: {
                str(block): {
                    "path": str(bundle.root),
                    "metadata_sha256": sha256_file(bundle.root / "metadata.json"),
                    "scores_sha256": sha256_file(bundle.root / "scores.jsonl"),
                    "identity_deltas": bundle.metadata.get("identity_deltas"),
                }
                for block, bundle in method_bundles.items()
            }
            for method_id, method_bundles in bundles.items()
        },
    }
    pre_qrels_path = output_dir / "pre_qrels_audit.json"
    _write_json(pre_qrels_path, pre_qrels)

    qrels_sha256 = sha256_file(qrels_path)
    if qrels_sha256 != load_m2_probe_manifest()["frozen_inputs"]["qrels_dev_sha256"]:
        raise ValueError("D5 evaluator qrels hash mismatch")
    gains = _load_dev_qrels(qrels_path, candidates)
    memberships = build_target_aware_surface_memberships(records_path, candidates, gains)
    request_ids = [record.request_id for record in records]
    clusters = np.asarray([normalize_query(record.query) for record in records], dtype=np.str_)
    folds = np.asarray([normalized_query_fold(record.query) for record in records], dtype=np.int8)
    strict = np.asarray(
        [request_id in memberships[STRICT_TRANSFER_SURFACE] for request_id in request_ids],
        dtype=bool,
    )
    family_rows: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    per_request: dict[str, np.ndarray] = {}
    for method_id in SUPPORTED_METHODS:
        results[method_id] = {}
        for block in FIXED_BLOCKS:
            bundle = bundles[method_id][block]
            scores = bundle.scores
            margins = {
                condition: _target_margins(
                    request_ids, candidates, gains, scores[condition]
                )
                for condition in ROPE_SCORE_CONDITIONS
            }
            ndcg = {
                condition: _ndcg_values(
                    request_ids, candidates, gains, scores[condition]
                )
                for condition in ROPE_SCORE_CONDITIONS
            }
            block_results: dict[str, Any] = {}
            for contrast, (compression, expansion) in ROPE_CONTRASTS.items():
                endpoint_values = {
                    "target_margin": margins[compression] - margins[expansion],
                    "ndcg@10": ndcg[compression] - ndcg[expansion],
                }
                contrast_results: dict[str, Any] = {}
                for endpoint, values in endpoint_values.items():
                    rows = []
                    compression_minus_baseline = (
                        (margins if endpoint == "target_margin" else ndcg)[compression]
                        - (margins if endpoint == "target_margin" else ndcg)[
                            "baseline_full"
                        ]
                    )
                    compression_baseline_rows = []
                    for fold_name, fold_mask in (
                        ("all", np.ones(len(records), dtype=bool)),
                        ("0", folds == 0),
                        ("1", folds == 1),
                    ):
                        mask = strict & eligibility & fold_mask & np.isfinite(values)
                        rows.append(
                            {
                                "surface": STRICT_TRANSFER_SURFACE,
                                "eligibility": "frozen_rope_anchor_eligible",
                                "normalized_query_fold": fold_name,
                                **cluster_mean_inference(values[mask], clusters[mask]),
                            }
                        )
                        active_mask = (
                            strict
                            & eligibility
                            & fold_mask
                            & np.isfinite(compression_minus_baseline)
                        )
                        active_inference = cluster_mean_inference(
                            compression_minus_baseline[active_mask],
                            clusters[active_mask],
                        )
                        # This interval is a frozen support gate, not another
                        # member of the 36-cell compression-minus-expansion
                        # FDR family.  Keep its p-value out of the payload so
                        # it cannot be mistaken for an unregistered test.
                        active_inference.pop("two_sided_p", None)
                        compression_baseline_rows.append(
                            {
                                "surface": STRICT_TRANSFER_SURFACE,
                                "eligibility": "frozen_rope_anchor_eligible",
                                "normalized_query_fold": fold_name,
                                **active_inference,
                            }
                        )
                    full_mask = strict & np.isfinite(values)
                    descriptive_full = {
                        "surface": STRICT_TRANSFER_SURFACE,
                        "eligibility": "all_requests_with_ineligible_copied_baseline",
                        "normalized_query_fold": "all",
                        **cluster_mean_inference(values[full_mask], clusters[full_mask]),
                    }
                    all_row = next(
                        row for row in rows if row["normalized_query_fold"] == "all"
                    )
                    family_rows.append(
                        {
                            "method_id": method_id,
                            "block_zero_based": block,
                            "contrast": contrast,
                            "endpoint": endpoint,
                            "two_sided_p": float(all_row["two_sided_p"]),
                        }
                    )
                    contrast_results[endpoint] = {
                        "registered_compression_minus_expansion": rows,
                        "registered_compression_minus_baseline_support_gate": (
                            compression_baseline_rows
                        ),
                        "descriptive_full_population": descriptive_full,
                        "descriptive_compression_minus_baseline_mean": float(
                            np.mean(compression_minus_baseline[strict & eligibility])
                        ),
                        "descriptive_expansion_minus_baseline_mean": float(
                            np.mean(
                                (margins if endpoint == "target_margin" else ndcg)[
                                    expansion
                                ][strict & eligibility]
                                - (margins if endpoint == "target_margin" else ndcg)[
                                    "baseline_full"
                                ][strict & eligibility]
                            )
                        ),
                    }
                    per_request[
                        f"{method_id}__b{block}__{contrast}__{endpoint}"
                    ] = values
                block_results[contrast] = contrast_results
            results[method_id][str(block)] = block_results
    if len(family_rows) != REGISTERED_FAMILY_SIZE:
        raise AssertionError("D5 registered family size is not 36")
    q_values = benjamini_hochberg([row["two_sided_p"] for row in family_rows])
    for family_row, q_value in zip(family_rows, q_values):
        family_row["bh_q"] = float(q_value)
        rows = results[family_row["method_id"]][str(family_row["block_zero_based"])][
            family_row["contrast"]
        ][family_row["endpoint"]]["registered_compression_minus_expansion"]
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
        "analysis_type": "transformer_deep_dive_d5_rope",
        "analysis_run_id": analysis_run_id,
        "primary_surface": STRICT_TRANSFER_SURFACE,
        "registered_contrast": "distance_compression_minus_distance_expansion",
        "implementation_digest": implementation_digest,
        "eligible_requests": int(eligibility.sum()),
        "strict_transfer_requests": int(strict.sum()),
        "strict_transfer_eligible_requests": int((strict & eligibility).sum()),
        "bootstrap": {"cluster": "normalized_query", "samples": 5000, "seed": 20260715},
        "multiple_testing": {
            "family": "model_x_block_x_rope_scope_x_endpoint",
            "family_size": REGISTERED_FAMILY_SIZE,
            "method": "benjamini_hochberg",
        },
        "position_support_gate": {
            "active_contrast": "compression_minus_baseline",
            "active_endpoint": "ndcg@10",
            "active_ci95_equivalence_band": [-0.005, 0.005],
            "requires_compression_minus_expansion_bh_q_below_alpha_0p05": True,
            "requires_all_fold0_fold1_same_nonzero_direction": True,
            "active_contrast_is_confirmatory_family_member": False,
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


def _audit_rope_bundle(
    root: str | Path,
    records: Sequence[Any],
    method_id: str,
    block: int,
) -> RoPEBundle:
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
        or metadata.get("common_offset_low_precision_passed") is not True
        or float(metadata.get("common_offset_low_precision_max_ratio", math.inf))
        > 1.0 + COMMON_OFFSET_BOUND_RATIO_TOLERANCE
        or metadata.get("method_id") != method_id
        or int(metadata.get("block_zero_based", -1)) != block
        or tuple(metadata.get("score_conditions", ())) != ROPE_SCORE_CONDITIONS
        or metadata.get("ineligible_scoring")
        != "copy_frozen_baseline_score"
    ):
        raise ValueError(f"D5 bundle metadata failed integrity: {root}")
    scores_path = root / "scores.jsonl"
    if metadata.get("scores_sha256") != sha256_file(scores_path):
        raise ValueError("D5 RoPE score hash differs")
    observed = audit_scalar_partial(scores_path, records, ROPE_SCORE_CONDITIONS)
    if observed["completed_requests"] != 8000 or observed[
        "completed_score_rows"
    ] != 160753:
        raise ValueError("D5 bundle has incomplete request/candidate coverage")
    scores: dict[str, dict[str, dict[str, float]]] = {
        condition: {} for condition in ROPE_SCORE_CONDITIONS
    }
    frozen_baseline = _load_bundle_frozen_baseline(
        metadata,
        records,
        label="D5 RoPE",
    )
    frozen_eligibility = _load_bundle_content_control_eligibility(
        metadata,
        records,
        label="D5 RoPE",
    )
    eligibility = []
    for block_row in iter_jsonl(scores_path):
        if int(block_row.get("block_zero_based", -1)) != block:
            raise ValueError("D5 score row block drift")
        eligible_value = block_row.get("content_control_eligible")
        if not isinstance(eligible_value, bool):
            raise ValueError("D5 RoPE content-control eligibility is not boolean")
        request_id = str(block_row["request_id"])
        if eligible_value is not frozen_eligibility[request_id]:
            raise ValueError(
                "D5 RoPE content-control eligibility differs from frozen controls"
            )
        eligibility.append(eligible_value)
        if not eligible_value:
            _audit_ineligible_frozen_conditions(
                request_id,
                block_row["rows"],
                ROPE_SCORE_CONDITIONS,
                frozen_baseline,
                label="D5 RoPE",
            )
        for row in block_row["rows"]:
            request_id = str(row["request_id"])
            item_id = str(row["candidate_item_id"])
            for condition in ROPE_SCORE_CONDITIONS:
                request = scores[condition].setdefault(request_id, {})
                request[item_id] = float(row["conditions"][condition])
    return RoPEBundle(
        root=root,
        metadata=metadata,
        scores=scores,
        eligibility=np.asarray(eligibility, dtype=bool),
    )


def _common_implementation_digest(bundles):
    metadata_rows = [
        bundle.metadata
        for method_bundles in bundles.values()
        for bundle in method_bundles.values()
    ]
    digests = {
        str(metadata.get("implementation_identity", {}).get("digest") or "")
        for metadata in metadata_rows
    }
    if len(digests) != 1 or not next(iter(digests), ""):
        raise ValueError("D5 RoPE bundles use different implementation digests")
    digest = next(iter(digests))
    if any(
        metadata.get("run_contract", {}).get("implementation_digest") != digest
        for metadata in metadata_rows
    ):
        raise ValueError("D5 RoPE implementation differs from run contract")
    return digest
