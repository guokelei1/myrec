"""Shared qrels-gated evaluator for registered D3 attention-edge bundles."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
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
    _load_content_controls,
    _load_frozen_baseline,
    _load_manifest,
)
from myrec.mechanism.attention_edge_scoring import ATTENTION_SCORE_CONDITIONS
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


ACTIVE_CONDITIONS = (
    "history_logits_mask",
    "history_value_edge_zero",
    "neutral_history_kv",
)
ENDPOINTS = ("target_margin", "ndcg@10")
REGISTERED_FAMILY_SIZE = 36


@dataclass(frozen=True)
class AttentionBundle:
    root: Path
    metadata: dict[str, Any]
    scores: dict[str, dict[str, dict[str, float]]]
    eligibility: np.ndarray


def evaluate_attention_edge_bundles(
    standardized_dir: str | Path,
    bundle_dirs: Mapping[str, Mapping[int, str | Path]],
    output_dir: str | Path,
    analysis_run_id: str,
    *,
    dev_eval_log_path: str | Path = "reports/dev_eval_log.jsonl",
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Evaluate the exact 2 models x 3 blocks x 3 conditions x 2 endpoints."""

    if set(bundle_dirs) != set(SUPPORTED_METHODS) or any(
        set(map(int, blocks)) != set(FIXED_BLOCKS)
        for blocks in bundle_dirs.values()
    ):
        raise ValueError("D3 evaluator requires Q2/Q3 and blocks 13/20/27")
    standardized_dir = Path(standardized_dir)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"D3 evaluation output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = standardized_dir / "records_dev.jsonl"
    qrels_path = standardized_dir / "qrels_dev.jsonl"
    raw_records = list(iter_jsonl(records_path))
    records = [sanitize_record_for_model(row) for row in raw_records]
    if len(records) != 8000:
        raise ValueError("D3 evaluator requires all 8000 dev requests")
    candidates = _audit_candidate_and_request_manifests(
        standardized_dir / "candidate_manifest.json",
        standardized_dir / "request_manifest.json",
        records,
        raw_records,
    )

    bundles: dict[str, dict[int, AttentionBundle]] = {}
    for method_id in SUPPORTED_METHODS:
        bundles[method_id] = {
            block: _audit_attention_bundle(
                bundle_dirs[method_id][block], records, method_id, block
            )
            for block in FIXED_BLOCKS
        }
        reference = bundles[method_id][13]
        for block in FIXED_BLOCKS[1:]:
            current = bundles[method_id][block]
            for key in (
                "method_id",
                "checkpoint_id",
                "config_sha256",
                "records_sha256",
                "candidate_manifest_sha256",
                "request_manifest_sha256",
                "dataset_manifest_sha256",
                "deep_dive_manifest_sha256",
            ):
                if current.metadata.get(key) != reference.metadata.get(key):
                    raise ValueError(
                        f"D3 bundle invariant differs for {method_id} block {block}: {key}"
                    )
            if not np.array_equal(current.eligibility, reference.eligibility):
                raise ValueError("D3 content-control eligibility differs across blocks")
    q2_eligibility = bundles[SUPPORTED_METHODS[0]][13].eligibility
    q3_eligibility = bundles[SUPPORTED_METHODS[1]][13].eligibility
    if not np.array_equal(q2_eligibility, q3_eligibility) or int(
        q2_eligibility.sum()
    ) != 7254:
        raise ValueError("D3 frozen content-control eligibility differs across models")
    implementation_digest = _common_implementation_digest(bundles)

    pre_qrels = {
        "schema_version": 1,
        "analysis_type": "d3_attention_edges_pre_qrels_integrity",
        "analysis_run_id": analysis_run_id,
        "status": "passed",
        "qrels_read": False,
        "checks": {
            "all_six_registered_bundles_present": True,
            "all_requests_and_candidates_complete_finite": True,
            "all_three_identity_conditions_at_most_1e-5": True,
            "candidate_and_request_manifests_reconstructed": True,
            "frozen_eligibility_identical_across_models_and_blocks": True,
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
                    "maximum_identity_delta": bundle.metadata[
                        "maximum_identity_delta"
                    ],
                }
                for block, bundle in method_bundles.items()
            }
            for method_id, method_bundles in bundles.items()
        },
    }
    pre_qrels_path = output_dir / "pre_qrels_audit.json"
    _write_json(pre_qrels_path, pre_qrels)

    frozen = load_m2_probe_manifest()["frozen_inputs"]
    qrels_sha256 = sha256_file(qrels_path)
    if qrels_sha256 != frozen["qrels_dev_sha256"]:
        raise ValueError("D3 evaluator qrels hash mismatch")
    gains = _load_dev_qrels(qrels_path, candidates)
    memberships = build_target_aware_surface_memberships(records_path, candidates, gains)
    request_ids = [record.request_id for record in records]
    clusters = np.asarray([normalize_query(record.query) for record in records], dtype=np.str_)
    folds = np.asarray([normalized_query_fold(record.query) for record in records], dtype=np.int8)
    strict = np.asarray(
        [request_id in memberships[STRICT_TRANSFER_SURFACE] for request_id in request_ids],
        dtype=bool,
    )
    eligibility = q2_eligibility

    family_rows: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    per_request: dict[str, np.ndarray] = {}
    for method_id in SUPPORTED_METHODS:
        results[method_id] = {}
        for block in FIXED_BLOCKS:
            bundle = bundles[method_id][block]
            condition_scores = {
                name: _condition_scores(bundle, name)
                for name in ATTENTION_SCORE_CONDITIONS
            }
            baseline_margins = _target_margins(
                request_ids, candidates, gains, condition_scores["baseline_full"]
            )
            baseline_ndcg = _ndcg_values(
                request_ids, candidates, gains, condition_scores["baseline_full"]
            )
            block_results: dict[str, Any] = {}
            for condition in ACTIVE_CONDITIONS:
                active_margins = _target_margins(
                    request_ids, candidates, gains, condition_scores[condition]
                )
                active_ndcg = _ndcg_values(
                    request_ids, candidates, gains, condition_scores[condition]
                )
                values_by_endpoint = {
                    "target_margin": active_margins - baseline_margins,
                    "ndcg@10": active_ndcg - baseline_ndcg,
                }
                condition_results: dict[str, Any] = {}
                for endpoint, values in values_by_endpoint.items():
                    registered_rows = []
                    for fold_name, fold_mask in (
                        ("all", np.ones(len(records), dtype=bool)),
                        ("0", folds == 0),
                        ("1", folds == 1),
                    ):
                        mask = strict & eligibility & fold_mask & np.isfinite(values)
                        registered_rows.append(
                            {
                                "surface": STRICT_TRANSFER_SURFACE,
                                "eligibility": "frozen_content_control_eligible",
                                "normalized_query_fold": fold_name,
                                **cluster_mean_inference(values[mask], clusters[mask]),
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
                        row
                        for row in registered_rows
                        if row["normalized_query_fold"] == "all"
                    )
                    family_row = {
                        "method_id": method_id,
                        "block_zero_based": block,
                        "condition": condition,
                        "endpoint": endpoint,
                        "two_sided_p": float(all_row["two_sided_p"]),
                    }
                    family_rows.append(family_row)
                    condition_results[endpoint] = {
                        "registered": registered_rows,
                        "descriptive_full_population": descriptive_full,
                    }
                    per_request[
                        f"{method_id}__b{block}__{condition}__{endpoint}"
                    ] = values
                block_results[condition] = condition_results
            results[method_id][str(block)] = block_results
    if len(family_rows) != REGISTERED_FAMILY_SIZE:
        raise AssertionError("D3 registered family size is not 36")
    q_values = benjamini_hochberg([row["two_sided_p"] for row in family_rows])
    for family_row, q_value in zip(family_rows, q_values):
        family_row["bh_q"] = float(q_value)
        registered = results[family_row["method_id"]][
            str(family_row["block_zero_based"])
        ][family_row["condition"]][family_row["endpoint"]]["registered"]
        next(row for row in registered if row["normalized_query_fold"] == "all")[
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
        "analysis_type": "transformer_deep_dive_d3_attention_edges",
        "analysis_run_id": analysis_run_id,
        "primary_surface": STRICT_TRANSFER_SURFACE,
        "registered_eligibility": "frozen_content_control_eligible",
        "implementation_digest": implementation_digest,
        "eligible_requests": int(eligibility.sum()),
        "strict_transfer_requests": int(strict.sum()),
        "strict_transfer_eligible_requests": int((strict & eligibility).sum()),
        "bootstrap": {
            "cluster": "normalized_query",
            "samples": 5000,
            "seed": 20260715,
        },
        "multiple_testing": {
            "family": "model_x_block_x_active_condition_x_endpoint",
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


def _audit_attention_bundle(
    root: str | Path,
    records: Sequence[Any],
    method_id: str,
    block: int,
) -> AttentionBundle:
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
        or int(metadata.get("block_zero_based", -1)) != block
        or tuple(metadata.get("score_conditions", ())) != ATTENTION_SCORE_CONDITIONS
        or metadata.get("ineligible_scoring")
        != "copy_frozen_baseline_score"
    ):
        raise ValueError(f"D3 bundle metadata failed integrity: {root}")
    scores_path = root / "scores.jsonl"
    if metadata.get("scores_sha256") != sha256_file(scores_path):
        raise ValueError("D3 attention-edge score hash differs")
    observed = audit_scalar_partial(
        scores_path, records, ATTENTION_SCORE_CONDITIONS
    )
    if observed["completed_requests"] != len(records) or observed[
        "completed_score_rows"
    ] != 160753:
        raise ValueError("D3 bundle has incomplete request/candidate coverage")
    scores: dict[str, dict[str, dict[str, float]]] = {
        condition: {} for condition in ATTENTION_SCORE_CONDITIONS
    }
    frozen_baseline = _load_bundle_frozen_baseline(
        metadata,
        records,
        label="D3 attention-edge",
    )
    frozen_eligibility = _load_bundle_content_control_eligibility(
        metadata,
        records,
        label="D3 attention-edge",
    )
    eligibility = []
    for ordinal, block_row in enumerate(iter_jsonl(scores_path)):
        if int(block_row.get("block_zero_based", -1)) != block:
            raise ValueError("D3 score row block drift")
        eligible_value = block_row.get("content_control_eligible")
        if not isinstance(eligible_value, bool):
            raise ValueError("D3 content-control eligibility is not boolean")
        request_id = str(block_row["request_id"])
        if eligible_value is not frozen_eligibility[request_id]:
            raise ValueError(
                "D3 content-control eligibility differs from frozen controls"
            )
        eligibility.append(eligible_value)
        if not eligible_value:
            _audit_ineligible_frozen_conditions(
                request_id,
                block_row["rows"],
                ATTENTION_SCORE_CONDITIONS,
                frozen_baseline,
                label="D3 attention-edge",
            )
        for row in block_row["rows"]:
            request_id = str(row["request_id"])
            item_id = str(row["candidate_item_id"])
            for condition in ATTENTION_SCORE_CONDITIONS:
                request = scores[condition].setdefault(request_id, {})
                request[item_id] = float(row["conditions"][condition])
    return AttentionBundle(
        root=root,
        metadata=metadata,
        scores=scores,
        eligibility=np.asarray(eligibility, dtype=bool),
    )


def _load_bundle_frozen_baseline(
    metadata: Mapping[str, Any],
    records: Sequence[Any],
    *,
    label: str,
    identity_key: str = "frozen_baseline",
) -> dict[tuple[str, str], float]:
    """Reload and byte-bind the frozen fallback used by a scalar bundle."""

    identity = metadata.get(identity_key)
    if not isinstance(identity, Mapping):
        raise ValueError(f"{label} frozen-baseline identity is absent")
    root = identity.get("root")
    method_id = metadata.get("method_id")
    checkpoint_id = metadata.get("checkpoint_id")
    if not isinstance(root, str) or not root:
        raise ValueError(f"{label} frozen-baseline root is invalid")
    if not isinstance(method_id, str) or not isinstance(checkpoint_id, str):
        raise ValueError(f"{label} frozen-baseline model identity is invalid")
    values, observed_identity = _load_frozen_baseline(
        Path(root),
        method_id,
        checkpoint_id,
        records,
    )
    if dict(identity) != observed_identity:
        raise ValueError(f"{label} frozen-baseline byte identity drift")
    return values


def _load_bundle_content_control_eligibility(
    metadata: Mapping[str, Any],
    records: Sequence[Any],
    *,
    label: str,
    identity_key: str = "content_control",
) -> dict[str, bool]:
    """Reload frozen control rows and bind the exact per-request mask."""

    manifest = _load_manifest(DEEP_DIVE_MANIFEST_PATH)
    if metadata.get("deep_dive_manifest_sha256") != manifest["_sha256"]:
        raise ValueError(f"{label} deep-dive manifest identity drift")
    method_id = metadata.get("method_id")
    if not isinstance(method_id, str):
        raise ValueError(f"{label} content-control model identity is invalid")
    controls, observed_identity = _load_content_controls(
        manifest,
        method_id,
        records,
    )
    identity = metadata.get(identity_key)
    if not isinstance(identity, Mapping) or dict(identity) != observed_identity:
        raise ValueError(f"{label} content-control byte identity drift")
    return {
        record.request_id: controls[record.request_id].get("eligible") is True
        for record in records
    }


def _audit_ineligible_frozen_conditions(
    request_id: str,
    rows: Sequence[Mapping[str, Any]],
    conditions: Sequence[str],
    frozen_baseline: Mapping[tuple[str, str], float],
    *,
    label: str,
) -> None:
    """Require every ineligible condition to be the bound frozen score."""

    for row in rows:
        row_request_id = str(row.get("request_id"))
        item_id = str(row.get("candidate_item_id"))
        if row_request_id != request_id:
            raise ValueError(f"{label} ineligible request identity drift")
        key = (request_id, item_id)
        if key not in frozen_baseline:
            raise ValueError(f"{label} ineligible candidate is absent from frozen baseline")
        expected = float(frozen_baseline[key])
        values = row.get("conditions")
        if not isinstance(values, Mapping):
            raise ValueError(f"{label} ineligible conditions are invalid")
        for condition in conditions:
            if condition not in values or float(values[condition]) != expected:
                raise ValueError(
                    f"{label} ineligible condition differs from frozen baseline: "
                    f"{request_id}/{item_id}/{condition}"
                )


def _condition_scores(
    bundle: AttentionBundle, condition: str
) -> dict[str, dict[str, float]]:
    return bundle.scores[condition]


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
        raise ValueError(
            "D3 attention-edge bundles use different implementation digests"
        )
    digest = next(iter(digests))
    if any(
        metadata.get("run_contract", {}).get("implementation_digest") != digest
        for metadata in metadata_rows
    ):
        raise ValueError("D3 attention-edge implementation differs from run contract")
    return digest


def _ndcg_values(
    request_ids: Sequence[str],
    candidates: Mapping[str, Sequence[str]],
    gains: Mapping[str, Mapping[str, float]],
    scores: Mapping[str, Mapping[str, float]],
) -> np.ndarray:
    values = []
    for request_id in request_ids:
        candidate_ids = candidates[request_id]
        values.append(
            gain_ndcg_at_k(
                request_id,
                list(candidate_ids),
                [float(scores[request_id][candidate_id]) for candidate_id in candidate_ids],
                [float(gains[request_id].get(candidate_id, 0.0)) for candidate_id in candidate_ids],
                10,
            )
        )
    return np.asarray(values, dtype=np.float64)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
