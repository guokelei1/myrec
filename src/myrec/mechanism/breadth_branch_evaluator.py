"""Shared qrels-gated 96-cell Q0/Q1 fixed branch extension evaluator."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from myrec.baselines.motivation_v12_contracts import sanitize_record_for_model
from myrec.mechanism.attention_edge_evaluator import _append_jsonl, _ndcg_values, _write_json
from myrec.mechanism.deep_dive_native_evaluator import (
    benjamini_hochberg,
    cluster_mean_inference,
)
from myrec.mechanism.patch_evaluator import _target_margins
from myrec.mechanism.q0_branch_runtime import Q0_METHOD_ID
from myrec.mechanism.q0_branch_scoring import Q0_BRANCH_CONDITIONS, Q0_BRANCH_NODES
from myrec.mechanism.q1_branch_runtime import Q1_METHOD_ID
from myrec.mechanism.q1_branch_scoring import Q1_BRANCH_CONDITIONS
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


METHODS = (Q0_METHOD_ID, Q1_METHOD_ID)
BLOCKS = (13, 20, 27)
COMPARISONS = ("same_minus_null", "same_minus_full")
ENDPOINTS = ("target_margin", "ndcg@10")
FAMILY_SIZE = 96


@dataclass(frozen=True)
class BreadthBundle:
    root: Path
    metadata: dict[str, Any]
    scores: dict[str, dict[str, dict[str, float]]]


def evaluate_breadth_branches(
    standardized_dir: str | Path,
    bundle_dirs: Mapping[str, Mapping[int, str | Path]],
    output_dir: str | Path,
    analysis_run_id: str,
    *,
    dev_eval_log_path: str | Path = "reports/dev_eval_log.jsonl",
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Evaluate Q0/Q1 without parameter-matched claims or pointwise Q1 proxy."""

    if set(bundle_dirs) != set(METHODS) or any(
        set(map(int, blocks)) != set(BLOCKS) for blocks in bundle_dirs.values()
    ):
        raise ValueError("breadth evaluator requires Q0/Q1 x blocks 13/20/27")
    standardized_dir = Path(standardized_dir)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"breadth branch output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = standardized_dir / "records_dev.jsonl"
    qrels_path = standardized_dir / "qrels_dev.jsonl"
    raw_records = list(iter_jsonl(records_path))
    records = [sanitize_record_for_model(row) for row in raw_records]
    if len(records) != 8000:
        raise ValueError("breadth branch evaluator requires full dev")
    candidates = _audit_candidate_and_request_manifests(
        standardized_dir / "candidate_manifest.json",
        standardized_dir / "request_manifest.json",
        records,
        raw_records,
    )
    bundles = {
        method_id: {
            block: _audit_bundle(
                bundle_dirs[method_id][block], records, method_id, block
            )
            for block in BLOCKS
        }
        for method_id in METHODS
    }
    implementation_digests = _per_method_implementation_digests(bundles)
    pre_qrels = {
        "schema_version": 1,
        "analysis_type": "d6_q0_q1_branch_pre_qrels_integrity",
        "analysis_run_id": analysis_run_id,
        "status": "passed",
        "qrels_read": False,
        "checks": {
            "all_six_registered_bundles_present": True,
            "all_requests_candidates_conditions_complete_finite": True,
            "all_four_node_identities_at_most_1e-5": True,
            "frozen_baseline_recompute_within_path_local_bf16_bound": True,
            "Q1_native_listwise_prefix_cache_and_all_response_tokens_attested": True,
            "candidate_and_request_manifests_reconstructed": True,
            "each_model_three_blocks_share_one_implementation_digest": True,
        },
        "implementation_digests": implementation_digests,
        "bundles": {
            method_id: {
                str(block): {
                    "path": str(bundle.root),
                    "metadata_sha256": sha256_file(bundle.root / "metadata.json"),
                    "scores_sha256": sha256_file(bundle.root / "scores.jsonl"),
                }
                for block, bundle in blocks.items()
            }
            for method_id, blocks in bundles.items()
        },
    }
    pre_qrels_path = output_dir / "pre_qrels_audit.json"
    _write_json(pre_qrels_path, pre_qrels)
    frozen = load_m2_probe_manifest()["frozen_inputs"]
    qrels_sha256 = sha256_file(qrels_path)
    if qrels_sha256 != frozen["qrels_dev_sha256"]:
        raise ValueError("breadth branch qrels hash differs")
    gains = _load_dev_qrels(qrels_path, candidates)
    strict = np.asarray(
        [_strict_transfer(record, candidates, gains) for record in records], dtype=bool
    )
    if int(strict.sum()) != 2152:
        raise ValueError("breadth branch strict-transfer request count differs")
    request_ids = [record.request_id for record in records]
    clusters = np.asarray([normalize_query(record.query) for record in records], dtype=np.str_)
    folds = np.asarray([normalized_query_fold(record.query) for record in records], dtype=np.int8)
    family_rows = []
    results = {}
    per_request = {}
    for method_id in METHODS:
        results[method_id] = {}
        for block in BLOCKS:
            bundle = bundles[method_id][block]
            endpoints = {
                condition: {
                    "target_margin": _target_margins(
                        request_ids, candidates, gains, bundle.scores[condition]
                    ),
                    "ndcg@10": _ndcg_values(
                        request_ids, candidates, gains, bundle.scores[condition]
                    ),
                }
                for condition in bundle.scores
            }
            block_results = {}
            for node in Q0_BRANCH_NODES:
                same = f"{node}_same_to_null"
                block_results[node] = {}
                for comparison in COMPARISONS:
                    reference = "baseline_null" if comparison == "same_minus_null" else "baseline_full"
                    block_results[node][comparison] = {}
                    for endpoint in ENDPOINTS:
                        values = endpoints[same][endpoint] - endpoints[reference][endpoint]
                        inference = []
                        for fold_name, fold_mask in (
                            ("all", np.ones(len(records), dtype=bool)),
                            ("0", folds == 0),
                            ("1", folds == 1),
                        ):
                            mask = strict & fold_mask & np.isfinite(values)
                            inference.append({
                                "surface": STRICT_TRANSFER_SURFACE,
                                "normalized_query_fold": fold_name,
                                **cluster_mean_inference(values[mask], clusters[mask]),
                            })
                        all_row = next(row for row in inference if row["normalized_query_fold"] == "all")
                        family_rows.append({
                            "method_id": method_id,
                            "block_zero_based": block,
                            "node": node,
                            "comparison": comparison,
                            "endpoint": endpoint,
                            "two_sided_p": float(all_row["two_sided_p"]),
                        })
                        block_results[node][comparison][endpoint] = inference
                        per_request[f"{method_id}__b{block}__{node}__{comparison}__{endpoint}"] = values
            results[method_id][str(block)] = block_results
    if len(family_rows) != FAMILY_SIZE:
        raise AssertionError("Q0/Q1 breadth family size is not 96")
    q_values = benjamini_hochberg([row["two_sided_p"] for row in family_rows])
    for row, q_value in zip(family_rows, q_values):
        row["bh_q"] = float(q_value)
        all_row = next(
            value
            for value in results[row["method_id"]][str(row["block_zero_based"])][row["node"]][row["comparison"]][row["endpoint"]]
            if value["normalized_query_fold"] == "all"
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
        "analysis_type": "transformer_deep_dive_d6_q0_q1_branch_extension",
        "analysis_run_id": analysis_run_id,
        "methods": list(METHODS),
        "blocks": list(BLOCKS),
        "nodes": list(Q0_BRANCH_NODES),
        "comparisons": list(COMPARISONS),
        "endpoints": list(ENDPOINTS),
        "primary_surface": STRICT_TRANSFER_SURFACE,
        "implementation_digests": implementation_digests,
        "strict_transfer_requests": int(strict.sum()),
        "multiple_testing": {
            "family_size": FAMILY_SIZE,
            "method": "benjamini_hochberg",
        },
        "family_rows": family_rows,
        "results": results,
        "q0_specialized_pretraining_boundary": "reported separately; no parameter-matched claim",
        "q1_native_boundary": "complete listwise slate, prefix KV cache, and every multi-token response state",
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
            "method_ids": list(METHODS),
            "split": "dev",
            "qrels_sha256": qrels_sha256,
            "metrics_path": str(metrics_path),
            "metrics_sha256": sha256_file(metrics_path),
        },
    )
    return metrics


def _audit_bundle(root, records, method_id, block):
    root = Path(root)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    conditions = Q0_BRANCH_CONDITIONS if method_id == Q0_METHOD_ID else Q1_BRANCH_CONDITIONS
    expected = {
        "analysis_stage": (
            "transformer_deep_dive_d6_q0_branch_extension"
            if method_id == Q0_METHOD_ID
            else "transformer_deep_dive_d6_q1_branch_extension"
        ),
        "method_id": method_id,
        "block_zero_based": block,
        "status": "completed",
        "result_eligible": True,
        "identity_passed": True,
        "complete_finite_score_coverage": True,
        "qrels_read": False,
        "source_test_opened": False,
        "score_conditions": list(conditions),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"breadth branch metadata differs: {key}")
    if float(metadata.get("maximum_identity_delta", math.inf)) > 1.0e-5:
        raise ValueError("breadth branch identity failed")
    if float(metadata.get("maximum_baseline_low_precision_ratio", math.inf)) > 1.0:
        raise ValueError("breadth branch baseline bound failed")
    if method_id == Q1_METHOD_ID and metadata.get("patch_scope") != "prompt readout and every response token":
        raise ValueError("Q1 branch token scope differs")
    path = root / "scores.jsonl"
    if metadata.get("scores_sha256") != sha256_file(path):
        raise ValueError("breadth branch scores hash differs")
    observed = audit_scalar_partial(path, records, conditions)
    if observed["completed_requests"] != len(records):
        raise ValueError("breadth branch request coverage differs")
    scores = {condition: {} for condition in conditions}
    for block_row in iter_jsonl(path):
        request_id = str(block_row["request_id"])
        if method_id == Q1_METHOD_ID:
            call_audit = block_row.get("call_audit", {})
            if (
                int(block_row.get("response_tokens", 0)) <= 0
                or call_audit.get("full_capture", {}).get("all_response_tokens_captured") is not True
                or call_audit.get("null_baseline", {}).get("continuation_calls", 0) <= 0
                or any(
                    value.get("same_to_null", {}).get("all_response_tokens_patched") is not True
                    or value.get("full_identity", {}).get("all_response_tokens_patched") is not True
                    for value in call_audit.get("patched", {}).values()
                )
                or set(call_audit.get("patched", {})) != set(Q0_BRANCH_NODES)
            ):
                raise ValueError("Q1 branch call/token audit differs")
        for condition in conditions:
            scores[condition][request_id] = {
                str(row["candidate_item_id"]): float(row["conditions"][condition])
                for row in block_row["rows"]
            }
    return BreadthBundle(root, metadata, scores)


def _per_method_implementation_digests(bundles):
    result = {}
    for method_id, blocks in bundles.items():
        metadata_rows = [bundle.metadata for bundle in blocks.values()]
        digests = {
            str(metadata.get("implementation_identity", {}).get("digest") or "")
            for metadata in metadata_rows
        }
        if len(digests) != 1 or not next(iter(digests), ""):
            raise ValueError(
                f"breadth branch bundles use different implementation digests: {method_id}"
            )
        digest = next(iter(digests))
        if any(
            metadata.get("run_contract", {}).get("implementation_digest") != digest
            for metadata in metadata_rows
        ):
            raise ValueError(
                f"breadth branch implementation differs from run contract: {method_id}"
            )
        result[method_id] = digest
    return result


def _strict_transfer(record, candidates, gains):
    history = {str(row["item_id"]) for row in record.history}
    slate = set(candidates[record.request_id])
    positive = any(float(value) > 0 for value in gains[record.request_id].values())
    return bool(history) and history.isdisjoint(slate) and positive
