"""Leakage-safe fold-0 selection and fold-1 confirmation for D2 sweeps."""

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
from myrec.mechanism.attention_edge_evaluator import _append_jsonl
from myrec.mechanism.deep_dive_native_evaluator import cluster_mean_inference
from myrec.mechanism.fold_qrels import audit_fold_qrels
from myrec.mechanism.postblock_sweep_runtime import SUPPORTED_METHODS
from myrec.mechanism.postblock_sweep_scoring import POSTBLOCK_CONDITIONS
from myrec.mechanism.representation_evaluator import (
    _audit_candidate_and_request_manifests,
)
from myrec.mechanism.representation_probe import (
    normalize_query,
    normalized_query_fold,
)
from myrec.mechanism.scalar_condition_bundle import audit_scalar_partial
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl


POSTBLOCK_BLOCKS = tuple(range(13, 28))
ENDPOINTS = ("target_margin", "ndcg@10")


@dataclass(frozen=True)
class PostblockBundle:
    root: Path
    metadata: dict[str, Any]
    scores: dict[str, dict[str, dict[str, float]]]


def select_postblock_fold0(
    standardized_dir: str | Path,
    qrels_split_dir: str | Path,
    method_id: str,
    bundle_dirs: Mapping[int, str | Path],
    output_dir: str | Path,
    analysis_run_id: str,
    *,
    dev_eval_log_path: str | Path = "reports/dev_eval_log.jsonl",
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Read only fold-0 qrels, select j, and atomically freeze the selection."""

    context = _prepare_fold_context(
        standardized_dir, method_id, bundle_dirs, output_dir, analysis_run_id, fold=0
    )
    output_dir = Path(output_dir)
    pre_qrels_path = output_dir / "pre_qrels_audit.json"
    _write_json_atomic(pre_qrels_path, context["pre_qrels"])
    qrels_path, qrels_manifest = audit_fold_qrels(
        standardized_dir, qrels_split_dir, 0
    )
    gains = _load_fold_qrels(qrels_path, context["candidates"])
    strict = _strict_transfer_mask(context["records"], context["candidates"], gains)
    effects, arrays = _fold_effects(
        context["records"], context["candidates"], gains, strict, context["bundles"]
    )
    margin_sequence = {
        block: float(effects[str(block)]["target_margin"]["mean"])
        for block in POSTBLOCK_BLOCKS
    }
    selected_block, adjacent = select_registered_block(margin_sequence)
    per_request_path = output_dir / "per_request_effects.npz"
    np.savez(
        per_request_path,
        **arrays,
        request_ids=np.asarray([record.request_id for record in context["records"]], dtype=np.str_),
        normalized_queries=np.asarray(
            [normalize_query(record.query) for record in context["records"]], dtype=np.str_
        ),
        strict_mask=strict,
    )
    selection = {
        "schema_version": 1,
        "analysis_type": "transformer_deep_dive_d2_postblock_fold0_selection",
        "analysis_run_id": analysis_run_id,
        "method_id": method_id,
        "checkpoint_id": context["checkpoint_id"],
        "implementation_digest": context["implementation_digest"],
        "fold": 0,
        "registered_blocks": list(POSTBLOCK_BLOCKS),
        "selector": "argmin_k_in_14_to_27(E_k-E_k_minus_1); tie_lower_k",
        "selected_block": selected_block,
        "selected_predecessor": None if selected_block is None else selected_block - 1,
        "minimum_adjacent_margin_step": (
            None if selected_block is None else float(adjacent[selected_block])
        ),
        "negative_transition_present": selected_block is not None,
        "selection_frozen_before_fold1": True,
        "strict_transfer_requests": int(strict.sum()),
        "fold_request_count": len(context["records"]),
        "effects": effects,
        "adjacent_margin_steps": {str(key): value for key, value in adjacent.items()},
        "input_bundles": context["bundle_identities"],
        "pre_qrels_audit_path": str(pre_qrels_path),
        "pre_qrels_audit_sha256": sha256_file(pre_qrels_path),
        "qrels_read": True,
        "qrels_fold_opened": 0,
        "other_fold_qrels_opened": False,
        "qrels_fold_sha256": sha256_file(qrels_path),
        "qrels_split_manifest_sha256": sha256_file(Path(qrels_split_dir) / "manifest.json"),
        "qrels_source_sha256": qrels_manifest["source_qrels_sha256"],
        "per_request_effects_path": str(per_request_path),
        "per_request_effects_sha256": sha256_file(per_request_path),
        "command": list(command or []),
        "status": "completed",
    }
    selection_path = output_dir / "selection.json"
    _write_json_atomic(selection_path, selection)
    _append_jsonl(
        Path(dev_eval_log_path),
        {
            "analysis_type": selection["analysis_type"],
            "run_id": analysis_run_id,
            "method_ids": [method_id],
            "split": "dev_fold0",
            "qrels_sha256": selection["qrels_fold_sha256"],
            "metrics_path": str(selection_path),
            "metrics_sha256": sha256_file(selection_path),
        },
    )
    return selection


def confirm_postblock_fold1(
    standardized_dir: str | Path,
    qrels_split_dir: str | Path,
    selection_path: str | Path,
    bundle_dirs: Mapping[int, str | Path],
    output_dir: str | Path,
    analysis_run_id: str,
    *,
    dev_eval_log_path: str | Path = "reports/dev_eval_log.jsonl",
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Confirm the immutable fold-0 transition without allowing reselection."""

    selection_path = Path(selection_path)
    selection = _read_json(selection_path)
    method_id = str(selection.get("method_id") or "")
    if (
        selection.get("analysis_type")
        != "transformer_deep_dive_d2_postblock_fold0_selection"
        or selection.get("status") != "completed"
        or selection.get("selection_frozen_before_fold1") is not True
        or selection.get("fold") != 0
    ):
        raise ValueError("fold-1 confirmation received an invalid selection record")
    context = _prepare_fold_context(
        standardized_dir, method_id, bundle_dirs, output_dir, analysis_run_id, fold=1
    )
    if context["checkpoint_id"] != selection.get("checkpoint_id"):
        raise ValueError("fold-1 checkpoint differs from frozen fold-0 selection")
    if context["implementation_digest"] != selection.get("implementation_digest"):
        raise ValueError("fold-1 implementation differs from frozen fold-0 selection")
    selection_identity = {"path": str(selection_path), "sha256": sha256_file(selection_path)}
    for block, bundle in context["bundles"].items():
        if bundle.metadata.get("fold0_selection") != selection_identity:
            raise ValueError(f"fold-1 block {block} does not bind the frozen selection")
    output_dir = Path(output_dir)
    context["pre_qrels"]["selection"] = selection_identity
    context["pre_qrels"]["fold0_fold1_share_one_implementation_digest"] = True
    pre_qrels_path = output_dir / "pre_qrels_audit.json"
    _write_json_atomic(pre_qrels_path, context["pre_qrels"])
    qrels_path, qrels_manifest = audit_fold_qrels(
        standardized_dir, qrels_split_dir, 1
    )
    gains = _load_fold_qrels(qrels_path, context["candidates"])
    strict = _strict_transfer_mask(context["records"], context["candidates"], gains)
    effects, arrays = _fold_effects(
        context["records"], context["candidates"], gains, strict, context["bundles"]
    )
    margin_sequence = {
        block: float(effects[str(block)]["target_margin"]["mean"])
        for block in POSTBLOCK_BLOCKS
    }
    adjacent = {
        block: margin_sequence[block] - margin_sequence[block - 1]
        for block in POSTBLOCK_BLOCKS[1:]
    }
    selected_block = selection.get("selected_block")
    if selected_block is None:
        fixed_confirmation = {
            "applicable": False,
            "reason": "fold0_had_no_negative_adjacent_step",
            "confirmed_negative_transition": False,
        }
    else:
        selected_block = int(selected_block)
        fold0_step = float(selection["minimum_adjacent_margin_step"])
        fold1_step = float(adjacent[selected_block])
        fixed_confirmation = {
            "applicable": True,
            "selected_block": selected_block,
            "fold0_step": fold0_step,
            "fold1_step": fold1_step,
            "same_negative_sign_both_folds": fold0_step < 0 and fold1_step < 0,
            "confirmed_negative_transition": fold0_step < 0 and fold1_step < 0,
        }
    per_request_path = output_dir / "per_request_effects.npz"
    np.savez(
        per_request_path,
        **arrays,
        request_ids=np.asarray([record.request_id for record in context["records"]], dtype=np.str_),
        normalized_queries=np.asarray(
            [normalize_query(record.query) for record in context["records"]], dtype=np.str_
        ),
        strict_mask=strict,
    )
    metrics = {
        "schema_version": 1,
        "analysis_type": "transformer_deep_dive_d2_postblock_fold1_confirmation",
        "analysis_run_id": analysis_run_id,
        "method_id": method_id,
        "checkpoint_id": context["checkpoint_id"],
        "implementation_digest": context["implementation_digest"],
        "fold": 1,
        "registered_blocks": list(POSTBLOCK_BLOCKS),
        "selection": selection_identity,
        "selected_block_immutable": True,
        "fixed_transition_confirmation": fixed_confirmation,
        "strict_transfer_requests": int(strict.sum()),
        "fold_request_count": len(context["records"]),
        "effects": effects,
        "adjacent_margin_steps": {str(key): value for key, value in adjacent.items()},
        "input_bundles": context["bundle_identities"],
        "pre_qrels_audit_path": str(pre_qrels_path),
        "pre_qrels_audit_sha256": sha256_file(pre_qrels_path),
        "qrels_read": True,
        "qrels_fold_opened": 1,
        "other_fold_qrels_opened": False,
        "qrels_fold_sha256": sha256_file(qrels_path),
        "qrels_split_manifest_sha256": sha256_file(Path(qrels_split_dir) / "manifest.json"),
        "qrels_source_sha256": qrels_manifest["source_qrels_sha256"],
        "per_request_effects_path": str(per_request_path),
        "per_request_effects_sha256": sha256_file(per_request_path),
        "command": list(command or []),
        "status": "completed",
    }
    metrics_path = output_dir / "metrics.json"
    _write_json_atomic(metrics_path, metrics)
    _append_jsonl(
        Path(dev_eval_log_path),
        {
            "analysis_type": metrics["analysis_type"],
            "run_id": analysis_run_id,
            "method_ids": [method_id],
            "split": "dev_fold1",
            "qrels_sha256": metrics["qrels_fold_sha256"],
            "metrics_path": str(metrics_path),
            "metrics_sha256": sha256_file(metrics_path),
        },
    )
    return metrics


def select_registered_block(
    block_effects: Mapping[int, float],
) -> tuple[int | None, dict[int, float]]:
    """Apply the frozen negative-step selector with lower-block tie breaking."""

    if set(map(int, block_effects)) != set(POSTBLOCK_BLOCKS):
        raise ValueError("selector requires blocks 13 through 27")
    values = {int(key): float(value) for key, value in block_effects.items()}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("selector effects must be finite")
    adjacent = {
        block: values[block] - values[block - 1] for block in POSTBLOCK_BLOCKS[1:]
    }
    minimum = min(adjacent.values())
    if minimum >= 0:
        return None, adjacent
    selected = min(block for block, value in adjacent.items() if value == minimum)
    return selected, adjacent


def _prepare_fold_context(
    standardized_dir: str | Path,
    method_id: str,
    bundle_dirs: Mapping[int, str | Path],
    output_dir: str | Path,
    analysis_run_id: str,
    *,
    fold: int,
) -> dict[str, Any]:
    if method_id not in SUPPORTED_METHODS:
        raise ValueError("post-block evaluator supports only Q2/Q3")
    if set(map(int, bundle_dirs)) != set(POSTBLOCK_BLOCKS):
        raise ValueError("post-block evaluator requires all 15 registered blocks")
    standardized_dir = Path(standardized_dir)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"post-block evaluator output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_records = list(iter_jsonl(standardized_dir / "records_dev.jsonl"))
    all_records = [sanitize_record_for_model(row) for row in raw_records]
    if len(all_records) != 8000:
        raise ValueError("post-block evaluator requires frozen 8000-request dev")
    all_candidates = _audit_candidate_and_request_manifests(
        standardized_dir / "candidate_manifest.json",
        standardized_dir / "request_manifest.json",
        all_records,
        raw_records,
    )
    records = [record for record in all_records if normalized_query_fold(record.query) == fold]
    candidates = {record.request_id: all_candidates[record.request_id] for record in records}
    bundles = {
        block: _audit_postblock_bundle(bundle_dirs[block], records, method_id, block, fold)
        for block in POSTBLOCK_BLOCKS
    }
    implementation_digest = _common_implementation_digest(bundles)
    reference = bundles[13].metadata
    invariants = (
        "method_id", "checkpoint_id", "config_sha256", "records_sha256",
        "candidate_manifest_sha256", "request_manifest_sha256",
        "dataset_manifest_sha256", "deep_dive_manifest_sha256",
        "cross_request_mapping_sha256", "normalized_query_fold",
    )
    for block, bundle in bundles.items():
        for key in invariants:
            if bundle.metadata.get(key) != reference.get(key):
                raise ValueError(f"post-block invariant differs at block {block}: {key}")
    identities = {
        str(block): {
            "path": str(bundle.root),
            "metadata_sha256": sha256_file(bundle.root / "metadata.json"),
            "scores_sha256": sha256_file(bundle.root / "scores.jsonl"),
        }
        for block, bundle in bundles.items()
    }
    pre_qrels = {
        "schema_version": 1,
        "analysis_type": f"d2_postblock_fold{fold}_pre_qrels_integrity",
        "analysis_run_id": analysis_run_id,
        "method_id": method_id,
        "fold": fold,
        "status": "passed",
        "qrels_read": False,
        "checks": {
            "all_15_registered_blocks_present": True,
            "fold_only_request_candidate_coverage_complete_finite": True,
            "full_and_null_identity_at_most_1e-5": True,
            "frozen_baseline_recompute_within_path_local_bf16_bound": True,
            "candidate_and_request_manifests_reconstructed_before_qrels": True,
            "all_15_bundles_share_one_implementation_digest": True,
        },
        "implementation_digest": implementation_digest,
        "bundles": identities,
    }
    return {
        "records": records,
        "candidates": candidates,
        "bundles": bundles,
        "bundle_identities": identities,
        "checkpoint_id": reference["checkpoint_id"],
        "implementation_digest": implementation_digest,
        "pre_qrels": pre_qrels,
    }


def _audit_postblock_bundle(
    root: str | Path,
    records: Sequence[Any],
    method_id: str,
    block: int,
    fold: int,
) -> PostblockBundle:
    root = Path(root)
    metadata = _read_json(root / "metadata.json")
    expected = {
        "analysis_stage": "transformer_deep_dive_d2_postblock_sweep",
        "method_id": method_id,
        "block_zero_based": block,
        "normalized_query_fold": fold,
        "status": "completed",
        "result_eligible": True,
        "identity_passed": True,
        "complete_finite_score_coverage": True,
        "qrels_read": False,
        "source_test_opened": False,
        "score_conditions": list(POSTBLOCK_CONDITIONS),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"post-block bundle metadata mismatch at {block}: {key}")
    implementation_digest = str(
        metadata.get("implementation_identity", {}).get("digest") or ""
    )
    if not implementation_digest:
        raise ValueError("post-block implementation digest is missing")
    if metadata.get("run_contract", {}).get("implementation_digest") != implementation_digest:
        raise ValueError("post-block implementation digest differs from run contract")
    if float(metadata.get("maximum_identity_delta", math.inf)) > 1.0e-5:
        raise ValueError("post-block identity bound failed")
    if float(metadata.get("maximum_baseline_low_precision_ratio", math.inf)) > 1.0:
        raise ValueError("post-block frozen baseline BF16 bound failed")
    scores_path = root / "scores.jsonl"
    if metadata.get("scores_sha256") != sha256_file(scores_path):
        raise ValueError("post-block score hash differs")
    observed = audit_scalar_partial(scores_path, records, POSTBLOCK_CONDITIONS)
    if (
        observed["completed_requests"] != len(records)
        or observed["completed_score_rows"] != sum(len(record.candidates) for record in records)
    ):
        raise ValueError("post-block bundle coverage differs")
    scores = {name: {} for name in POSTBLOCK_CONDITIONS}
    for block_row in iter_jsonl(scores_path):
        request_id = str(block_row["request_id"])
        for name in POSTBLOCK_CONDITIONS:
            scores[name][request_id] = {
                str(row["candidate_item_id"]): float(row["conditions"][name])
                for row in block_row["rows"]
            }
    return PostblockBundle(root, metadata, scores)


def _common_implementation_digest(
    bundles: Mapping[int, PostblockBundle],
) -> str:
    digests = {
        str(bundle.metadata.get("implementation_identity", {}).get("digest") or "")
        for bundle in bundles.values()
    }
    if "" in digests or len(digests) != 1:
        raise ValueError("post-block bundles use different implementation digests")
    return next(iter(digests))


def _load_fold_qrels(
    path: Path, candidates: Mapping[str, Sequence[str]]
) -> dict[str, dict[str, float]]:
    result = {}
    for row in iter_jsonl(path):
        request_id = str(row.get("request_id") or "")
        if not request_id or request_id in result:
            raise ValueError("fold qrels contains empty/duplicate request identity")
        relevance = row.get("relevance") or {}
        if not isinstance(relevance, dict):
            raise ValueError("fold qrels relevance is not an object")
        gains = {str(key): float(value) for key, value in relevance.items() if float(value) > 0}
        if any(not math.isfinite(value) or value < 0 for value in gains.values()):
            raise ValueError("fold qrels contains invalid gain")
        result[request_id] = gains
    if set(result) != set(candidates):
        raise ValueError("fold qrels request coverage differs")
    if any(set(result[key]) - set(candidates[key]) for key in result):
        raise ValueError("fold qrels contains an out-of-slate item")
    return result


def _strict_transfer_mask(records, candidates, gains) -> np.ndarray:
    values = []
    for record in records:
        history = {str(row["item_id"]) for row in record.history}
        slate = set(candidates[record.request_id])
        positive = any(float(value) > 0 for value in gains[record.request_id].values())
        values.append(bool(history) and history.isdisjoint(slate) and positive)
    return np.asarray(values, dtype=bool)


def _fold_effects(records, candidates, gains, strict, bundles):
    request_ids = [record.request_id for record in records]
    clusters = np.asarray([normalize_query(record.query) for record in records], dtype=np.str_)
    effects = {}
    arrays = {}
    for block in POSTBLOCK_BLOCKS:
        bundle = bundles[block]
        margin_same = _target_margins(request_ids, candidates, gains, bundle.scores["same_full_to_null"])
        margin_null = _target_margins(request_ids, candidates, gains, bundle.scores["baseline_null"])
        ndcg_same = _ndcg(request_ids, candidates, gains, bundle.scores["same_full_to_null"])
        ndcg_null = _ndcg(request_ids, candidates, gains, bundle.scores["baseline_null"])
        block_effects = {
            "target_margin": margin_same - margin_null,
            "ndcg@10": ndcg_same - ndcg_null,
        }
        effects[str(block)] = {}
        for endpoint, values in block_effects.items():
            mask = strict & np.isfinite(values)
            effects[str(block)][endpoint] = cluster_mean_inference(values[mask], clusters[mask])
            arrays[f"block_{block}__{endpoint}"] = values
    return effects, arrays


def _target_margins(request_ids, candidates, gains, scores):
    result = np.full(len(request_ids), np.nan, dtype=np.float64)
    for ordinal, request_id in enumerate(request_ids):
        item_ids = list(candidates[request_id])
        gain_values = [float(gains[request_id].get(item_id, 0.0)) for item_id in item_ids]
        maximum = max(gain_values, default=0.0)
        lower = [index for index, value in enumerate(gain_values) if value < maximum]
        if maximum <= 0 or not lower:
            continue
        target = next(index for index, value in enumerate(gain_values) if value == maximum)
        competitor = max(lower, key=lambda index: scores[request_id][item_ids[index]])
        result[ordinal] = scores[request_id][item_ids[target]] - scores[request_id][item_ids[competitor]]
    return result


def _ndcg(request_ids, candidates, gains, scores):
    result = np.full(len(request_ids), np.nan, dtype=np.float64)
    for ordinal, request_id in enumerate(request_ids):
        item_ids = list(candidates[request_id])
        if not any(float(gains[request_id].get(item_id, 0.0)) > 0 for item_id in item_ids):
            continue
        result[ordinal] = gain_ndcg_at_k(
            request_id,
            item_ids,
            [float(scores[request_id][item_id]) for item_id in item_ids],
            [float(gains[request_id].get(item_id, 0.0)) for item_id in item_ids],
            10,
        )
    return result


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
