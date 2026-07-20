"""Fold-1 shared evaluation for D2 selected-block Transformer branches."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from myrec.baselines.motivation_v12_contracts import sanitize_record_for_model
from myrec.mechanism.attention_edge_evaluator import _append_jsonl, _write_json
from myrec.mechanism.attention_edge_runtime import _load_frozen_baseline
from myrec.mechanism.deep_dive_native_evaluator import cluster_mean_inference
from myrec.mechanism.fold_qrels import audit_fold_qrels
from myrec.mechanism.postblock_sweep_evaluator import (
    _load_fold_qrels,
    _ndcg,
    _strict_transfer_mask,
    _target_margins,
)
from myrec.mechanism.representation_evaluator import (
    _audit_candidate_and_request_manifests,
)
from myrec.mechanism.representation_probe import normalize_query, normalized_query_fold
from myrec.mechanism.scalar_condition_bundle import audit_scalar_partial
from myrec.mechanism.selected_branch_scoring import (
    SELECTED_NODES,
    selected_branch_conditions,
)
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl


ENDPOINTS = ("target_margin", "ndcg@10")
CONTRAST_GROUPS = {
    "same": 7,
    "same_minus_cross": 7,
    "same_minus_wrong_history": 7,
    "adjacent_node": 6,
    "direction_scale": 21,
}
SELECTED_BRANCH_FOLD_SCOPE = {
    "selection_fold": 0,
    "node_inference_fold": 1,
    "fold0_role": "postblock_adjacent_transition_selection_only",
    "fold1_role": (
        "postblock_transition_confirmation_and_selected_branch_node_inference"
    ),
    "node_effect_two_fold_replication_tested": False,
    "split_sample_component_localization": True,
    "claim_boundary": (
        "The adjacent post-block transition is selected on fold 0 and confirmed "
        "on fold 1; seven-node component effects, p-values, and BH gates use "
        "fold 1 only and are not themselves two-fold replications."
    ),
}


@dataclass(frozen=True)
class SelectedBranchBundle:
    root: Path
    metadata: dict[str, Any]
    scores: dict[str, dict[str, dict[str, float]]]
    wrong_eligible: dict[str, bool]


def evaluate_selected_branches(
    standardized_dir: str | Path,
    qrels_split_dir: str | Path,
    bundle_dir: str | Path,
    output_dir: str | Path,
    analysis_run_id: str,
    *,
    dev_eval_log_path: str | Path = "reports/dev_eval_log.jsonl",
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Evaluate every frozen contrast on fold 1 after bundle integrity gates."""

    standardized_dir = Path(standardized_dir)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"selected-branch evaluation output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = standardized_dir / "records_dev.jsonl"
    raw_records = list(iter_jsonl(records_path))
    all_records = [sanitize_record_for_model(row) for row in raw_records]
    if len(all_records) != 8000:
        raise ValueError("selected-branch evaluator requires frozen 8000-request dev")
    all_candidates = _audit_candidate_and_request_manifests(
        standardized_dir / "candidate_manifest.json",
        standardized_dir / "request_manifest.json",
        all_records,
        raw_records,
    )
    records = [record for record in all_records if normalized_query_fold(record.query) == 1]
    candidates = {record.request_id: all_candidates[record.request_id] for record in records}
    bundle = _audit_selected_branch_bundle(
        bundle_dir,
        records,
        all_records=all_records,
    )
    implementation_digest = _selected_branch_implementation_digest(bundle.metadata)
    pre_qrels = {
        "schema_version": 1,
        "analysis_type": "d2_selected_branch_fold1_pre_qrels_integrity",
        "analysis_run_id": analysis_run_id,
        "status": "passed",
        "method_id": bundle.metadata["method_id"],
        "selected_block": bundle.metadata["selected_block"],
        "qrels_read": False,
        "checks": {
            "fold1_request_candidate_coverage_complete_finite": True,
            "all_14_identity_controls_at_most_1e-5": True,
            "frozen_baseline_recompute_within_path_local_bf16_bound": True,
            "wrong_user_ineligible_scores_equal_frozen_null": True,
            "candidate_and_request_manifests_reconstructed": True,
            "minimal_selected_branch_contract_bound": True,
            "selected_branch_implementation_digest_bound": True,
        },
        "implementation_digest": implementation_digest,
        "bundle": {
            "path": str(bundle.root),
            "metadata_sha256": sha256_file(bundle.root / "metadata.json"),
            "scores_sha256": sha256_file(bundle.root / "scores.jsonl"),
        },
    }
    pre_qrels_path = output_dir / "pre_qrels_audit.json"
    _write_json(pre_qrels_path, pre_qrels)

    qrels_path, qrels_manifest = audit_fold_qrels(
        standardized_dir, qrels_split_dir, 1
    )
    gains = _load_fold_qrels(qrels_path, candidates)
    strict = _strict_transfer_mask(records, candidates, gains)
    request_ids = [record.request_id for record in records]
    clusters = np.asarray([normalize_query(record.query) for record in records], dtype=np.str_)
    wrong_eligible = np.asarray(
        [bundle.wrong_eligible[request_id] for request_id in request_ids], dtype=bool
    )
    endpoints = {
        condition: {
            "target_margin": _target_margins(
                request_ids, candidates, gains, bundle.scores[condition]
            ),
            "ndcg@10": _ndcg(
                request_ids, candidates, gains, bundle.scores[condition]
            ),
        }
        for condition in selected_branch_conditions()
    }
    contrast_specs = selected_branch_contrast_specs()
    results = {}
    family_rows = []
    per_request = {}
    for contrast_id, spec in contrast_specs.items():
        group = spec["group"]
        mask_base = strict & (wrong_eligible if group == "same_minus_wrong_history" else True)
        result = {
            "group": group,
            "node": spec.get("node"),
            "left_node": spec.get("left_node"),
            "right_node": spec.get("right_node"),
            "control": spec.get("control"),
            "eligible_surface": (
                "strict_transfer_and_frozen_wrong_user_eligible"
                if group == "same_minus_wrong_history"
                else "strict_transfer"
            ),
            "endpoints": {},
        }
        for endpoint in ENDPOINTS:
            values = endpoints[spec["left"]][endpoint] - endpoints[spec["right"]][endpoint]
            mask = mask_base & np.isfinite(values)
            inference = cluster_mean_inference(values[mask], clusters[mask])
            result["endpoints"][endpoint] = inference
            family_rows.append(
                {
                    "contrast_id": contrast_id,
                    "group": group,
                    "endpoint": endpoint,
                    "two_sided_p": float(inference["two_sided_p"]),
                    "mean": float(inference["mean"]),
                    "ci95": list(inference["ci95"]),
                }
            )
            per_request[f"{contrast_id}__{endpoint}"] = values
        results[contrast_id] = result
    _assert_group_counts(family_rows)
    per_request_path = output_dir / "per_request_contrasts.npz"
    np.savez(
        per_request_path,
        **per_request,
        request_ids=np.asarray(request_ids, dtype=np.str_),
        normalized_queries=clusters,
        strict_mask=strict,
        wrong_user_eligible_mask=wrong_eligible,
    )
    metrics = {
        "schema_version": 1,
        "analysis_type": "transformer_deep_dive_d2_selected_branch",
        "analysis_run_id": analysis_run_id,
        "method_id": bundle.metadata["method_id"],
        "checkpoint_id": bundle.metadata["checkpoint_id"],
        "selected_block": bundle.metadata["selected_block"],
        "evidence_role": bundle.metadata["evidence_role"],
        "implementation_digest": implementation_digest,
        "normalized_query_fold": 1,
        "fold_scope": dict(SELECTED_BRANCH_FOLD_SCOPE),
        "strict_transfer_requests": int(strict.sum()),
        "strict_transfer_wrong_user_eligible_requests": int(
            (strict & wrong_eligible).sum()
        ),
        "bootstrap": {"cluster": "normalized_query", "samples": 5000, "seed": 20260715},
        "family_policy": {
            "BH_applied_only_in_two_model_synthesis": True,
            "per_endpoint_separate_families": True,
            "planned_two_model_units": {
                "same": 14,
                "same_minus_cross": 14,
                "same_minus_wrong_history": 14,
                "adjacent_node": 12,
                "direction_scale": 42,
            },
        },
        "results": results,
        "family_rows": family_rows,
        "input_bundle": pre_qrels["bundle"],
        "pre_qrels_audit_path": str(pre_qrels_path),
        "pre_qrels_audit_sha256": sha256_file(pre_qrels_path),
        "qrels_read": True,
        "qrels_fold_opened": 1,
        "other_fold_qrels_opened": False,
        "qrels_fold_sha256": sha256_file(qrels_path),
        "qrels_split_manifest_sha256": sha256_file(Path(qrels_split_dir) / "manifest.json"),
        "qrels_source_sha256": qrels_manifest["source_qrels_sha256"],
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
            "method_ids": [metrics["method_id"]],
            "split": "dev_fold1",
            "qrels_sha256": metrics["qrels_fold_sha256"],
            "metrics_path": str(metrics_path),
            "metrics_sha256": sha256_file(metrics_path),
        },
    )
    return metrics


def selected_branch_contrast_specs() -> dict[str, dict[str, Any]]:
    specs = {}
    for node in SELECTED_NODES:
        same = f"{node}.same_full_to_null"
        specs[f"same__{node}"] = {
            "group": "same",
            "node": node,
            "left": same,
            "right": "baseline_null",
        }
        specs[f"same_minus_cross__{node}"] = {
            "group": "same_minus_cross",
            "node": node,
            "left": same,
            "right": f"{node}.cross_full_to_null",
        }
        specs[f"same_minus_wrong__{node}"] = {
            "group": "same_minus_wrong_history",
            "node": node,
            "left": same,
            "right": f"{node}.wrong_history_to_null",
        }
        for short, condition in (
            ("norm", "donor_direction_at_recipient_rms"),
            ("direction", "recipient_direction_at_donor_rms"),
            ("random", "random_direction_at_recipient_rms"),
        ):
            specs[f"{short}__{node}"] = {
                "group": "direction_scale",
                "node": node,
                "control": condition,
                "left": same,
                "right": f"{node}.{condition}",
            }
    for left, right in zip(SELECTED_NODES[:-1], SELECTED_NODES[1:]):
        specs[f"adjacent__{left}__to__{right}"] = {
            "group": "adjacent_node",
            "left_node": left,
            "right_node": right,
            "left": f"{right}.same_full_to_null",
            "right": f"{left}.same_full_to_null",
        }
    return specs


def _audit_selected_branch_bundle(
    bundle_dir: str | Path,
    records: Sequence[Any],
    *,
    all_records: Sequence[Any],
) -> SelectedBranchBundle:
    root = Path(bundle_dir)
    metadata = _read_json(root / "metadata.json")
    expected = {
        "analysis_stage": "transformer_deep_dive_d2_selected_branch",
        "status": "completed",
        "result_eligible": True,
        "complete_finite_score_coverage": True,
        "identity_passed": True,
        "normalized_query_fold": 1,
        "selected_nodes": list(SELECTED_NODES),
        "score_conditions": list(selected_branch_conditions()),
        "qrels_read": False,
        "source_test_opened": False,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"selected-branch bundle metadata mismatch: {key}")
    if float(metadata.get("maximum_identity_delta", math.inf)) > 1.0e-5:
        raise ValueError("selected-branch identity bound failed")
    if float(metadata.get("maximum_baseline_low_precision_ratio", math.inf)) > 1.0:
        raise ValueError("selected-branch baseline BF16 bound failed")
    if metadata.get("method_id") == "q3_tallrec_generalqwen" and float(
        metadata.get("shared_prompt_path_max_abs_delta", math.inf)
    ) != 0.0:
        raise ValueError("selected-branch Q3 shared prompt path identity failed")
    branch_contract = metadata.get("branch_contract")
    if not isinstance(branch_contract, dict):
        raise ValueError("selected-branch bundle lacks minimal branch contract")
    contract_path = Path(branch_contract["path"])
    if sha256_file(contract_path) != branch_contract.get("sha256"):
        raise ValueError("selected-branch contract bytes changed")
    contract = _read_json(contract_path)
    if (
        contract.get("contract_type")
        != "transformer_deep_dive_d2_selected_branch_contract"
        or contract.get("selected_block") != metadata.get("selected_block")
        or contract.get("method_id") != metadata.get("method_id")
        or contract.get("checkpoint_id") != metadata.get("checkpoint_id")
        or contract.get("evidence_role") != metadata.get("evidence_role")
        or contract.get("qrels_values_exposed_to_scorer") is not False
    ):
        raise ValueError("selected-branch minimal contract binding drift")
    _audit_selected_branch_source_contract(contract, metadata)
    frozen_null_identity = metadata.get("frozen_null_baseline")
    if not isinstance(frozen_null_identity, dict):
        raise ValueError("selected-branch bundle lacks frozen null identity")
    frozen_null, observed_null_identity = _load_frozen_baseline(
        Path(str(frozen_null_identity.get("root") or "")),
        str(metadata["method_id"]),
        str(metadata["checkpoint_id"]),
        all_records,
    )
    if observed_null_identity != frozen_null_identity:
        raise ValueError("selected-branch frozen null bytes changed")
    scores_path = root / "scores.jsonl"
    if metadata.get("scores_sha256") != sha256_file(scores_path):
        raise ValueError("selected-branch score bytes changed")
    observed = audit_scalar_partial(
        scores_path, records, selected_branch_conditions()
    )
    if (
        observed["completed_requests"] != len(records)
        or observed["completed_score_rows"]
        != sum(len(record.candidates) for record in records)
    ):
        raise ValueError("selected-branch request/candidate coverage drift")
    scores = {name: {} for name in selected_branch_conditions()}
    wrong_eligible = {}
    for block_row in iter_jsonl(scores_path):
        request_id = str(block_row["request_id"])
        eligible = block_row.get("wrong_user_eligible") is True
        wrong_eligible[request_id] = eligible
        for condition in selected_branch_conditions():
            scores[condition][request_id] = {
                str(row["candidate_item_id"]): float(row["conditions"][condition])
                for row in block_row["rows"]
            }
        if any((row.get("wrong_user_eligible") is True) != eligible for row in block_row["rows"]):
            raise ValueError("selected-branch wrong-user eligibility differs within request")
        if not eligible:
            _audit_ineligible_wrong_user_rows(
                request_id,
                block_row["rows"],
                frozen_null,
            )
    return SelectedBranchBundle(root, metadata, scores, wrong_eligible)


def _audit_ineligible_wrong_user_rows(
    request_id: str,
    rows: Sequence[Mapping[str, Any]],
    frozen_null: Mapping[tuple[str, str], float],
) -> None:
    """Check the registered frozen-null fallback, not a BF16 recomputation."""

    for row in rows:
        candidate_id = str(row["candidate_item_id"])
        key = (request_id, candidate_id)
        if key not in frozen_null:
            raise ValueError("ineligible wrong-user frozen null key is missing")
        expected = float(frozen_null[key])
        if any(
            float(row["conditions"][f"{node}.wrong_history_to_null"])
            != expected
            for node in SELECTED_NODES
        ):
            raise ValueError(
                "ineligible wrong-user condition does not copy frozen null"
            )


def _selected_branch_implementation_digest(metadata: Mapping[str, Any]) -> str:
    digest = str(metadata.get("implementation_identity", {}).get("digest") or "")
    if not digest:
        raise ValueError("selected-branch implementation digest is missing")
    if metadata.get("run_contract", {}).get("implementation_digest") != digest:
        raise ValueError("selected-branch implementation digest differs from run contract")
    return digest


def _audit_selected_branch_source_contract(
    contract: Mapping[str, Any], metadata: Mapping[str, Any]
) -> None:
    selection_identity = contract.get("selection")
    confirmation_identity = contract.get("confirmation")
    if not isinstance(selection_identity, dict) or not isinstance(
        confirmation_identity, dict
    ):
        raise ValueError("selected-branch contract lacks fold source identities")
    selection_path = Path(str(selection_identity.get("path") or ""))
    confirmation_path = Path(str(confirmation_identity.get("path") or ""))
    if (
        not selection_path.is_file()
        or sha256_file(selection_path) != selection_identity.get("sha256")
        or not confirmation_path.is_file()
        or sha256_file(confirmation_path) != confirmation_identity.get("sha256")
    ):
        raise ValueError("selected-branch fold source bytes changed")
    selection = _read_json(selection_path)
    confirmation = _read_json(confirmation_path)
    digest = str(contract.get("postblock_implementation_digest") or "")
    if (
        not digest
        or selection.get("implementation_digest") != digest
        or confirmation.get("implementation_digest") != digest
        or selection.get("analysis_type")
        != "transformer_deep_dive_d2_postblock_fold0_selection"
        or confirmation.get("analysis_type")
        != "transformer_deep_dive_d2_postblock_fold1_confirmation"
        or selection.get("method_id") != metadata.get("method_id")
        or confirmation.get("method_id") != metadata.get("method_id")
        or selection.get("checkpoint_id") != metadata.get("checkpoint_id")
        or confirmation.get("checkpoint_id") != metadata.get("checkpoint_id")
        or selection.get("selected_block") != metadata.get("selected_block")
        or confirmation.get("selection") != selection_identity
    ):
        raise ValueError("selected-branch fold source binding drift")


def _assert_group_counts(rows: Sequence[Mapping[str, Any]]) -> None:
    for endpoint in ENDPOINTS:
        counts = {
            group: sum(
                row["group"] == group and row["endpoint"] == endpoint for row in rows
            )
            for group in CONTRAST_GROUPS
        }
        if counts != CONTRAST_GROUPS:
            raise AssertionError(f"selected-branch registered family count drift: {counts}")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value
