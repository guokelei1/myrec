"""Shared fold-1 evaluator for reverse component-state removal."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from myrec.baselines.motivation_v12_contracts import sanitize_record_for_model
from myrec.mechanism.attention_edge_evaluator import (
    _append_jsonl,
    _load_bundle_content_control_eligibility,
    _load_bundle_frozen_baseline,
    _write_json,
)
from myrec.mechanism.component_necessity_runtime import (
    EXTENSION_MANIFEST_PATH,
    _load_extension_manifest,
)
from myrec.mechanism.component_necessity_scoring import (
    NECESSITY_NODES,
    component_necessity_conditions,
)
from myrec.mechanism.deep_dive_native_evaluator import (
    benjamini_hochberg,
    cluster_mean_inference,
)
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
from myrec.mechanism.selected_branch_scoring import SELECTED_NODES
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl


METHODS = ("q2_recranker_generalqwen", "q3_tallrec_generalqwen")
ENDPOINTS = ("target_margin", "ndcg@10")
DONOR_MODES = ("neutral", "null")
FAMILY_SIZE_PER_ENDPOINT = 16
NDCG_EQUIVALENCE_BAND = (-0.005, 0.005)


@dataclass(frozen=True)
class NecessityBundle:
    root: Path
    metadata: dict[str, Any]
    scores: dict[str, dict[str, dict[str, float]]]
    content_neutral_eligible: dict[str, bool]


def evaluate_component_necessity(
    standardized_dir: str | Path,
    qrels_split_dir: str | Path,
    model_inputs: Mapping[str, Mapping[str, str | Path]],
    output_dir: str | Path,
    analysis_run_id: str,
    *,
    extension_manifest_path: str | Path = EXTENSION_MANIFEST_PATH,
    dev_eval_log_path: str | Path = "reports/dev_eval_log.jsonl",
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Evaluate both models, retaining gate-stopped cells as planned p=1 units."""

    if set(model_inputs) != set(METHODS):
        raise ValueError("component-necessity evaluator requires explicit Q2 and Q3")
    for method_id, value in model_inputs.items():
        if set(value) not in ({"bundle"}, {"gate_contract"}):
            raise ValueError(
                f"{method_id} requires exactly one of bundle or gate_contract"
            )
    standardized_dir = Path(standardized_dir)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"component-necessity output is not empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    extension_manifest = _load_extension_manifest(extension_manifest_path)
    records_path = standardized_dir / "records_dev.jsonl"
    raw_records = list(iter_jsonl(records_path))
    all_records = [sanitize_record_for_model(row) for row in raw_records]
    if len(all_records) != 8000:
        raise ValueError("component-necessity evaluator requires frozen 8000-request dev")
    all_candidates = _audit_candidate_and_request_manifests(
        standardized_dir / "candidate_manifest.json",
        standardized_dir / "request_manifest.json",
        all_records,
        raw_records,
    )
    records = [
        record for record in all_records if normalized_query_fold(record.query) == 1
    ]
    candidates = {record.request_id: all_candidates[record.request_id] for record in records}

    bundles: dict[str, NecessityBundle] = {}
    stopped: dict[str, dict[str, Any]] = {}
    input_identities = {}
    for method_id in METHODS:
        source = model_inputs[method_id]
        if "bundle" in source:
            bundle = _audit_bundle(
                source["bundle"],
                records,
                all_records=all_records,
                method_id=method_id,
                extension_manifest=extension_manifest,
            )
            bundles[method_id] = bundle
            input_identities[method_id] = {
                "status": "completed_bundle",
                "path": str(bundle.root),
                "metadata_sha256": sha256_file(bundle.root / "metadata.json"),
                "scores_sha256": sha256_file(bundle.root / "scores.jsonl"),
            }
        else:
            gate = _audit_gate_contract(
                source["gate_contract"],
                method_id=method_id,
                expected_checkpoint_id=extension_manifest["frozen_inputs"]["models"][
                    method_id
                ]["checkpoint_id"],
            )
            stopped[method_id] = gate
            gate_path = Path(source["gate_contract"])
            input_identities[method_id] = {
                "status": "gate_stopped",
                "path": str(gate_path),
                "sha256": sha256_file(gate_path),
                "evidence_role": gate["evidence_role"],
            }
    pre_qrels = {
        "schema_version": 1,
        "analysis_type": "component_necessity_pre_qrels_integrity",
        "analysis_run_id": analysis_run_id,
        "status": "passed",
        "qrels_read": False,
        "checks": {
            "each_model_has_exactly_one_completed_bundle_or_gate_stop": True,
            "completed_bundles_have_fold1_complete_finite_coverage": True,
            "all_four_full_to_full_identities_at_most_1e-5": True,
            "frozen_baseline_recompute_within_path_local_bf16_bound": True,
            "position_preserving_content_neutral_rows_and_path_audits_bound": True,
            "parent_selected_branch_and_contract_sha_bound": True,
            "candidate_and_request_manifests_reconstructed": True,
            "extension_plan_and_manifest_hashes_bound": True,
        },
        "extension_manifest_sha256": extension_manifest["_sha256"],
        "inputs": input_identities,
    }
    pre_qrels_path = output_dir / "pre_qrels_audit.json"
    _write_json(pre_qrels_path, pre_qrels)

    qrels_path, qrels_manifest = audit_fold_qrels(
        standardized_dir, qrels_split_dir, 1
    )
    gains = _load_fold_qrels(qrels_path, candidates)
    strict = _strict_transfer_mask(records, candidates, gains)
    request_ids = [record.request_id for record in records]
    clusters = np.asarray(
        [normalize_query(record.query) for record in records], dtype=np.str_
    )
    family_rows = []
    results = {}
    per_request = {}
    for method_id in METHODS:
        results[method_id] = {}
        if method_id in stopped:
            for node in NECESSITY_NODES:
                results[method_id][node] = {}
                for donor_mode in DONOR_MODES:
                    results[method_id][node][donor_mode] = {}
                    for endpoint in ENDPOINTS:
                        inference = _missing_inference(
                            donor_mode=donor_mode,
                            reason=stopped[method_id]["evidence_role"],
                        )
                        results[method_id][node][donor_mode][endpoint] = inference
                        family_rows.append(
                            _family_row(
                                method_id, node, donor_mode, endpoint, inference
                            )
                        )
            continue
        bundle = bundles[method_id]
        endpoints = {
            condition: {
                "target_margin": _target_margins(
                    request_ids, candidates, gains, bundle.scores[condition]
                ),
                "ndcg@10": _ndcg(
                    request_ids, candidates, gains, bundle.scores[condition]
                ),
            }
            for condition in component_necessity_conditions()
        }
        for node in NECESSITY_NODES:
            results[method_id][node] = {}
            for donor_mode in DONOR_MODES:
                results[method_id][node][donor_mode] = {}
                removal = f"{node}.{donor_mode}_to_full_removal"
                eligible = np.asarray(
                    [bundle.content_neutral_eligible[request_id] for request_id in request_ids],
                    dtype=bool,
                )
                per_request[
                    f"{method_id}__content_neutral_eligible_mask"
                ] = eligible
                for endpoint in ENDPOINTS:
                    values = endpoints[removal][endpoint] - endpoints["baseline_full"][
                        endpoint
                    ]
                    mask = strict & np.isfinite(values)
                    if donor_mode == "neutral":
                        mask &= eligible
                    inference = {
                        **cluster_mean_inference(values[mask], clusters[mask]),
                        "status": "completed",
                        "donor_mode": donor_mode,
                        "donor_role": (
                            "primary_position_preserving_content_removal"
                            if donor_mode == "neutral"
                            else "position_confounded_sensitivity"
                        ),
                        "contrast": (
                            f"{donor_mode}_to_full_removal_minus_baseline_full"
                        ),
                        "eligible_surface": (
                            "strict_transfer_and_frozen_content_neutral_eligible"
                            if donor_mode == "neutral"
                            else "strict_transfer"
                        ),
                        "expected_harm_removal_sign": "positive",
                    }
                    results[method_id][node][donor_mode][endpoint] = inference
                    family_rows.append(
                        _family_row(
                            method_id, node, donor_mode, endpoint, inference
                        )
                    )
                    per_request[
                        f"{method_id}__{node}__{donor_mode}__{endpoint}"
                    ] = values
    _apply_bh_and_gates(family_rows, results)
    if len(family_rows) != 2 * FAMILY_SIZE_PER_ENDPOINT:
        raise AssertionError("component-necessity family size drift")
    per_request_path = output_dir / "per_request_contrasts.npz"
    np.savez(
        per_request_path,
        **per_request,
        request_ids=np.asarray(request_ids, dtype=np.str_),
        normalized_queries=clusters,
        strict_mask=strict,
    )
    metrics = {
        "schema_version": 1,
        "analysis_type": "transformer_component_necessity_extension",
        "analysis_run_id": analysis_run_id,
        "methods": list(METHODS),
        "nodes": list(NECESSITY_NODES),
        "donor_modes": list(DONOR_MODES),
        "endpoints": list(ENDPOINTS),
        "normalized_query_fold": 1,
        "strict_transfer_requests": int(strict.sum()),
        "bootstrap": {
            "cluster": "normalized_query",
            "samples": 5000,
            "seed": 20260715,
        },
        "family_policy": {
            "separate_by_endpoint": True,
            "units_per_endpoint": FAMILY_SIZE_PER_ENDPOINT,
            "method": "benjamini_hochberg",
            "missing_or_gate_stopped_p": 1.0,
        },
        "family_rows": family_rows,
        "results": results,
        "claim_boundary": {
            "highest_authorized_claim": (
                "component_state_is_a_necessary_mediator_conditional_on_full_"
                "recipient_context"
            ),
            "operator_necessity_authorized": False,
            "exclusive_origin_authorized": False,
            "cross_dataset_or_model_scale_generalization_authorized": False,
            "primary_support_requires_position_preserving_neutral_removal": True,
            "null_removal_alone_authorizes_support": False,
            "design_ranking_requires_parent_sufficiency_and_specificity": True,
        },
        "input_identities": input_identities,
        "extension_manifest_sha256": extension_manifest["_sha256"],
        "pre_qrels_audit_path": str(pre_qrels_path),
        "pre_qrels_audit_sha256": sha256_file(pre_qrels_path),
        "qrels_read": True,
        "qrels_fold_opened": 1,
        "other_fold_qrels_opened": False,
        "qrels_fold_sha256": sha256_file(qrels_path),
        "qrels_split_manifest_sha256": sha256_file(
            Path(qrels_split_dir) / "manifest.json"
        ),
        "qrels_source_sha256": qrels_manifest["source_qrels_sha256"],
        "per_request_contrasts_path": str(per_request_path),
        "per_request_contrasts_sha256": sha256_file(per_request_path),
        "command": list(command or []),
        "source_test_opened": False,
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
            "split": "dev_fold1",
            "qrels_sha256": metrics["qrels_fold_sha256"],
            "metrics_path": str(metrics_path),
            "metrics_sha256": sha256_file(metrics_path),
        },
    )
    return metrics


def _audit_bundle(
    root: str | Path,
    records: Sequence[Any],
    *,
    all_records: Sequence[Any],
    method_id: str,
    extension_manifest: Mapping[str, Any],
) -> NecessityBundle:
    root = Path(root)
    metadata = _read_json(root / "metadata.json")
    expected = {
        "analysis_stage": "transformer_component_necessity_extension",
        "status": "completed",
        "result_eligible": True,
        "complete_finite_score_coverage": True,
        "identity_passed": True,
        "method_id": method_id,
        "normalized_query_fold": 1,
        "selected_nodes": list(NECESSITY_NODES),
        "score_conditions": list(component_necessity_conditions()),
        "deep_dive_manifest_sha256": extension_manifest["parent_deep_dive"][
            "manifest_sha256"
        ],
        "extension_manifest_sha256": extension_manifest["_sha256"],
        "operator_necessity_tested": False,
        "exclusive_origin_tested": False,
        "qrels_read": False,
        "source_test_opened": False,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"component-necessity bundle mismatch: {key}")
    if float(metadata.get("maximum_identity_delta", math.inf)) > 1.0e-5:
        raise ValueError("component-necessity identity bound failed")
    if float(metadata.get("maximum_baseline_low_precision_ratio", math.inf)) > 1.0:
        raise ValueError("component-necessity baseline BF16 bound failed")
    if method_id == "q3_tallrec_generalqwen" and float(
        metadata.get("shared_prompt_path_max_abs_delta", math.inf)
    ) != 0.0:
        raise ValueError("component-necessity Q3 shared prompt identity failed")
    frozen = extension_manifest["frozen_inputs"]["models"][method_id]
    if (
        metadata.get("checkpoint_id") != frozen["checkpoint_id"]
        or metadata.get("config_sha256") != frozen["config_sha256"]
    ):
        raise ValueError("component-necessity model binding drift")
    digest = str(metadata.get("implementation_identity", {}).get("digest") or "")
    if not digest or metadata.get("run_contract", {}).get(
        "implementation_digest"
    ) != digest:
        raise ValueError("component-necessity implementation digest drift")
    branch = metadata.get("branch_contract", {})
    branch_path = Path(str(branch.get("path") or ""))
    if not branch_path.is_file() or sha256_file(branch_path) != branch.get("sha256"):
        raise ValueError("component-necessity branch contract bytes changed")
    contract = _read_json(branch_path)
    if (
        contract.get("selected_block") != metadata.get("selected_block")
        or contract.get("method_id") != method_id
        or contract.get("fold1_negative_transition_reproduced") is not True
        or contract.get("evidence_role")
        != "registered_confirmatory_branch_localization"
        or contract.get("selected_nodes") != list(SELECTED_NODES)
    ):
        raise ValueError("component-necessity branch contract content drift")
    parent = metadata.get("parent_selected_branch", {})
    parent_root = Path(str(parent.get("path") or ""))
    if (
        not (parent_root / "metadata.json").is_file()
        or sha256_file(parent_root / "metadata.json") != parent.get("metadata_sha256")
        or not (parent_root / "scores.jsonl").is_file()
        or sha256_file(parent_root / "scores.jsonl") != parent.get("scores_sha256")
    ):
        raise ValueError("component-necessity parent selected-branch bytes changed")
    scores_path = root / "scores.jsonl"
    if metadata.get("scores_sha256") != sha256_file(scores_path):
        raise ValueError("component-necessity score bytes changed")
    frozen_full = _load_bundle_frozen_baseline(
        metadata,
        all_records,
        label="component-necessity full",
        identity_key="frozen_full_baseline",
    )
    frozen_null = _load_bundle_frozen_baseline(
        metadata,
        all_records,
        label="component-necessity null",
        identity_key="frozen_null_baseline",
    )
    frozen_control_eligibility = _load_bundle_content_control_eligibility(
        metadata,
        all_records,
        label="component-necessity",
        identity_key="content_neutral_control",
    )
    observed = audit_scalar_partial(
        scores_path, records, component_necessity_conditions()
    )
    if observed["completed_requests"] != len(records) or observed[
        "completed_score_rows"
    ] != sum(len(record.candidates) for record in records):
        raise ValueError("component-necessity request/candidate coverage drift")
    scores = {name: {} for name in component_necessity_conditions()}
    content_neutral_eligible = {}
    maximum_full_delta = 0.0
    maximum_null_delta = 0.0
    maximum_baseline_ratio = 0.0
    for block_row in iter_jsonl(scores_path):
        request_id = str(block_row["request_id"])
        raw_eligibilities = [
            row.get("content_neutral_eligible") for row in block_row["rows"]
        ]
        if any(not isinstance(value, bool) for value in raw_eligibilities):
            raise ValueError(
                "component-necessity neutral eligibility is not boolean"
            )
        row_eligibilities = set(raw_eligibilities)
        if len(row_eligibilities) != 1:
            raise ValueError("component-necessity neutral eligibility differs within request")
        eligible = row_eligibilities.pop()
        content_neutral_eligible[request_id] = eligible
        for row in block_row["rows"]:
            full_delta, null_delta, baseline_ratio = _baseline_recompute_deltas(
                row,
                frozen_full,
                frozen_null,
            )
            maximum_full_delta = max(maximum_full_delta, full_delta)
            maximum_null_delta = max(maximum_null_delta, null_delta)
            maximum_baseline_ratio = max(
                maximum_baseline_ratio,
                baseline_ratio,
            )
        for condition in component_necessity_conditions():
            scores[condition][request_id] = {
                str(row["candidate_item_id"]): float(row["conditions"][condition])
                for row in block_row["rows"]
            }
        if not eligible:
            for node in NECESSITY_NODES:
                neutral = scores[f"{node}.neutral_to_full_removal"][request_id]
                identity = scores[f"{node}.full_to_full_identity"][request_id]
                if neutral != identity:
                    raise ValueError(
                        "ineligible component-necessity neutral score is not identity"
                    )
    for key, observed_value in (
        ("maximum_full_baseline_delta", maximum_full_delta),
        ("maximum_null_baseline_delta", maximum_null_delta),
        ("maximum_baseline_low_precision_ratio", maximum_baseline_ratio),
    ):
        if float(metadata.get(key, math.inf)) != observed_value:
            raise ValueError(
                f"component-necessity baseline audit differs from metadata: {key}"
            )
    if maximum_baseline_ratio > 1.0:
        raise ValueError("component-necessity recomputed baseline BF16 bound failed")
    observed_eligible = sum(content_neutral_eligible.values())
    if observed_eligible != int(
        metadata.get("fold1_content_neutral_eligible_requests", -1)
    ):
        raise ValueError("component-necessity fold1 neutral eligibility count drift")
    expected_fold1_eligibility = {
        record.request_id: frozen_control_eligibility[record.request_id]
        for record in records
    }
    if content_neutral_eligible != expected_fold1_eligibility:
        raise ValueError(
            "component-necessity score eligibility differs from frozen controls"
        )
    content_control = metadata.get("content_neutral_control", {})
    frozen_content = extension_manifest["frozen_inputs"]["content_neutral"]
    expected_rows_sha = frozen_content[
        "q2_rows_sha256" if method_id.startswith("q2_") else "q3_rows_sha256"
    ]
    if (
        content_control.get("manifest_sha256")
        != frozen_content["manifest_sha256"]
        or content_control.get("rows_sha256") != expected_rows_sha
    ):
        raise ValueError("component-necessity content-neutral metadata drift")
    return NecessityBundle(root, metadata, scores, content_neutral_eligible)


def _baseline_recompute_deltas(
    row: Mapping[str, Any],
    frozen_full: Mapping[tuple[str, str], float],
    frozen_null: Mapping[tuple[str, str], float],
) -> tuple[float, float, float]:
    request_id = str(row.get("request_id"))
    item_id = str(row.get("candidate_item_id"))
    key = (request_id, item_id)
    if key not in frozen_full or key not in frozen_null:
        raise ValueError(
            "component-necessity candidate is absent from a frozen baseline"
        )
    values = row.get("conditions")
    if not isinstance(values, Mapping):
        raise ValueError("component-necessity candidate conditions are invalid")
    observed_full = float(values["baseline_full"])
    observed_null = float(values["baseline_null"])
    expected_full = float(frozen_full[key])
    expected_null = float(frozen_null[key])
    full_delta = abs(observed_full - expected_full)
    null_delta = abs(observed_null - expected_null)
    full_bound = 8.0 * (2.0**-7) * max(1.0, abs(expected_full))
    null_bound = 8.0 * (2.0**-7) * max(1.0, abs(expected_null))
    return (
        full_delta,
        null_delta,
        max(full_delta / full_bound, null_delta / null_bound),
    )


def _audit_gate_contract(
    path: str | Path, *, method_id: str, expected_checkpoint_id: str
) -> dict[str, Any]:
    contract = _read_json(Path(path))
    if (
        contract.get("contract_type")
        != "transformer_deep_dive_d2_selected_branch_contract"
        or contract.get("status") != "completed"
        or contract.get("method_id") != method_id
        or contract.get("checkpoint_id") != expected_checkpoint_id
        or contract.get("selected_nodes") != list(SELECTED_NODES)
        or contract.get("qrels_values_exposed_to_scorer") is not False
        or contract.get("source_test_opened") is not False
    ):
        raise ValueError("component-necessity gate contract is inadmissible")
    if (
        contract.get("branch_scoring_eligible") is True
        and contract.get("fold1_negative_transition_reproduced") is True
        and contract.get("evidence_role")
        == "registered_confirmatory_branch_localization"
    ):
        raise ValueError("confirmed component-necessity model requires a bundle")
    return contract


def _family_row(
    method_id: str,
    node: str,
    donor_mode: str,
    endpoint: str,
    inference: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "method_id": method_id,
        "node": node,
        "donor_mode": donor_mode,
        "endpoint": endpoint,
        "status": inference["status"],
        "mean": inference.get("mean"),
        "ci95": inference.get("ci95"),
        "two_sided_p": float(inference["two_sided_p"]),
    }


def _missing_inference(*, donor_mode: str, reason: str) -> dict[str, Any]:
    return {
        "status": "gate_stopped",
        "reason": reason,
        "donor_mode": donor_mode,
        "donor_role": (
            "primary_position_preserving_content_removal"
            if donor_mode == "neutral"
            else "position_confounded_sensitivity"
        ),
        "contrast": f"{donor_mode}_to_full_removal_minus_baseline_full",
        "expected_harm_removal_sign": "positive",
        "n": 0,
        "clusters": 0,
        "mean": None,
        "ci95": None,
        "two_sided_p": 1.0,
    }


def _apply_bh_and_gates(
    family_rows: list[dict[str, Any]],
    results: Mapping[
        str, Mapping[str, Mapping[str, Mapping[str, dict[str, Any]]]]
    ],
) -> None:
    for endpoint in ENDPOINTS:
        rows = [row for row in family_rows if row["endpoint"] == endpoint]
        if len(rows) != FAMILY_SIZE_PER_ENDPOINT:
            raise AssertionError("component-necessity endpoint family size drift")
        q_values = benjamini_hochberg([row["two_sided_p"] for row in rows])
        for row, q_value in zip(rows, q_values):
            row["bh_q"] = float(q_value)
            result = results[row["method_id"]][row["node"]][row["donor_mode"]][
                endpoint
            ]
            result["bh_q"] = float(q_value)
            if result["status"] != "completed":
                result["positive_removal_gate_passed"] = False
                result["primary_position_preserving_gate_passed"] = False
                result["ndcg_practically_equivalent"] = False
                row["positive_removal_gate_passed"] = False
                row["primary_position_preserving_gate_passed"] = False
                row["ndcg_practically_equivalent"] = False
                continue
            mean = float(result["mean"])
            ci = [float(value) for value in result["ci95"]]
            gate = mean > 0.0 and ci[0] > 0.0 and float(q_value) < 0.05
            result["positive_removal_gate_passed"] = gate
            result["primary_position_preserving_gate_passed"] = (
                row["donor_mode"] == "neutral" and gate
            )
            row["positive_removal_gate_passed"] = gate
            row["primary_position_preserving_gate_passed"] = (
                row["donor_mode"] == "neutral" and gate
            )
            equivalent = endpoint == "ndcg@10" and (
                ci[0] >= NDCG_EQUIVALENCE_BAND[0]
                and ci[1] <= NDCG_EQUIVALENCE_BAND[1]
            )
            result["ndcg_practically_equivalent"] = equivalent
            row["ndcg_practically_equivalent"] = equivalent


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value
