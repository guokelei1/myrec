"""Fail-closed synthesis of component sufficiency, specificity, and necessity.

This module does not open qrels or score bundles.  It combines only completed
registered evaluator outputs and verifies that the forward and reverse
interventions share the same selected-branch parent bytes.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from myrec.mechanism.attention_edge_evaluator import _write_json
from myrec.mechanism.component_necessity_evaluator import (
    DONOR_MODES,
    ENDPOINTS,
    FAMILY_SIZE_PER_ENDPOINT,
    METHODS,
    NDCG_EQUIVALENCE_BAND,
)
from myrec.mechanism.component_necessity_runtime import (
    EXTENSION_MANIFEST_PATH,
    _load_extension_manifest,
)
from myrec.mechanism.component_necessity_scoring import NECESSITY_NODES
from myrec.mechanism.deep_dive_native_evaluator import benjamini_hochberg
from myrec.mechanism.selected_branch_evaluator import (
    CONTRAST_GROUPS,
    SELECTED_BRANCH_FOLD_SCOPE,
    selected_branch_contrast_specs,
)
from myrec.utils.hashing import sha256_file, sha256_text


PRIMARY_ENDPOINT = "target_margin"


def synthesize_component_design_gates(
    necessity_metrics_path: str | Path,
    selected_synthesis_path: str | Path,
    output_dir: str | Path,
    analysis_run_id: str,
    *,
    extension_manifest_path: str | Path = EXTENSION_MANIFEST_PATH,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build design-facing gates without turning a selected layer into a method."""

    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"component-design synthesis output is not empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    necessity_path = Path(necessity_metrics_path)
    selected_path = Path(selected_synthesis_path)
    extension_manifest = _load_extension_manifest(extension_manifest_path)
    necessity = _audit_necessity_metrics(
        necessity_path, extension_manifest_sha256=extension_manifest["_sha256"]
    )
    selected = _audit_selected_synthesis(selected_path)
    lineage = _audit_shared_parent_lineage(necessity, selected)

    selected_rows = {
        (str(row["method_id"]), str(row["contrast_id"]), str(row["endpoint"])): row
        for row in selected["rows"]
    }
    rows: list[dict[str, Any]] = []
    for method_id in METHODS:
        for node in NECESSITY_NODES:
            endpoint_gates: dict[str, Any] = {}
            for endpoint in ENDPOINTS:
                neutral = necessity["results"][method_id][node]["neutral"][endpoint]
                null = necessity["results"][method_id][node]["null"][endpoint]
                parent_same = selected_rows[
                    (method_id, f"same__{node}", endpoint)
                ]
                parent_specificity = selected_rows[
                    (method_id, f"same_minus_wrong__{node}", endpoint)
                ]
                parent_cross = selected_rows[
                    (method_id, f"same_minus_cross__{node}", endpoint)
                ]
                parent_direction_controls = [
                    selected_rows[(method_id, f"{short}__{node}", endpoint)]
                    for short in ("norm", "direction", "random")
                ]
                neutral_gate = (
                    neutral.get("primary_position_preserving_gate_passed") is True
                )
                same_gate = parent_same.get("registered_support") is True
                specificity_gate = (
                    parent_specificity.get("registered_support") is True
                )
                cross_gate = parent_cross.get("registered_support") is True
                direction_scale_gate = all(
                    row.get("registered_support") is True
                    for row in parent_direction_controls
                )
                registered_state_gate = bool(
                    neutral_gate and same_gate and specificity_gate
                )
                design_target_eligible = node != "block_output_residual"
                endpoint_gates[endpoint] = {
                    "position_preserving_removal_gate_passed": neutral_gate,
                    "parent_same_request_sufficiency_gate_passed": same_gate,
                    "parent_history_specificity_gate_passed": specificity_gate,
                    "parent_cross_request_stress_gate_passed": cross_gate,
                    "parent_direction_scale_controls_passed": direction_scale_gate,
                    "registered_component_state_gate_passed": registered_state_gate,
                    "functional_node_design_target_eligible": design_target_eligible,
                    "robust_design_prioritization_gate_passed": bool(
                        design_target_eligible
                        and registered_state_gate
                        and cross_gate
                        and direction_scale_gate
                    ),
                    "null_position_confounded_sensitivity_gate_passed": (
                        null.get("positive_removal_gate_passed") is True
                    ),
                    "necessity_status": neutral.get("status"),
                    "parent_same_missing": parent_same.get("missing") is True,
                    "parent_specificity_missing": (
                        parent_specificity.get("missing") is True
                    ),
                    "parent_structural_control_missing_count": sum(
                        row.get("missing") is True
                        for row in (parent_cross, *parent_direction_controls)
                    ),
                    "ndcg_practically_equivalent": (
                        neutral.get("ndcg_practically_equivalent") is True
                        if endpoint == "ndcg@10"
                        else None
                    ),
                }
            rows.append(
                {
                    "method_id": method_id,
                    "node": node,
                    "endpoint_gates": endpoint_gates,
                    "primary_target_margin_component_state_gate_passed": endpoint_gates[
                        PRIMARY_ENDPOINT
                    ]["registered_component_state_gate_passed"],
                    "primary_target_margin_design_gate_passed": endpoint_gates[
                        PRIMARY_ENDPOINT
                    ]["robust_design_prioritization_gate_passed"],
                    "claim_role": _claim_role(node),
                }
            )

    model_summaries = {
        method_id: _summarize_model(method_id, rows) for method_id in METHODS
    }
    cross_model_state_nodes = [
        node
        for node in NECESSITY_NODES
        if all(
            _row_for(rows, method_id, node)[
                "primary_target_margin_component_state_gate_passed"
            ]
            for method_id in METHODS
        )
    ]
    cross_model_design_nodes = [
        node
        for node in NECESSITY_NODES
        if all(
            _row_for(rows, method_id, node)[
                "primary_target_margin_design_gate_passed"
            ]
            for method_id in METHODS
        )
    ]
    result = {
        "schema_version": 1,
        "analysis_type": "transformer_component_design_gate_synthesis",
        "analysis_run_id": analysis_run_id,
        "status": "completed",
        "models": list(METHODS),
        "nodes": list(NECESSITY_NODES),
        "primary_endpoint": PRIMARY_ENDPOINT,
        "input_identities": {
            "component_necessity_metrics": {
                "path": str(necessity_path),
                "sha256": sha256_file(necessity_path),
            },
            "selected_branch_synthesis": {
                "path": str(selected_path),
                "sha256": sha256_file(selected_path),
            },
            "component_necessity_extension_manifest": {
                "path": str(extension_manifest_path),
                "sha256": extension_manifest["_sha256"],
            },
        },
        "shared_parent_lineage": lineage,
        "gate_definition": {
            "all_required": [
                "neutral_position_preserving_removal_positive_ci_and_bh",
                "parent_same_request_full_to_null_sufficiency_expected_sign_and_bh",
                "parent_same_minus_wrong_history_specificity_expected_sign_and_bh",
            ],
            "additional_design_prioritization_controls": [
                "parent_same_minus_cross_request_stress_expected_sign_and_bh",
                "parent_same_minus_norm_matched_control_expected_sign_and_bh",
                "parent_same_minus_direction_control_expected_sign_and_bh",
                "parent_same_minus_random_direction_control_expected_sign_and_bh",
            ],
            "registered_state_support_and_design_priority_are_distinct": True,
            "block_output_residual_is_state_ceiling_not_design_target": True,
            "null_donor_is_sensitivity_only": True,
            "primary_endpoint": PRIMARY_ENDPOINT,
            "fold_scope": dict(SELECTED_BRANCH_FOLD_SCOPE),
        },
        "rows": rows,
        "model_summaries": model_summaries,
        "cross_model_functional_support": {
            "component_state_supported_nodes": cross_model_state_nodes,
            "design_prioritized_nodes": cross_model_design_nodes,
            "any_shared_component_state_node": bool(cross_model_state_nodes),
            "any_shared_design_prioritized_node": bool(cross_model_design_nodes),
            "component_path_design_ranking_eligible": bool(
                cross_model_design_nodes
            ),
        },
        "claim_boundary": {
            "highest_authorized_claim": (
                "component_state_is_a_history_specific_necessary_mediator_and_"
                "sufficient_carrier_at_a_split_sample_localized_transition"
            ),
            "exact_layer_index_is_architecture_evidence": False,
            "selected_layer_may_be_reused_as_method_hyperparameter": False,
            "operator_necessity_authorized": False,
            "exclusive_origin_authorized": False,
            "direct_history_token_flow_authorized": False,
            "cross_dataset_or_model_scale_generalization_authorized": False,
            "single_model_support_may_change_global_architecture_ranking": False,
            "both_models_same_functional_node_required_for_global_ranking": True,
            "diagnostic_intervention_as_paper_method_authorized": False,
            "ranking_or_utility_improvement_from_gate_alone_authorized": False,
            "registered_behavior": "harmful_full_history_target_margin_response",
            "positive_neutral_removal_means_harm_reduction": True,
            "component_is_beneficial_for_transfer_authorized": False,
            "strengthen_or_preserve_component_authorized": False,
            "block_output_state_ceiling_authorizes_residual_operator_claim": False,
        },
        "upstream_scientific_effect_values_consumed": True,
        "qrels_read_by_this_synthesis": False,
        "score_bundles_read_by_this_synthesis": False,
        "source_test_opened": False,
        "command": list(command or []),
    }
    report_text = render_component_design_markdown(result)
    result["report"] = {
        "path": str(output_dir / "report.md"),
        "sha256": sha256_text(report_text),
    }
    _write_json(output_dir / "metrics.json", result)
    (output_dir / "report.md").write_text(report_text, encoding="utf-8")
    return result


def render_component_design_markdown(result: Mapping[str, Any]) -> str:
    """Render functional component gates without exposing the selected layer index."""

    lines = [
        "# Transformer Component Design-Gate Synthesis",
        "",
        "This supplement combines position-preserving reverse removal with the "
        "same-parent sufficiency and wrong-user specificity controls. Absolute layer "
        "indices are lineage metadata only and are intentionally omitted.",
        "",
        "| Model | Functional node | Neutral removal | Sufficiency | History specificity | Structural controls | State gate | Design priority | Null sensitivity |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["rows"]:
        gate = row["endpoint_gates"][PRIMARY_ENDPOINT]
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row["method_id"]),
                    str(row["claim_role"]),
                    _yes_no(gate["position_preserving_removal_gate_passed"]),
                    _yes_no(gate["parent_same_request_sufficiency_gate_passed"]),
                    _yes_no(gate["parent_history_specificity_gate_passed"]),
                    _yes_no(
                        gate["parent_cross_request_stress_gate_passed"]
                        and gate["parent_direction_scale_controls_passed"]
                    ),
                    _yes_no(gate["registered_component_state_gate_passed"]),
                    _yes_no(gate["robust_design_prioritization_gate_passed"]),
                    _yes_no(gate["null_position_confounded_sensitivity_gate_passed"]),
                )
            )
            + " |"
        )
    lines.extend(["", "## Functional interpretation", ""])
    for method_id in METHODS:
        summary = result["model_summaries"][method_id]
        lines.append(
            f"- `{method_id}`: " + ", ".join(summary["interpretations"])
        )
    cross = result["cross_model_functional_support"]
    lines.extend(
        [
            "",
            "## Cross-model design boundary",
            "",
            (
                "Shared history-specific component-state nodes: "
                + (
                    ", ".join(
                        f"`{node}`"
                        for node in cross["component_state_supported_nodes"]
                    )
                    if cross["component_state_supported_nodes"]
                    else "none"
                )
                + "."
            ),
            (
                "Shared design-prioritized nodes after all structural controls: "
                + (
                    ", ".join(
                        f"`{node}`" for node in cross["design_prioritized_nodes"]
                    )
                    if cross["design_prioritized_nodes"]
                    else "none"
                )
                + "."
            ),
            "",
            "A passing state gate does not establish operator necessity, exclusive "
            "origin, direct token flow, or transfer across model scale and datasets.",
            "It identifies mediation of the registered harmful full-history response; "
            "it does not show that the component benefits transfer or should be "
            "strengthened.",
            "",
        ]
    )
    return "\n".join(lines)


def _audit_necessity_metrics(
    path: Path, *, extension_manifest_sha256: str
) -> dict[str, Any]:
    metrics = _read_json(path)
    expected = {
        "analysis_type": "transformer_component_necessity_extension",
        "status": "completed",
        "methods": list(METHODS),
        "nodes": list(NECESSITY_NODES),
        "donor_modes": list(DONOR_MODES),
        "endpoints": list(ENDPOINTS),
        "normalized_query_fold": 1,
        "extension_manifest_sha256": extension_manifest_sha256,
        "qrels_read": True,
        "qrels_fold_opened": 1,
        "other_fold_qrels_opened": False,
        "source_test_opened": False,
    }
    for key, value in expected.items():
        if metrics.get(key) != value:
            raise ValueError(f"component-necessity metrics mismatch: {key}")
    if (
        metrics.get("bootstrap")
        != {"cluster": "normalized_query", "samples": 5000, "seed": 20260715}
        or isinstance(metrics.get("strict_transfer_requests"), bool)
        or not isinstance(metrics.get("strict_transfer_requests"), int)
        or metrics["strict_transfer_requests"] <= 0
    ):
        raise ValueError("component-necessity evaluator population differs")
    policy = metrics.get("family_policy", {})
    if (
        policy.get("separate_by_endpoint") is not True
        or policy.get("units_per_endpoint") != FAMILY_SIZE_PER_ENDPOINT
        or policy.get("method") != "benjamini_hochberg"
        or float(policy.get("missing_or_gate_stopped_p", -1.0)) != 1.0
    ):
        raise ValueError("component-necessity family policy drift")
    claim = metrics.get("claim_boundary", {})
    if (
        claim.get("primary_support_requires_position_preserving_neutral_removal")
        is not True
        or claim.get("null_removal_alone_authorizes_support") is not False
        or claim.get("design_ranking_requires_parent_sufficiency_and_specificity")
        is not True
        or claim.get("operator_necessity_authorized") is not False
        or claim.get("exclusive_origin_authorized") is not False
        or claim.get("cross_dataset_or_model_scale_generalization_authorized")
        is not False
    ):
        raise ValueError("component-necessity claim boundary drift")
    family_rows = metrics.get("family_rows")
    if not isinstance(family_rows, list) or len(family_rows) != 2 * FAMILY_SIZE_PER_ENDPOINT:
        raise ValueError("component-necessity family row coverage drift")
    keys = {
        (
            str(row.get("method_id")),
            str(row.get("node")),
            str(row.get("donor_mode")),
            str(row.get("endpoint")),
        )
        for row in family_rows
    }
    expected_keys = {
        (method_id, node, donor_mode, endpoint)
        for method_id in METHODS
        for node in NECESSITY_NODES
        for donor_mode in DONOR_MODES
        for endpoint in ENDPOINTS
    }
    if keys != expected_keys:
        raise ValueError("component-necessity family keys drift")
    results = metrics.get("results")
    if not isinstance(results, dict):
        raise ValueError("component-necessity results missing")
    by_key = {
        (
            str(row["method_id"]),
            str(row["node"]),
            str(row["donor_mode"]),
            str(row["endpoint"]),
        ): row
        for row in family_rows
    }
    for endpoint in ENDPOINTS:
        endpoint_rows = [
            row for row in family_rows if row.get("endpoint") == endpoint
        ]
        if len(endpoint_rows) != FAMILY_SIZE_PER_ENDPOINT:
            raise ValueError("component-necessity endpoint family size drift")
        observed_p = [
            _finite_probability(row.get("two_sided_p"), "two_sided_p")
            for row in endpoint_rows
        ]
        expected_q = benjamini_hochberg(observed_p)
        observed_q = [
            _finite_probability(row.get("bh_q"), "bh_q")
            for row in endpoint_rows
        ]
        if observed_q != [float(value) for value in expected_q]:
            raise ValueError("component-necessity BH values differ")
    for key in expected_keys:
        method_id, node, donor_mode, endpoint = key
        try:
            result = results[method_id][node][donor_mode][endpoint]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"component-necessity result missing: {key}") from exc
        row = by_key[key]
        if (
            result.get("status") != row.get("status")
            or _inference_signature(result) != _inference_signature(row)
        ):
            raise ValueError(f"component-necessity nested/family inference differs: {key}")
        for gate_name in (
            "positive_removal_gate_passed",
            "primary_position_preserving_gate_passed",
            "ndcg_practically_equivalent",
        ):
            if result.get(gate_name) is not row.get(gate_name):
                raise ValueError(
                    f"component-necessity nested/family gate differs: {key} {gate_name}"
                )
        if result.get("primary_position_preserving_gate_passed") is True:
            mean = result.get("mean")
            ci95 = result.get("ci95")
            q_value = result.get("bh_q")
            if (
                donor_mode != "neutral"
                or result.get("status") != "completed"
                or result.get("positive_removal_gate_passed") is not True
                or not isinstance(mean, (int, float))
                or isinstance(mean, bool)
                or not math.isfinite(float(mean))
                or float(mean) <= 0.0
                or not isinstance(ci95, list)
                or len(ci95) != 2
                or not all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    for value in ci95
                )
                or float(ci95[0]) <= 0.0
                or not isinstance(q_value, (int, float))
                or isinstance(q_value, bool)
                or not math.isfinite(float(q_value))
                or not 0.0 <= float(q_value) < 0.05
            ):
                raise ValueError(
                    "component-necessity positive gate is not effect-derived: "
                    f"{key}"
                )
        if donor_mode == "null" and result.get(
            "primary_position_preserving_gate_passed"
        ) is not False:
            raise ValueError("null donor cannot pass the primary position-preserving gate")
        _audit_necessity_gate_derivation(
            result,
            donor_mode=donor_mode,
            endpoint=endpoint,
            key=key,
        )
    input_identities = metrics.get("input_identities", {})
    if set(input_identities) != set(METHODS):
        raise ValueError("component-necessity input identity coverage drift")
    pre_qrels_path = Path(str(metrics.get("pre_qrels_audit_path") or ""))
    _require_file_sha(
        pre_qrels_path,
        metrics.get("pre_qrels_audit_sha256"),
        "component-necessity pre-qrels audit",
    )
    pre_qrels = _read_json(pre_qrels_path)
    expected_checks = {
        "each_model_has_exactly_one_completed_bundle_or_gate_stop": True,
        "completed_bundles_have_fold1_complete_finite_coverage": True,
        "all_four_full_to_full_identities_at_most_1e-5": True,
        "frozen_baseline_recompute_within_path_local_bf16_bound": True,
        "position_preserving_content_neutral_rows_and_path_audits_bound": True,
        "parent_selected_branch_and_contract_sha_bound": True,
        "candidate_and_request_manifests_reconstructed": True,
        "extension_plan_and_manifest_hashes_bound": True,
    }
    if (
        pre_qrels.get("analysis_type")
        != "component_necessity_pre_qrels_integrity"
        or pre_qrels.get("status") != "passed"
        or pre_qrels.get("qrels_read") is not False
        or pre_qrels.get("checks") != expected_checks
        or pre_qrels.get("extension_manifest_sha256")
        != extension_manifest_sha256
        or pre_qrels.get("inputs") != input_identities
    ):
        raise ValueError("component-necessity pre-qrels audit differs")
    _require_file_sha(
        Path(str(metrics.get("per_request_contrasts_path") or "")),
        metrics.get("per_request_contrasts_sha256"),
        "component-necessity per-request contrasts",
    )
    for field in (
        "qrels_fold_sha256",
        "qrels_split_manifest_sha256",
        "qrels_source_sha256",
    ):
        _require_sha256(metrics.get(field), f"component-necessity {field}")
    return metrics


def _inference_signature(value: Mapping[str, Any]) -> dict[str, Any]:
    status = value.get("status")
    if status == "gate_stopped":
        if (
            value.get("mean") is not None
            or value.get("ci95") is not None
            or float(value.get("two_sided_p", -1.0)) != 1.0
            or float(value.get("bh_q", -1.0)) != 1.0
        ):
            raise ValueError("component-necessity gate-stopped inference differs")
        return {
            "mean": None,
            "ci95": None,
            "two_sided_p": 1.0,
            "bh_q": 1.0,
        }
    if status != "completed":
        raise ValueError("component-necessity inference status differs")
    interval = value.get("ci95")
    if not isinstance(interval, list) or len(interval) != 2:
        raise ValueError("component-necessity confidence interval differs")
    lower = _finite_number(interval[0], "ci95 lower")
    upper = _finite_number(interval[1], "ci95 upper")
    if lower > upper:
        raise ValueError("component-necessity confidence interval is reversed")
    return {
        "mean": _finite_number(value.get("mean"), "mean"),
        "ci95": [lower, upper],
        "two_sided_p": _finite_probability(
            value.get("two_sided_p"), "two_sided_p"
        ),
        "bh_q": _finite_probability(value.get("bh_q"), "bh_q"),
    }


def _audit_necessity_gate_derivation(
    result: Mapping[str, Any],
    *,
    donor_mode: str,
    endpoint: str,
    key: tuple[str, str, str, str],
) -> None:
    if result.get("status") == "gate_stopped":
        expected_positive = False
        expected_primary = False
        expected_equivalent = False
    else:
        signature = _inference_signature(result)
        expected_positive = bool(
            signature["mean"] > 0.0
            and signature["ci95"][0] > 0.0
            and signature["bh_q"] < 0.05
        )
        expected_primary = donor_mode == "neutral" and expected_positive
        expected_equivalent = bool(
            endpoint == "ndcg@10"
            and signature["ci95"][0] >= NDCG_EQUIVALENCE_BAND[0]
            and signature["ci95"][1] <= NDCG_EQUIVALENCE_BAND[1]
        )
    if (
        result.get("positive_removal_gate_passed") is not expected_positive
        or result.get("primary_position_preserving_gate_passed")
        is not expected_primary
        or result.get("ndcg_practically_equivalent") is not expected_equivalent
    ):
        raise ValueError(
            f"component-necessity gate is not effect-derived: {key}"
        )


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"component-necessity {label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"component-necessity {label} is non-finite")
    return result


def _finite_probability(value: Any, label: str) -> float:
    result = _finite_number(value, label)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"component-necessity {label} is outside [0,1]")
    return result


def _require_sha256(value: Any, label: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{label} is not SHA-256")
    return text


def _require_file_sha(path: Path, expected: Any, label: str) -> None:
    expected_sha = _require_sha256(expected, label)
    if not path.is_file() or sha256_file(path) != expected_sha:
        raise ValueError(f"{label} bytes differ")


def _audit_selected_synthesis(path: Path) -> dict[str, Any]:
    metrics = _read_json(path)
    if (
        metrics.get("analysis_type")
        != "transformer_deep_dive_d2_selected_branch_synthesis"
        or metrics.get("status") != "completed"
        or metrics.get("models") != list(METHODS)
        or metrics.get("fold_scope") != SELECTED_BRANCH_FOLD_SCOPE
    ):
        raise ValueError("selected-branch synthesis contract drift")
    specs = selected_branch_contrast_specs()
    expected_keys = {
        (method_id, contrast_id, endpoint)
        for method_id in METHODS
        for contrast_id in specs
        for endpoint in ENDPOINTS
    }
    rows = metrics.get("rows")
    if not isinstance(rows, list) or len(rows) != len(expected_keys):
        raise ValueError("selected-branch synthesis row coverage drift")
    observed_keys = {
        (
            str(row.get("method_id")),
            str(row.get("contrast_id")),
            str(row.get("endpoint")),
        )
        for row in rows
    }
    if observed_keys != expected_keys:
        raise ValueError("selected-branch synthesis row keys drift")
    by_key = {
        (
            str(row.get("method_id")),
            str(row.get("contrast_id")),
            str(row.get("endpoint")),
        ): row
        for row in rows
    }
    for (method_id, contrast_id, endpoint), row in by_key.items():
        expected_group = specs[contrast_id]["group"]
        if row.get("group") != expected_group or type(row.get("missing")) is not bool:
            raise ValueError(
                "selected-branch synthesis row group/missing scope differs: "
                f"{method_id}:{contrast_id}:{endpoint}"
            )
        if type(row.get("registered_support")) is not bool:
            raise ValueError("selected-branch registered-support flag is not boolean")
        if row["registered_support"]:
            mean = row.get("mean")
            q_value = row.get("bh_q")
            if (
                row["missing"]
                or row.get("expected_sign") != "negative"
                or row.get("expected_sign_met") is not True
                or row.get("bh_significant") is not True
                or row.get("evidence_role")
                != "registered_confirmatory_branch_localization"
                or not isinstance(mean, (int, float))
                or isinstance(mean, bool)
                or not math.isfinite(float(mean))
                or float(mean) >= 0.0
                or not isinstance(q_value, (int, float))
                or isinstance(q_value, bool)
                or not math.isfinite(float(q_value))
                or not 0.0 <= float(q_value) < 0.05
            ):
                raise ValueError(
                    "selected-branch support is not sign/BH/fold derived: "
                    f"{method_id}:{contrast_id}:{endpoint}"
                )
    expected_families = {
        f"{group}__{endpoint}": 2 * units
        for group, units in CONTRAST_GROUPS.items()
        for endpoint in ENDPOINTS
    }
    families = metrics.get("families", {})
    if set(families) != set(expected_families):
        raise ValueError("selected-branch synthesis family coverage drift")
    for family_id, planned_size in expected_families.items():
        if families[family_id].get("planned_family_size") != planned_size:
            raise ValueError(f"selected-branch planned family size drift: {family_id}")
    input_metrics = metrics.get("input_metrics", {})
    if set(input_metrics) - set(METHODS):
        raise ValueError("selected-branch synthesis input model drift")
    return metrics


def _audit_shared_parent_lineage(
    necessity: Mapping[str, Any], selected: Mapping[str, Any]
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    selected_inputs = selected["input_metrics"]
    for method_id in METHODS:
        necessity_identity = necessity["input_identities"][method_id]
        status = necessity_identity.get("status")
        if status == "gate_stopped":
            gate_path = Path(str(necessity_identity.get("path") or ""))
            if (
                not gate_path.is_file()
                or sha256_file(gate_path) != necessity_identity.get("sha256")
                or method_id in selected_inputs
            ):
                raise ValueError(f"gate-stopped lineage drift: {method_id}")
            output[method_id] = {
                "status": "gate_stopped",
                "shared_parent_bytes_verified": True,
            }
            continue
        if status != "completed_bundle" or method_id not in selected_inputs:
            raise ValueError(f"completed component lineage lacks parent metrics: {method_id}")

        necessity_root = Path(str(necessity_identity.get("path") or ""))
        necessity_metadata_path = necessity_root / "metadata.json"
        necessity_scores_path = necessity_root / "scores.jsonl"
        if (
            not necessity_metadata_path.is_file()
            or not necessity_scores_path.is_file()
            or sha256_file(necessity_metadata_path)
            != necessity_identity.get("metadata_sha256")
            or sha256_file(necessity_scores_path)
            != necessity_identity.get("scores_sha256")
        ):
            raise ValueError(f"component-necessity bundle bytes changed: {method_id}")
        necessity_metadata = _read_json(necessity_metadata_path)
        if (
            necessity_metadata.get("analysis_stage")
            != "transformer_component_necessity_extension"
            or necessity_metadata.get("method_id") != method_id
            or necessity_metadata.get("status") != "completed"
        ):
            raise ValueError(f"component-necessity metadata drift: {method_id}")

        selected_identity = selected_inputs[method_id]
        selected_metrics_path = Path(str(selected_identity.get("path") or ""))
        if (
            not selected_metrics_path.is_file()
            or sha256_file(selected_metrics_path) != selected_identity.get("sha256")
        ):
            raise ValueError(f"selected-branch metrics bytes changed: {method_id}")
        selected_metrics = _read_json(selected_metrics_path)
        if (
            selected_metrics.get("analysis_type")
            != "transformer_deep_dive_d2_selected_branch"
            or selected_metrics.get("status") != "completed"
            or selected_metrics.get("method_id") != method_id
            or selected_metrics.get("implementation_digest")
            != selected_identity.get("implementation_digest")
        ):
            raise ValueError(f"selected-branch metrics identity drift: {method_id}")
        parent_from_necessity = necessity_metadata.get("parent_selected_branch")
        parent_from_selected = selected_metrics.get("input_bundle")
        if parent_from_necessity != parent_from_selected:
            raise ValueError(
                f"necessity and sufficiency do not share parent bytes: {method_id}"
            )
        if necessity_metadata.get("selected_block") != selected_metrics.get(
            "selected_block"
        ):
            raise ValueError(f"necessity and sufficiency selected block differs: {method_id}")
        output[method_id] = {
            "status": "completed_shared_parent",
            "shared_parent_bytes_verified": True,
            "parent_selected_branch": parent_from_selected,
            "selected_block_recorded_for_lineage_only": selected_metrics.get(
                "selected_block"
            ),
            "exact_layer_index_is_architecture_evidence": False,
        }
    return output


def _summarize_model(
    method_id: str, rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    state_supported = [
        node
        for node in NECESSITY_NODES
        if _row_for(rows, method_id, node)[
            "primary_target_margin_component_state_gate_passed"
        ]
    ]
    design_prioritized = [
        node
        for node in NECESSITY_NODES
        if _row_for(rows, method_id, node)[
            "primary_target_margin_design_gate_passed"
        ]
    ]
    flags = set(state_supported)
    interpretations: list[str] = []
    if "block_input_residual" in flags:
        interpretations.append("harmful_state_arrives_from_upstream_or_is_distributed")
    if "attention_o_projection" in flags:
        interpretations.append("attention_output_state_is_a_necessary_mediator")
    if "mlp_down_projection" in flags:
        interpretations.append("mlp_output_state_is_a_necessary_mediator")
    if (
        "block_output_residual" in flags
        and "attention_o_projection" not in flags
        and "mlp_down_projection" not in flags
    ):
        interpretations.append("residual_or_nonlinear_interaction_remains_unresolved")
    if not interpretations:
        interpretations.append("no_registered_component_state_gate_passed")
    return {
        "component_state_supported_nodes": state_supported,
        "design_prioritized_nodes": design_prioritized,
        "interpretations": interpretations,
        "model_scoped_only": True,
        "operator_or_origin_claim_authorized": False,
    }


def _claim_role(node: str) -> str:
    return {
        "block_input_residual": "upstream_incoming_state_control",
        "attention_o_projection": "attention_branch_state_mediator",
        "mlp_down_projection": "mlp_branch_state_mediator",
        "block_output_residual": "complete_block_state_ceiling",
    }[node]


def _row_for(
    rows: Sequence[Mapping[str, Any]], method_id: str, node: str
) -> Mapping[str, Any]:
    matches = [
        row
        for row in rows
        if row["method_id"] == method_id and row["node"] == node
    ]
    if len(matches) != 1:
        raise AssertionError("component-design row identity drift")
    return matches[0]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"
