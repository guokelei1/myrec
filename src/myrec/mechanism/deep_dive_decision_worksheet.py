"""Outcome-independent, deliberately non-final deep-dive decision worksheet.

The worksheet exposes the complete frozen decision topology while keeping every
scientific judgement empty.  It is intentionally not accepted by
``validate_deep_dive_report_decisions`` and therefore cannot be mistaken for a
finished mechanism report before all registered evidence has closed.
"""

from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any

from myrec.mechanism.deep_dive_closeout_audit import (
    EXPECTED_DELIVERABLES,
    EXPECTED_FROZEN_ASSETS,
)
from myrec.mechanism.deep_dive_opportunity_catalog import (
    OPPORTUNITY_DESIGN_CATALOG,
    OPPORTUNITY_DESIGN_CATALOG_SHA256,
    OPPORTUNITY_IDS,
)
from myrec.mechanism.deep_dive_report_contract import (
    ATTRIBUTION_EVIDENCE_STRENGTHS,
    COMPONENT_ALLOWED_DELIVERABLES,
    COMPONENT_DELIVERABLE_MODEL_COVERAGE,
    COMPONENT_IDS,
    COMPONENT_PROBE_CLAIM_BOUNDARIES,
    COMPONENT_STATUSES,
    NEGATIVE_EVIDENCE_BASES,
    COMPONENT_SUPPORT_MECHANICAL_DEPENDENCIES,
    COMPONENT_SUPPORT_REQUIRED_CAUSAL_EVIDENCE,
    HYPOTHESIS_ALLOWED_DELIVERABLES,
    HYPOTHESIS_IDS,
    HYPOTHESIS_STATUSES,
    HYPOTHESIS_SUPPORTED_REQUIRED_EVIDENCE_GROUPS,
    CROSS_MODEL_ATTRIBUTION_BOUNDARY,
    CROSS_MODEL_ATTRIBUTION_SCOPES,
    OPPORTUNITY_ALLOWED_DELIVERABLES,
    OPPORTUNITY_ALLOWED_MODEL_SCOPE,
    OPPORTUNITY_PRIMARY_PER_MODEL_EVIDENCE_GROUPS,
    OPPORTUNITY_PRIMARY_REQUIRED_EVIDENCE_GROUPS,
    OPPORTUNITY_STATUSES,
    PRIMARY_ATTRIBUTION_CRITERION_DESCRIPTIONS,
    PRIMARY_ATTRIBUTION_DERIVATION_PRECEDENCE,
    PRIMARY_ATTRIBUTION_ENDPOINT_SCOPE,
    PRIMARY_ATTRIBUTION_FOLD_SCOPE,
    PRIMARY_ATTRIBUTION_INTERVENTION_SCOPE,
    PRIMARY_ATTRIBUTION_LAYER_SELECTION_SCOPE,
    PRIMARY_ATTRIBUTION_STRENGTH_RULES,
    PRIMARY_ATTRIBUTION_ALLOWED_DELIVERABLES,
    PRIMARY_ATTRIBUTION_MODELS,
    PRIMARY_LOSS_COMPONENTS,
    REQUIRED_ASSERTIONS,
    REQUIRED_NARRATIVE_FIELDS,
    RESULT_LEVEL_EQUIVALENCE_COMPONENT_GATE_DESCRIPTIONS,
    TRANSFER_FAILURE_CAUSAL_SCOPES,
    TRANSFER_EXPLANATION_LADDER_SCOPE,
    TRANSFER_EXPLANATION_LEVELS,
)


WORKSHEET_ANALYSIS_TYPE = "transformer_deep_dive_decision_worksheet"
WORKSHEET_STATUS = "todo_not_final_report_input"
PRIMARY_FLAG_FIELDS = (
    "fold1_transition_reproduced",
    "attention_branch_registered_support",
    "mlp_branch_registered_support",
    "postblock_registered_support",
    "residual_composition_criterion_met",
    "residual_norm_interaction_criterion_met",
    "node_effect_two_fold_replication_tested",
    "split_sample_component_localization",
    "earliest_loss_layer_established",
    "global_unique_loss_layer_established",
    "history_token_flow_directly_observed_by_layer_scan",
    "exact_layer_index_is_architecture_evidence",
    "cross_model_exact_layer_generalization_authorized",
    "design_implication_requires_component_or_distributed_pattern_evidence",
    "strict_transfer_ndcg_component_corroborated",
    "strict_transfer_ndcg_beyond_equivalence_component_corroborated",
    "target_margin_component_is_not_automatically_ndcg_cause",
    "component_erasure_boundary_established",
    "necessity_tested",
    "exclusive_component_origin_established",
    "additive_or_shapley_contribution_estimated",
    "descriptive_localization_used_as_primary_cause",
)


class DeepDiveDecisionWorksheetError(ValueError):
    """The frozen decision topology cannot be safely materialized."""


def build_deep_dive_decision_worksheet(root: str | Path) -> dict[str, Any]:
    """Build a complete TODO worksheet without opening outcomes or qrels."""

    root_path = Path(root).resolve()
    frozen_sources = _verify_frozen_sources(root_path)
    return {
        "schema_version": 1,
        "analysis_type": WORKSHEET_ANALYSIS_TYPE,
        "status": WORKSHEET_STATUS,
        "final_report_input": False,
        "final_validator_expected_to_reject": True,
        "scientific_effect_values_read": False,
        "qrels_read": False,
        "source_test_opened": False,
        "frozen_sources": frozen_sources,
        "opportunity_design_catalog_sha256": (
            OPPORTUNITY_DESIGN_CATALOG_SHA256
        ),
        "instructions": [
            "Do not fill scientific decisions before all 19 deliverables pass closeout.",
            "Use only the allowed evidence and model scope recorded on each row.",
            "Retain negative outcomes and every admitted mechanical-failure record.",
            "Assign every admitted mechanical-failure path to at least one component row.",
            "After filling, construct a separate final decisions payload and validate it.",
        ],
        "closeout_gate": {
            "all_19_deliverables_required": True,
            "required_deliverables": [
                {
                    "deliverable_id": deliverable_id,
                    "path": EXPECTED_DELIVERABLES[deliverable_id],
                }
                for deliverable_id in sorted(EXPECTED_DELIVERABLES)
            ],
            "decision_values_populated": False,
            "all_admitted_mechanical_failures_require_component_assignment": True,
        },
        "narratives": {
            field: None for field in REQUIRED_NARRATIVE_FIELDS
        },
        "component_evidence_matrix": [
            _component_row(component_id) for component_id in COMPONENT_IDS
        ],
        "component_practical_equivalence_gate_catalog": [
            {
                "component_id": component_id,
                "deliverable": deliverable,
                "registered_practical_equivalence_gate": description,
            }
            for (component_id, deliverable), description in sorted(
                RESULT_LEVEL_EQUIVALENCE_COMPONENT_GATE_DESCRIPTIONS.items()
            )
        ],
        "primary_loss_attribution": [
            _primary_attribution_row(method_id)
            for method_id in PRIMARY_ATTRIBUTION_MODELS
        ],
        "cross_model_primary_attribution": {
            "allowed_scopes": list(CROSS_MODEL_ATTRIBUTION_SCOPES),
            "boundary": deepcopy(CROSS_MODEL_ATTRIBUTION_BOUNDARY),
            "decision": {
                "scope": None,
                "q2_primary_component": None,
                "q3_primary_component": None,
                "shared_primary_component": None,
                "both_models_resolved": None,
                "same_component_across_models": None,
                "generalization_beyond_q2_q3_authorized": None,
                "universal_llm4rec_mechanism_claim_authorized": None,
                "rationale": None,
                "claim_boundary": None,
            },
        },
        "hypothesis_status_matrix": [
            _hypothesis_row(hypothesis_id) for hypothesis_id in HYPOTHESIS_IDS
        ],
        "architecture_opportunity_ranking": [
            _opportunity_row(opportunity_id)
            for opportunity_id in OPPORTUNITY_IDS
        ],
        "required_boundary_assertions": dict(REQUIRED_ASSERTIONS),
        "boundary_assertions": {
            field: None for field in REQUIRED_ASSERTIONS
        },
    }


def _component_row(component_id: str) -> dict[str, Any]:
    allowed = sorted(COMPONENT_ALLOWED_DELIVERABLES[component_id])
    coverage = {
        deliverable: sorted(
            COMPONENT_DELIVERABLE_MODEL_COVERAGE[component_id][deliverable]
        )
        for deliverable in allowed
    }
    allowed_models = sorted(
        set().union(*(set(models) for models in coverage.values()))
    )
    causal = sorted(COMPONENT_SUPPORT_REQUIRED_CAUSAL_EVIDENCE[component_id])
    equivalence_routes = [
        {
            "deliverable": deliverable,
            "registered_practical_equivalence_gate": description,
        }
        for (route_component, deliverable), description in sorted(
            RESULT_LEVEL_EQUIVALENCE_COMPONENT_GATE_DESCRIPTIONS.items()
        )
        if route_component == component_id
    ]
    return {
        "component_id": component_id,
        "allowed_statuses": sorted(COMPONENT_STATUSES),
        "allowed_negative_evidence_bases": sorted(NEGATIVE_EVIDENCE_BASES),
        "allowed_model_scope": allowed_models,
        "allowed_evidence_deliverables": allowed,
        "causal_support_capable_deliverables": causal,
        "causal_support_mechanical_dependencies": {
            deliverable: sorted(dependencies)
            for deliverable, dependencies in sorted(
                COMPONENT_SUPPORT_MECHANICAL_DEPENDENCIES.get(
                    component_id, {}
                ).items()
            )
        },
        "support_authorized_in_this_stage": bool(causal),
        "registered_practical_equivalence_routes": equivalence_routes,
        "registered_probe_claim_boundary": COMPONENT_PROBE_CLAIM_BOUNDARIES[
            component_id
        ],
        "deliverable_model_coverage": coverage,
        "decision": {
            "model_scope": None,
            "status": None,
            "negative_evidence_basis": None,
            "finding": None,
            "claim_boundary": None,
            "optimization_implication": None,
            "evidence_deliverables": None,
            "mechanical_failure_records": None,
        },
    }


def _primary_attribution_row(method_id: str) -> dict[str, Any]:
    decision = {
        "primary_component": None,
        "evidence_strength": None,
        **{field: None for field in PRIMARY_FLAG_FIELDS},
        "selected_branch_node_inference_fold": None,
        "selected_transition_interpretation": None,
        "layer_scan_role": None,
        "layer_scan_observed_state_scope": None,
        "history_effect_interpretation": None,
        "primary_attribution_endpoint": None,
        "transfer_failure_causal_scope": None,
        "transfer_explanation_level": None,
        "causal_intervention_role": None,
        "primary_component_interpretation": None,
        "within_block_adjacent_change_role": None,
        "rationale": None,
        "claim_boundary": None,
        "evidence_deliverables": None,
    }
    return {
        "method_id": method_id,
        "allowed_primary_components": sorted(PRIMARY_LOSS_COMPONENTS),
        "allowed_evidence_strengths": sorted(
            ATTRIBUTION_EVIDENCE_STRENGTHS
        ),
        "allowed_evidence_deliverables": sorted(
            PRIMARY_ATTRIBUTION_ALLOWED_DELIVERABLES
        ),
        "required_decisive_evidence": [
            "d2_postblock",
            "d2_selected_branches",
        ],
        "machine_derived_primary_component_required": True,
        "machine_derived_evidence_strength_required": True,
        "deterministic_precedence": list(
            PRIMARY_ATTRIBUTION_DERIVATION_PRECEDENCE
        ),
        "criterion_descriptions": dict(
            PRIMARY_ATTRIBUTION_CRITERION_DESCRIPTIONS
        ),
        "evidence_strength_rules": dict(PRIMARY_ATTRIBUTION_STRENGTH_RULES),
        "fold_scope": dict(PRIMARY_ATTRIBUTION_FOLD_SCOPE),
        "layer_selection_scope": dict(
            PRIMARY_ATTRIBUTION_LAYER_SELECTION_SCOPE
        ),
        "endpoint_scope": dict(PRIMARY_ATTRIBUTION_ENDPOINT_SCOPE),
        "intervention_scope": dict(PRIMARY_ATTRIBUTION_INTERVENTION_SCOPE),
        "allowed_transfer_failure_causal_scopes": sorted(
            TRANSFER_FAILURE_CAUSAL_SCOPES
        ),
        "transfer_explanation_ladder": deepcopy(
            TRANSFER_EXPLANATION_LADDER_SCOPE
        ),
        "allowed_transfer_explanation_levels": list(
            TRANSFER_EXPLANATION_LEVELS
        ),
        "residual_node_support_alone_is_not_composition": True,
        "incoming_block_state_support_blocks_residual_or_norm_attribution": True,
        "normalization_attribution_requires_supported_post_norm_with_unsupported_paired_pre_norm_state": True,
        "decision": decision,
    }


def _hypothesis_row(hypothesis_id: str) -> dict[str, Any]:
    required_groups = HYPOTHESIS_SUPPORTED_REQUIRED_EVIDENCE_GROUPS[
        hypothesis_id
    ]
    allowed_statuses = set(HYPOTHESIS_STATUSES)
    if hypothesis_id == "H5":
        allowed_statuses -= {"supported", "rejected"}
    return {
        "hypothesis_id": hypothesis_id,
        "allowed_statuses": sorted(allowed_statuses),
        "allowed_negative_evidence_bases": sorted(NEGATIVE_EVIDENCE_BASES),
        "allowed_evidence_deliverables": sorted(
            HYPOTHESIS_ALLOWED_DELIVERABLES[hypothesis_id]
        ),
        "supported_required_evidence_groups": [
            sorted(group) for group in required_groups
        ],
        "support_authorized_in_this_stage": bool(required_groups),
        "decision": {
            "status": None,
            "negative_evidence_basis": None,
            "rationale": None,
            "remaining_uncertainty": None,
            "evidence_deliverables": None,
        },
    }


def _opportunity_row(opportunity_id: str) -> dict[str, Any]:
    per_model_groups = OPPORTUNITY_PRIMARY_PER_MODEL_EVIDENCE_GROUPS.get(
        opportunity_id, ()
    )
    return {
        "opportunity_id": opportunity_id,
        "frozen_design": deepcopy(OPPORTUNITY_DESIGN_CATALOG[opportunity_id]),
        "allowed_statuses": sorted(OPPORTUNITY_STATUSES),
        "allowed_model_scope": sorted(
            OPPORTUNITY_ALLOWED_MODEL_SCOPE[opportunity_id]
        ),
        "allowed_evidence_deliverables": sorted(
            OPPORTUNITY_ALLOWED_DELIVERABLES[opportunity_id]
        ),
        "primary_required_evidence_groups": [
            sorted(group)
            for group in OPPORTUNITY_PRIMARY_REQUIRED_EVIDENCE_GROUPS[
                opportunity_id
            ]
        ],
        "primary_per_model_evidence_groups": [
            sorted(group) for group in per_model_groups
        ],
        "decision": {
            "rank": None,
            "status": None,
            "model_scope": None,
            "rationale": None,
            "falsification_gate": None,
            "evidence_deliverables": None,
        },
    }


def _verify_frozen_sources(root: Path) -> list[dict[str, Any]]:
    rows = []
    for relative_path, expected_sha256 in sorted(
        EXPECTED_FROZEN_ASSETS.items()
    ):
        path = root / relative_path
        try:
            observed_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            raise DeepDiveDecisionWorksheetError(
                f"cannot read frozen source: {relative_path}"
            ) from exc
        if observed_sha256 != expected_sha256:
            raise DeepDiveDecisionWorksheetError(
                f"frozen source hash differs: {relative_path}"
            )
        rows.append(
            {
                "path": relative_path,
                "sha256": observed_sha256,
                "expected_sha256": expected_sha256,
            }
        )
    return rows


__all__ = [
    "DeepDiveDecisionWorksheetError",
    "PRIMARY_FLAG_FIELDS",
    "WORKSHEET_ANALYSIS_TYPE",
    "WORKSHEET_STATUS",
    "build_deep_dive_decision_worksheet",
]
