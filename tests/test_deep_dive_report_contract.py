from __future__ import annotations

import json
import re
from copy import deepcopy

import pytest

from myrec.mechanism.deep_dive_closeout_audit import EXPECTED_DELIVERABLES
from myrec.mechanism.deep_dive_opportunity_catalog import (
    OPPORTUNITY_DESIGN_CATALOG,
)
from myrec.mechanism.deep_dive_report_contract import (
    COMPONENT_ALLOWED_DELIVERABLES,
    COMPONENT_DELIVERABLE_MODEL_COVERAGE,
    COMPONENT_IDS,
    COMPONENT_PROBE_CLAIM_BOUNDARIES,
    CROSS_MODEL_ATTRIBUTION_BOUNDARY,
    CROSS_MODEL_ATTRIBUTION_SCOPES,
    D7_OBJECTIVE_FAMILY_ENDPOINTS,
    D7_OBJECTIVE_FAMILY_STATES,
    D7_OBJECTIVE_FAMILY_SURFACES,
    HYPOTHESIS_ALLOWED_DELIVERABLES,
    HYPOTHESIS_IDS,
    OPPORTUNITY_ALLOWED_DELIVERABLES,
    OPPORTUNITY_IDS,
    PRIMARY_ATTRIBUTION_DERIVATION_PRECEDENCE,
    PRIMARY_ATTRIBUTION_FOLD_SCOPE,
    PRIMARY_ATTRIBUTION_INTERVENTION_SCOPE,
    REQUIRED_ASSERTIONS,
    REQUIRED_NARRATIVE_FIELDS,
    RESULT_LEVEL_SUPPORTED_COMPONENT_GATE_DESCRIPTIONS,
    RESULT_LEVEL_SUPPORTED_COMPONENT_ROUTES,
    TRANSFER_EXPLANATION_LEVELS,
    DeepDiveReportContractError,
    _validate_primary_attribution_against_evidence,
    _validate_supported_components_against_evidence,
    component_result_practical_equivalence,
    component_result_support,
    derive_primary_attribution_evidence,
    derive_cross_model_primary_attribution,
    validate_deep_dive_report_against_closeout,
    validate_deep_dive_report_decisions,
)


def _valid_decisions():
    return {
        "narratives": {
            field: "Bounded narrative." for field in REQUIRED_NARRATIVE_FIELDS
        },
        "component_evidence_matrix": [
            {
                "component_id": component_id,
                "model_scope": sorted(
                    COMPONENT_DELIVERABLE_MODEL_COVERAGE[component_id][
                        sorted(COMPONENT_ALLOWED_DELIVERABLES[component_id])[0]
                    ]
                ),
                "status": "unresolved",
                "negative_evidence_basis": "not_applicable",
                "finding": "Bounded finding.",
                "claim_boundary": COMPONENT_PROBE_CLAIM_BOUNDARIES[component_id],
                "optimization_implication": "Bounded implication.",
                "evidence_deliverables": [
                    sorted(COMPONENT_ALLOWED_DELIVERABLES[component_id])[0]
                ],
                "mechanical_failure_records": [],
            }
            for component_id in COMPONENT_IDS
        ],
        "primary_loss_attribution": [
            {
                "method_id": method_id,
                "primary_component": "unresolved",
                "evidence_strength": "unresolved",
                "fold1_transition_reproduced": False,
                "attention_branch_registered_support": False,
                "mlp_branch_registered_support": False,
                "postblock_registered_support": False,
                "residual_composition_criterion_met": False,
                "residual_norm_interaction_criterion_met": False,
                "selected_branch_node_inference_fold": 1,
                "node_effect_two_fold_replication_tested": False,
                "split_sample_component_localization": True,
                "selected_transition_interpretation": (
                    "largest_fold0_mean_negative_adjacent_postblock_step"
                ),
                "earliest_loss_layer_established": False,
                "global_unique_loss_layer_established": False,
                "layer_scan_role": (
                    "unbiased_localization_for_component_decomposition"
                ),
                "layer_scan_observed_state_scope": (
                    "native_candidate_scoring_positions_only"
                ),
                "history_effect_interpretation": (
                    "accumulated_state_sufficiency_not_token_path"
                ),
                "history_token_flow_directly_observed_by_layer_scan": False,
                "exact_layer_index_is_architecture_evidence": False,
                "cross_model_exact_layer_generalization_authorized": False,
                "design_implication_requires_component_or_distributed_pattern_evidence": True,
                "primary_attribution_endpoint": "target_margin",
                "strict_transfer_ndcg_component_corroborated": False,
                "strict_transfer_ndcg_beyond_equivalence_component_corroborated": False,
                "target_margin_component_is_not_automatically_ndcg_cause": True,
                "transfer_failure_causal_scope": "unresolved",
                "transfer_explanation_level": "unresolved_or_gate_stopped",
                "causal_intervention_role": "null_context_sufficiency",
                "primary_component_interpretation": (
                    "registered_candidate_bottleneck_not_unique_origin"
                ),
                "within_block_adjacent_change_role": (
                    "descriptive_only_without_registered_directional_gate"
                ),
                "component_erasure_boundary_established": False,
                "necessity_tested": False,
                "exclusive_component_origin_established": False,
                "additive_or_shapley_contribution_estimated": False,
                "descriptive_localization_used_as_primary_cause": False,
                "rationale": "Bounded rationale.",
                "claim_boundary": PRIMARY_ATTRIBUTION_INTERVENTION_SCOPE[
                    "claim_boundary"
                ],
                "evidence_deliverables": [
                    "d2_postblock",
                    "d2_selected_branches",
                ],
            }
            for method_id in (
                "q2_recranker_generalqwen",
                "q3_tallrec_generalqwen",
            )
        ],
        "cross_model_primary_attribution": {
            "scope": "no_registered_component_resolution",
            "q2_primary_component": "unresolved",
            "q3_primary_component": "unresolved",
            "shared_primary_component": None,
            "both_models_resolved": False,
            "same_component_across_models": False,
            "generalization_beyond_q2_q3_authorized": False,
            "universal_llm4rec_mechanism_claim_authorized": False,
            "rationale": "Bounded cross-model rationale.",
            "claim_boundary": CROSS_MODEL_ATTRIBUTION_BOUNDARY[
                "claim_boundary"
            ],
        },
        "hypothesis_status_matrix": [
            {
                "hypothesis_id": hypothesis_id,
                "status": "weakened" if hypothesis_id != "H5" else "unresolved",
                "negative_evidence_basis": (
                    "mixed_registered_evidence"
                    if hypothesis_id != "H5"
                    else "not_applicable"
                ),
                "rationale": "Bounded rationale.",
                "remaining_uncertainty": "Bounded uncertainty.",
                "evidence_deliverables": [
                    sorted(HYPOTHESIS_ALLOWED_DELIVERABLES[hypothesis_id])[0]
                ],
            }
            for hypothesis_id in HYPOTHESIS_IDS
        ],
        "architecture_opportunity_ranking": [
            {
                "opportunity_id": opportunity_id,
                "rank": rank,
                "status": "primary" if rank == 1 else "secondary",
                "model_scope": (
                    ["q2_recranker_generalqwen"]
                    if opportunity_id == "OP_H4_SURFACE_AWARE_GRADIENT_BUDGET"
                    else [
                        "q2_recranker_generalqwen",
                        "q3_tallrec_generalqwen",
                    ]
                ),
                "rationale": "Bounded rationale.",
                "falsification_gate": "Bounded gate.",
                **deepcopy(OPPORTUNITY_DESIGN_CATALOG[opportunity_id]),
                "evidence_deliverables": [
                    *(
                        ["d3_attention_edges", "d3_attention_heads"]
                        if rank == 1
                        else [
                            sorted(
                                OPPORTUNITY_ALLOWED_DELIVERABLES[opportunity_id]
                            )[0]
                        ]
                    )
                ],
            }
            for rank, opportunity_id in enumerate(OPPORTUNITY_IDS, start=1)
        ],
        "boundary_assertions": dict(REQUIRED_ASSERTIONS),
    }


def _sync_cross_model(decisions):
    derived = derive_cross_model_primary_attribution(
        decisions["primary_loss_attribution"]
    )
    decisions["cross_model_primary_attribution"] = {
        **derived,
        "rationale": "Bounded cross-model rationale.",
    }


def test_deep_dive_report_contract_accepts_complete_exact_matrices():
    validate_deep_dive_report_decisions(_valid_decisions())


def test_every_causal_support_pair_has_a_result_level_route():
    from myrec.mechanism.deep_dive_report_contract import (
        COMPONENT_SUPPORT_REQUIRED_CAUSAL_EVIDENCE,
    )

    expected = {
        (component_id, deliverable)
        for component_id, deliverables in (
            COMPONENT_SUPPORT_REQUIRED_CAUSAL_EVIDENCE.items()
        )
        for deliverable in deliverables
    }
    assert RESULT_LEVEL_SUPPORTED_COMPONENT_ROUTES == expected
    assert set(RESULT_LEVEL_SUPPORTED_COMPONENT_GATE_DESCRIPTIONS) == expected
    assert all(
        description.strip()
        for description in RESULT_LEVEL_SUPPORTED_COMPONENT_GATE_DESCRIPTIONS.values()
    )


def test_deep_dive_report_contract_rejects_missing_component():
    decisions = _valid_decisions()
    decisions["component_evidence_matrix"].pop()
    with pytest.raises(DeepDiveReportContractError, match="exactly once"):
        validate_deep_dive_report_decisions(decisions)


def test_deep_dive_report_contract_rejects_missing_narrative():
    decisions = _valid_decisions()
    decisions["narratives"].pop("signal_attenuation_answer")
    with pytest.raises(DeepDiveReportContractError, match="exact required fields"):
        validate_deep_dive_report_decisions(decisions)


def test_deep_dive_report_contract_rejects_empty_narrative():
    decisions = _valid_decisions()
    decisions["narratives"]["remaining_uncertainty"] = ""
    with pytest.raises(DeepDiveReportContractError, match="empty remaining_uncertainty"):
        validate_deep_dive_report_decisions(decisions)


def test_deep_dive_report_contract_rejects_unadmitted_evidence():
    decisions = _valid_decisions()
    decisions["hypothesis_status_matrix"][0]["evidence_deliverables"] = ["unknown"]
    with pytest.raises(DeepDiveReportContractError, match="unadmitted"):
        validate_deep_dive_report_decisions(decisions)


def test_deep_dive_report_contract_rejects_semantically_irrelevant_evidence():
    decisions = _valid_decisions()
    decisions["component_evidence_matrix"][0]["evidence_deliverables"] = [
        "d7_optimizer_replay"
    ]
    with pytest.raises(DeepDiveReportContractError, match="semantically irrelevant"):
        validate_deep_dive_report_decisions(decisions)


def test_deep_dive_report_contract_rejects_descriptive_only_mlp_output_support():
    decisions = _valid_decisions()
    row = next(
        item
        for item in decisions["component_evidence_matrix"]
        if item["component_id"] == "mlp_output"
    )
    row["status"] = "supported"
    row["evidence_deliverables"] = ["d4_mlp_groups"]
    with pytest.raises(DeepDiveReportContractError, match="causal evidence: mlp_output"):
        validate_deep_dive_report_decisions(decisions)
    row["evidence_deliverables"].append("d2_selected_branches")
    validate_deep_dive_report_decisions(decisions)


def test_mlp_feature_groups_cannot_be_promoted_from_exploratory_localization():
    decisions = _valid_decisions()
    row = next(
        item
        for item in decisions["component_evidence_matrix"]
        if item["component_id"] == "mlp_feature_formation"
    )
    row["status"] = "supported"
    row["evidence_deliverables"] = ["d4_mlp_groups"]
    with pytest.raises(
        DeepDiveReportContractError,
        match="causal evidence: mlp_feature_formation",
    ):
        validate_deep_dive_report_decisions(decisions)


def test_deep_dive_report_contract_keeps_optimizer_evidence_descriptive():
    decisions = _valid_decisions()
    row = next(
        item
        for item in decisions["component_evidence_matrix"]
        if item["component_id"] == "optimizer_effective_update"
    )
    row["status"] = "supported"
    row["evidence_deliverables"] = ["d7_optimizer_replay"]
    with pytest.raises(
        DeepDiveReportContractError, match="causal evidence: optimizer_effective_update"
    ):
        validate_deep_dive_report_decisions(decisions)


def test_lora_parameterization_identity_and_geometry_remain_descriptive():
    decisions = _valid_decisions()
    row = next(
        item
        for item in decisions["component_evidence_matrix"]
        if item["component_id"] == "lora_parameterization"
    )
    row["status"] = "supported"
    row["model_scope"] = ["q3_tallrec_generalqwen"]
    row["evidence_deliverables"] = ["d7_q3_lora_path"]
    with pytest.raises(
        DeepDiveReportContractError,
        match="causal evidence: lora_parameterization",
    ):
        validate_deep_dive_report_decisions(decisions)


def test_q2_cannot_borrow_q3_lora_replay_coverage():
    decisions = _valid_decisions()
    row = next(
        item
        for item in decisions["component_evidence_matrix"]
        if item["component_id"] == "lora_parameterization"
    )
    row["model_scope"] = ["q2_recranker_generalqwen"]
    row["evidence_deliverables"] = ["d7_optimizer_replay"]
    with pytest.raises(
        DeepDiveReportContractError,
        match="scope lacks direct evidence: lora_parameterization",
    ):
        validate_deep_dive_report_decisions(decisions)


@pytest.mark.parametrize(
    "component_id",
    [
        "token_embedding",
        "attention_output",
        "mlp_feature_formation",
        "mlp_output",
    ],
)
def test_q3_cannot_borrow_q2_full_parameter_update_coverage(component_id):
    decisions = _valid_decisions()
    row = next(
        item
        for item in decisions["component_evidence_matrix"]
        if item["component_id"] == component_id
    )
    row["model_scope"] = ["q3_tallrec_generalqwen"]
    row["evidence_deliverables"] = ["d7_optimizer_replay"]
    with pytest.raises(
        DeepDiveReportContractError,
        match=f"scope lacks direct evidence: {component_id}",
    ):
        validate_deep_dive_report_decisions(decisions)


def test_component_model_scope_requires_direct_evidence_for_every_model():
    decisions = _valid_decisions()
    row = next(
        item
        for item in decisions["component_evidence_matrix"]
        if item["component_id"] == "native_readout"
    )
    row["status"] = "supported"
    row["model_scope"] = [
        "q2_recranker_generalqwen",
        "q3_tallrec_generalqwen",
    ]
    row["evidence_deliverables"] = ["d6_q2_native_readout"]
    with pytest.raises(DeepDiveReportContractError, match="lacks direct evidence"):
        validate_deep_dive_report_decisions(decisions)


def test_component_model_scope_rejects_unknown_model():
    decisions = _valid_decisions()
    decisions["component_evidence_matrix"][0]["model_scope"] = ["q9_unknown"]
    with pytest.raises(DeepDiveReportContractError, match="model scope is invalid"):
        validate_deep_dive_report_decisions(decisions)


def test_deep_dive_report_contract_rejects_boundary_expansion():
    decisions = deepcopy(_valid_decisions())
    decisions["boundary_assertions"]["source_test_opened"] = True
    with pytest.raises(DeepDiveReportContractError, match="assertions differ"):
        validate_deep_dive_report_decisions(decisions)


@pytest.mark.parametrize(
    "assertion",
    (
        "exact_layer_index_used_as_architecture_design_parameter",
        "layer_scan_alone_used_to_rank_architecture_opportunity",
        "selected_branch_sufficiency_alone_used_to_rank_architecture_opportunity",
        "layer_shape_generalized_beyond_frozen_models_or_dataset",
        "p_gt_0p05_or_missing_support_used_as_weakened_or_rejected_evidence",
    ),
)
def test_layer_scan_cannot_authorize_architecture_design(assertion):
    decisions = deepcopy(_valid_decisions())
    decisions["boundary_assertions"][assertion] = True
    with pytest.raises(DeepDiveReportContractError, match="assertions differ"):
        validate_deep_dive_report_decisions(decisions)


def test_frozen_opportunity_catalog_has_no_exact_layer_design_parameter():
    serialized = json.dumps(
        OPPORTUNITY_DESIGN_CATALOG,
        ensure_ascii=False,
        sort_keys=True,
    )
    assert re.search(
        r"\b(?:layer|block)\s*(?:[-:=]\s*)?\d+\b",
        serialized,
        flags=re.IGNORECASE,
    ) is None


def test_human_decisions_cannot_rewrite_machine_claim_boundaries():
    decisions = deepcopy(_valid_decisions())
    decisions["component_evidence_matrix"][0]["claim_boundary"] = (
        "Broader causal claim."
    )
    with pytest.raises(
        DeepDiveReportContractError,
        match="component claim boundary differs",
    ):
        validate_deep_dive_report_decisions(decisions)

    decisions = deepcopy(_valid_decisions())
    decisions["primary_loss_attribution"][0]["claim_boundary"] = (
        "Unique erasure cause."
    )
    with pytest.raises(
        DeepDiveReportContractError,
        match="primary attribution claim boundary differs",
    ):
        validate_deep_dive_report_decisions(decisions)


def test_weakened_or_rejected_status_requires_explicit_negative_evidence_basis():
    decisions = deepcopy(_valid_decisions())
    component = decisions["component_evidence_matrix"][0]
    component["status"] = "weakened"
    component["negative_evidence_basis"] = "not_applicable"
    with pytest.raises(
        DeepDiveReportContractError,
        match="weakened component requires explicit",
    ):
        validate_deep_dive_report_decisions(decisions)

    decisions = deepcopy(_valid_decisions())
    hypothesis = decisions["hypothesis_status_matrix"][0]
    hypothesis["status"] = "weakened"
    hypothesis["negative_evidence_basis"] = "not_applicable"
    with pytest.raises(
        DeepDiveReportContractError,
        match="weakened hypothesis requires explicit",
    ):
        validate_deep_dive_report_decisions(decisions)

    hypothesis["status"] = "rejected"
    hypothesis["negative_evidence_basis"] = "mixed_registered_evidence"
    with pytest.raises(
        DeepDiveReportContractError,
        match="rejected hypothesis requires equivalence",
    ):
        validate_deep_dive_report_decisions(decisions)


def test_supported_h3_requires_causal_and_independent_readout_evidence():
    decisions = _valid_decisions()
    row = next(
        item
        for item in decisions["hypothesis_status_matrix"]
        if item["hypothesis_id"] == "H3"
    )
    row["status"] = "supported"
    row["evidence_deliverables"] = ["d2_selected_branches"]
    with pytest.raises(DeepDiveReportContractError, match="independent evidence groups: H3"):
        validate_deep_dive_report_decisions(decisions)
    row["evidence_deliverables"].append("d6_q2_native_readout")
    validate_deep_dive_report_decisions(decisions)


def test_supported_h1_requires_routing_components_on_one_shared_model():
    decisions = _valid_decisions()
    h1 = next(
        row
        for row in decisions["hypothesis_status_matrix"]
        if row["hypothesis_id"] == "H1"
    )
    h1["status"] = "supported"
    h1["evidence_deliverables"] = [
        "d2_selected_branches",
        "d3_attention_edges",
        "d5_context",
    ]
    with pytest.raises(
        DeepDiveReportContractError,
        match="contradicts component evidence matrix: H1",
    ):
        validate_deep_dive_report_decisions(decisions)
    qk = _supported_component(
        decisions,
        "attention_query_key_routing",
        "q2_recranker_generalqwen",
        "d3_attention_edges",
    )
    qk["evidence_deliverables"].append("d3_attention_heads")
    history = _supported_component(
        decisions,
        "history_routing",
        "q3_tallrec_generalqwen",
        "d2_selected_branches",
    )
    with pytest.raises(
        DeepDiveReportContractError, match="lack a shared model scope: H1"
    ):
        validate_deep_dive_report_decisions(decisions)
    history["model_scope"] = qk["model_scope"]
    validate_deep_dive_report_decisions(decisions)


def test_supported_h4_requires_registered_loss_gradient_component():
    decisions = _valid_decisions()
    h4 = next(
        row
        for row in decisions["hypothesis_status_matrix"]
        if row["hypothesis_id"] == "H4"
    )
    h4["status"] = "supported"
    h4["evidence_deliverables"] = ["d7_q2_objective", "d7_optimizer_replay"]
    with pytest.raises(
        DeepDiveReportContractError,
        match="contradicts component evidence matrix: H4",
    ):
        validate_deep_dive_report_decisions(decisions)
    _supported_component(
        decisions,
        "loss_gradient",
        "q2_recranker_generalqwen",
        "d7_q2_objective",
    )
    validate_deep_dive_report_decisions(decisions)


def test_h5_cannot_be_supported_without_an_independent_seed():
    decisions = _valid_decisions()
    row = next(
        item
        for item in decisions["hypothesis_status_matrix"]
        if item["hypothesis_id"] == "H5"
    )
    row["status"] = "supported"
    row["evidence_deliverables"] = list(EXPECTED_DELIVERABLES)
    with pytest.raises(
        DeepDiveReportContractError, match="independent second seed"
    ):
        validate_deep_dive_report_decisions(decisions)
    row["status"] = "rejected"
    row["negative_evidence_basis"] = "registered_independent_counterexample"
    with pytest.raises(DeepDiveReportContractError, match="independent second seed"):
        validate_deep_dive_report_decisions(decisions)


def test_deep_dive_report_contract_requires_one_rank1_primary_opportunity():
    decisions = _valid_decisions()
    decisions["architecture_opportunity_ranking"][0]["status"] = "secondary"
    with pytest.raises(DeepDiveReportContractError, match="eligible rank-1 primary"):
        validate_deep_dive_report_decisions(decisions)
    decisions["architecture_opportunity_ranking"][0]["status"] = "primary"
    decisions["architecture_opportunity_ranking"][1]["status"] = "primary"
    with pytest.raises(DeepDiveReportContractError, match="at most one rank-1 primary"):
        validate_deep_dive_report_decisions(decisions)


def test_architecture_ranking_allows_no_primary_when_no_candidate_is_eligible():
    decisions = _valid_decisions()
    for row in decisions["hypothesis_status_matrix"]:
        if row["hypothesis_id"] in {"H1", "H2", "H3", "H4"}:
            row["status"] = "unresolved"
    for row in decisions["architecture_opportunity_ranking"]:
        row["status"] = "deprioritized"
    validate_deep_dive_report_decisions(decisions)


def test_no_primary_cannot_be_disguised_as_secondary_rank1():
    decisions = _valid_decisions()
    for row in decisions["hypothesis_status_matrix"]:
        if row["hypothesis_id"] in {"H1", "H2", "H3", "H4"}:
            row["status"] = "unresolved"
    decisions["architecture_opportunity_ranking"][0]["status"] = "secondary"
    with pytest.raises(
        DeepDiveReportContractError,
        match="requires a deprioritized or rejected rank-1",
    ):
        validate_deep_dive_report_decisions(decisions)


def test_architecture_ranking_allows_all_rejected_negative_closeout():
    decisions = _valid_decisions()
    for row in decisions["hypothesis_status_matrix"]:
        if row["hypothesis_id"] in {"H1", "H2", "H3", "H4"}:
            row["status"] = "rejected"
            row["negative_evidence_basis"] = (
                "registered_independent_counterexample"
            )
    for row in decisions["architecture_opportunity_ranking"]:
        row["status"] = "rejected"
    validate_deep_dive_report_decisions(decisions)


def test_deep_dive_report_contract_rejects_descriptive_only_primary_router():
    decisions = _valid_decisions()
    decisions["architecture_opportunity_ranking"][0]["evidence_deliverables"] = [
        "d3_attention_heads"
    ]
    with pytest.raises(DeepDiveReportContractError, match="confirmatory evidence groups"):
        validate_deep_dive_report_decisions(decisions)


def test_combined_primary_opportunity_requires_h2_and_h3_evidence():
    decisions = _valid_decisions()
    router = decisions["architecture_opportunity_ranking"][0]
    combined = decisions["architecture_opportunity_ranking"][3]
    router["rank"], combined["rank"] = combined["rank"], router["rank"]
    router["status"] = "secondary"
    combined["status"] = "primary"
    combined["evidence_deliverables"] = ["d1_representation"]
    with pytest.raises(DeepDiveReportContractError, match="confirmatory evidence groups"):
        validate_deep_dive_report_decisions(decisions)
    combined["evidence_deliverables"].append("d2_selected_branches")
    with pytest.raises(DeepDiveReportContractError, match="confirmatory evidence groups"):
        validate_deep_dive_report_decisions(decisions)
    combined["evidence_deliverables"].extend(
        ["d6_q0_q1_branches", "d6_q2_native_readout"]
    )
    with pytest.raises(
        DeepDiveReportContractError, match="cross-model borrowed evidence"
    ):
        validate_deep_dive_report_decisions(decisions)
    combined["evidence_deliverables"].append("d6_q3_native_readout")
    validate_deep_dive_report_decisions(decisions)


def test_primary_h3_requires_readout_evidence_for_every_scoped_model():
    decisions = _valid_decisions()
    router = decisions["architecture_opportunity_ranking"][0]
    h3 = next(
        row
        for row in decisions["architecture_opportunity_ranking"]
        if row["opportunity_id"]
        == "OP_H3_CANDIDATE_CONDITIONED_SIGNED_PREFERENCE_RESIDUAL"
    )
    router["status"] = "secondary"
    router["rank"], h3["rank"] = h3["rank"], router["rank"]
    h3["status"] = "primary"
    h3["model_scope"] = [
        "q2_recranker_generalqwen",
        "q3_tallrec_generalqwen",
    ]
    h3["evidence_deliverables"] = [
        "d2_selected_branches",
        "d6_q2_native_readout",
    ]
    with pytest.raises(
        DeepDiveReportContractError, match="cross-model borrowed evidence"
    ):
        validate_deep_dive_report_decisions(decisions)
    h3["evidence_deliverables"].append("d6_q3_native_readout")
    validate_deep_dive_report_decisions(decisions)


def test_opportunity_model_scope_requires_direct_registered_evidence():
    decisions = _valid_decisions()
    decisions["architecture_opportunity_ranking"][1]["model_scope"] = [
        "q0_qwen3_reranker_06b"
    ]
    with pytest.raises(DeepDiveReportContractError, match="scope lacks direct evidence"):
        validate_deep_dive_report_decisions(decisions)


def test_opportunity_cannot_expand_beyond_preregistered_model_scope():
    decisions = _valid_decisions()
    decisions["architecture_opportunity_ranking"][0]["model_scope"] = [
        "q0_qwen3_reranker_06b"
    ]
    with pytest.raises(DeepDiveReportContractError, match="preregistered model scope"):
        validate_deep_dive_report_decisions(decisions)


def test_h4_opportunity_cannot_borrow_q3_descriptive_update_geometry():
    decisions = _valid_decisions()
    row = next(
        value
        for value in decisions["architecture_opportunity_ranking"]
        if value["opportunity_id"] == "OP_H4_SURFACE_AWARE_GRADIENT_BUDGET"
    )
    row["model_scope"] = ["q3_tallrec_generalqwen"]
    with pytest.raises(
        DeepDiveReportContractError, match="preregistered model scope"
    ):
        validate_deep_dive_report_decisions(decisions)


def test_combined_signed_path_cannot_borrow_q0_q1_breadth_scope():
    decisions = _valid_decisions()
    row = next(
        value
        for value in decisions["architecture_opportunity_ranking"]
        if value["opportunity_id"]
        == "OP_H2_H3_FACTORIZED_SIGNED_PREFERENCE_PATH"
    )
    row["model_scope"] = ["q0_qwen3_reranker_06b"]
    with pytest.raises(
        DeepDiveReportContractError, match="preregistered model scope"
    ):
        validate_deep_dive_report_decisions(decisions)


def test_opportunity_design_card_requires_critical_ablations():
    decisions = _valid_decisions()
    decisions["architecture_opportunity_ranking"][0]["critical_ablations"] = [
        "Only one"
    ]
    with pytest.raises(DeepDiveReportContractError, match="invalid critical_ablations"):
        validate_deep_dive_report_decisions(decisions)


def test_opportunity_design_card_requires_all_prior_work_comparators():
    decisions = _valid_decisions()
    decisions["architecture_opportunity_ranking"][0][
        "prior_work_differences"
    ].pop("MemRerank")
    with pytest.raises(DeepDiveReportContractError, match="comparators differ"):
        validate_deep_dive_report_decisions(decisions)


def test_opportunity_design_card_cannot_be_rewritten_after_results():
    decisions = _valid_decisions()
    decisions["architecture_opportunity_ranking"][0][
        "innovation_claim"
    ] += " Outcome-fitted addition."
    with pytest.raises(DeepDiveReportContractError, match="design catalog drift"):
        validate_deep_dive_report_decisions(decisions)


def test_primary_opportunity_cannot_override_unresolved_hypothesis():
    decisions = _valid_decisions()
    h1 = next(
        row
        for row in decisions["hypothesis_status_matrix"]
        if row["hypothesis_id"] == "H1"
    )
    h1["status"] = "unresolved"
    with pytest.raises(
        DeepDiveReportContractError, match="supported or weakened linked hypotheses"
    ):
        validate_deep_dive_report_decisions(decisions)


def test_opportunity_must_be_rejected_with_its_linked_hypothesis():
    decisions = _valid_decisions()
    h1 = next(
        row
        for row in decisions["hypothesis_status_matrix"]
        if row["hypothesis_id"] == "H1"
    )
    h1["status"] = "rejected"
    h1["negative_evidence_basis"] = "registered_independent_counterexample"
    with pytest.raises(
        DeepDiveReportContractError, match="contradicts a rejected linked hypothesis"
    ):
        validate_deep_dive_report_decisions(decisions)
    decisions["architecture_opportunity_ranking"][0]["status"] = "rejected"
    decisions["architecture_opportunity_ranking"][1]["status"] = "primary"
    decisions["architecture_opportunity_ranking"][0]["rank"] = 2
    decisions["architecture_opportunity_ranking"][1]["rank"] = 1
    decisions["architecture_opportunity_ranking"][1][
        "evidence_deliverables"
    ] = ["d1_representation", "d6_q0_q1_branches"]
    validate_deep_dive_report_decisions(decisions)


def test_deep_dive_report_contract_requires_per_model_primary_attribution():
    decisions = _valid_decisions()
    decisions["primary_loss_attribution"].pop()
    with pytest.raises(DeepDiveReportContractError, match="exactly once"):
        validate_deep_dive_report_decisions(decisions)


def test_deep_dive_report_contract_rejects_descriptive_primary_cause():
    decisions = _valid_decisions()
    decisions["primary_loss_attribution"][0][
        "descriptive_localization_used_as_primary_cause"
    ] = True
    with pytest.raises(DeepDiveReportContractError, match="descriptive head/group"):
        validate_deep_dive_report_decisions(decisions)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "selected_branch_node_inference_fold",
            0,
            "node inference must be scoped to fold 1",
        ),
        (
            "node_effect_two_fold_replication_tested",
            True,
            "cannot claim two-fold node-effect replication",
        ),
        (
            "split_sample_component_localization",
            False,
            "must retain the split-sample scope",
        ),
    ],
)
def test_primary_attribution_preserves_fold_scope(field, value, message):
    decisions = _valid_decisions()
    decisions["primary_loss_attribution"][0][field] = value
    with pytest.raises(DeepDiveReportContractError, match=message):
        validate_deep_dive_report_decisions(decisions)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("selected_transition_interpretation", "earliest_loss_layer"),
        ("earliest_loss_layer_established", True),
        ("global_unique_loss_layer_established", True),
        ("layer_scan_role", "architecture_target_layer"),
        ("layer_scan_observed_state_scope", "all_history_token_states"),
        ("history_effect_interpretation", "direct_history_token_path"),
        ("history_token_flow_directly_observed_by_layer_scan", True),
        ("exact_layer_index_is_architecture_evidence", True),
        ("cross_model_exact_layer_generalization_authorized", True),
        (
            "design_implication_requires_component_or_distributed_pattern_evidence",
            False,
        ),
    ],
)
def test_primary_attribution_preserves_layer_selection_scope(field, value):
    decisions = _valid_decisions()
    decisions["primary_loss_attribution"][0][field] = value
    with pytest.raises(
        DeepDiveReportContractError,
        match="layer-selection scope differs",
    ):
        validate_deep_dive_report_decisions(decisions)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "primary_attribution_endpoint",
            "ndcg@10",
            "endpoint must remain target_margin",
        ),
        (
            "target_margin_component_is_not_automatically_ndcg_cause",
            False,
            "cannot be promoted automatically",
        ),
        (
            "transfer_failure_causal_scope",
            "target_margin_only",
            "conflicts with endpoint evidence",
        ),
    ],
)
def test_primary_attribution_preserves_transfer_endpoint_scope(
    field, value, message
):
    decisions = _valid_decisions()
    decisions["primary_loss_attribution"][0][field] = value
    with pytest.raises(DeepDiveReportContractError, match=message):
        validate_deep_dive_report_decisions(decisions)


def test_primary_attribution_rejects_transfer_explanation_level_drift():
    decisions = _valid_decisions()
    decisions["primary_loss_attribution"][0]["transfer_explanation_level"] = (
        "target_margin_component_sufficiency"
    )
    with pytest.raises(
        DeepDiveReportContractError,
        match="transfer explanation level conflicts with registered evidence",
    ):
        validate_deep_dive_report_decisions(decisions)


def test_transfer_explanation_ladder_is_strictly_ordered():
    assert TRANSFER_EXPLANATION_LEVELS == (
        "unresolved_or_gate_stopped",
        "reproduced_layer_transition_without_unique_component",
        "target_margin_component_sufficiency",
        "target_margin_component_with_statistical_ndcg_corroboration",
        "target_margin_component_with_utility_relevant_ndcg_corroboration",
    )


def test_cross_model_primary_attribution_rejects_universalization():
    decisions = _valid_decisions()
    decisions["cross_model_primary_attribution"][
        "universal_llm4rec_mechanism_claim_authorized"
    ] = True
    with pytest.raises(
        DeepDiveReportContractError,
        match="cross-model primary attribution differs",
    ):
        validate_deep_dive_report_decisions(decisions)


def test_cross_model_primary_attribution_derives_all_four_scopes():
    def rows(q2_component, q3_component):
        return [
            {
                "method_id": "q2_recranker_generalqwen",
                "primary_component": q2_component,
            },
            {
                "method_id": "q3_tallrec_generalqwen",
                "primary_component": q3_component,
            },
        ]

    cases = (
        (
            "unresolved",
            "unresolved",
            "no_registered_component_resolution",
        ),
        (
            "attention_output",
            "unresolved",
            "single_model_registered_component_only",
        ),
        (
            "attention_output",
            "mlp",
            "model_heterogeneous_registered_components",
        ),
        (
            "attention_output",
            "attention_output",
            "shared_registered_component_sufficiency_across_q2_q3",
        ),
    )
    assert tuple(scope for *_, scope in cases) == CROSS_MODEL_ATTRIBUTION_SCOPES
    for q2_component, q3_component, expected_scope in cases:
        result = derive_cross_model_primary_attribution(
            rows(q2_component, q3_component)
        )
        assert result["scope"] == expected_scope
        assert result["generalization_beyond_q2_q3_authorized"] is False
        assert result["universal_llm4rec_mechanism_claim_authorized"] is False


def test_utility_ndcg_corroboration_requires_statistical_corroboration():
    decisions = _valid_decisions()
    row = decisions["primary_loss_attribution"][0]
    row[
        "strict_transfer_ndcg_beyond_equivalence_component_corroborated"
    ] = True
    with pytest.raises(
        DeepDiveReportContractError,
        match="requires statistical corroboration",
    ):
        validate_deep_dive_report_decisions(decisions)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("causal_intervention_role", "necessity"),
        ("primary_component_interpretation", "unique_causal_origin"),
        ("within_block_adjacent_change_role", "confirmatory_erasure_boundary"),
        ("component_erasure_boundary_established", True),
        ("necessity_tested", True),
        ("exclusive_component_origin_established", True),
        ("additive_or_shapley_contribution_estimated", True),
    ],
)
def test_primary_attribution_preserves_sufficiency_only_scope(field, value):
    decisions = _valid_decisions()
    decisions["primary_loss_attribution"][0][field] = value
    with pytest.raises(
        DeepDiveReportContractError,
        match="intervention scope differs",
    ):
        validate_deep_dive_report_decisions(decisions)


def test_deep_dive_report_contract_requires_d2_causal_attribution_evidence():
    decisions = _valid_decisions()
    decisions["primary_loss_attribution"][0]["evidence_deliverables"] = [
        "d3_attention_heads",
        "d4_mlp_groups",
    ]
    with pytest.raises(DeepDiveReportContractError, match="requires both D2"):
        validate_deep_dive_report_decisions(decisions)


def test_deep_dive_report_contract_enforces_residual_composition_criterion():
    decisions = _valid_decisions()
    row = decisions["primary_loss_attribution"][0]
    row.update(
        {
            "primary_component": "residual_composition",
            "evidence_strength": "registered_confirmatory",
            "fold1_transition_reproduced": True,
            "postblock_registered_support": True,
            "transfer_failure_causal_scope": "target_margin_only",
            "transfer_explanation_level": "target_margin_component_sufficiency",
        }
    )
    with pytest.raises(DeepDiveReportContractError, match="conflicts"):
        validate_deep_dive_report_decisions(decisions)
    row["residual_composition_criterion_met"] = True
    component = next(
        item
        for item in decisions["component_evidence_matrix"]
        if item["component_id"] == "residual_composition"
    )
    component["status"] = "supported"
    component["evidence_deliverables"] = ["d2_selected_branches"]
    _sync_cross_model(decisions)
    validate_deep_dive_report_decisions(decisions)


def test_token_embedding_cannot_borrow_rope_or_readout_causality():
    decisions = _valid_decisions()
    component = next(
        item
        for item in decisions["component_evidence_matrix"]
        if item["component_id"] == "token_embedding"
    )
    component["status"] = "supported"
    component["model_scope"] = ["q2_recranker_generalqwen"]
    component["evidence_deliverables"] = ["d6_q2_native_readout"]
    with pytest.raises(
        DeepDiveReportContractError,
        match="causal evidence: token_embedding",
    ):
        validate_deep_dive_report_decisions(decisions)


def test_positional_encoding_support_requires_rope_causal_evidence():
    decisions = _valid_decisions()
    component = next(
        item
        for item in decisions["component_evidence_matrix"]
        if item["component_id"] == "positional_encoding_rope"
    )
    component["status"] = "supported"
    component["model_scope"] = [
        "q2_recranker_generalqwen",
        "q3_tallrec_generalqwen",
    ]
    component["evidence_deliverables"] = ["d5_rope"]
    validate_deep_dive_report_decisions(decisions)


def test_attention_value_transport_requires_edge_intervention_not_head_observation():
    decisions = _valid_decisions()
    component = next(
        item
        for item in decisions["component_evidence_matrix"]
        if item["component_id"] == "attention_value_transport"
    )
    component["status"] = "supported"
    component["evidence_deliverables"] = ["d3_attention_heads"]
    with pytest.raises(
        DeepDiveReportContractError,
        match="causal evidence: attention_value_transport",
    ):
        validate_deep_dive_report_decisions(decisions)
    component["evidence_deliverables"] = ["d3_attention_edges"]
    with pytest.raises(
        DeepDiveReportContractError,
        match="lacks its mechanical dependency",
    ):
        validate_deep_dive_report_decisions(decisions)
    component["evidence_deliverables"].append("d3_attention_heads")
    validate_deep_dive_report_decisions(decisions)


@pytest.mark.parametrize(
    "component_id",
    ["attention_query_key_routing", "history_routing"],
)
def test_attention_edge_component_support_requires_head_reconstruction_dependency(
    component_id,
):
    decisions = _valid_decisions()
    component = next(
        item
        for item in decisions["component_evidence_matrix"]
        if item["component_id"] == component_id
    )
    component["status"] = "supported"
    component["evidence_deliverables"] = ["d3_attention_edges"]
    with pytest.raises(
        DeepDiveReportContractError,
        match="lacks its mechanical dependency",
    ):
        validate_deep_dive_report_decisions(decisions)
    component["evidence_deliverables"].append("d3_attention_heads")
    validate_deep_dive_report_decisions(decisions)


def test_deep_dive_report_contract_requires_postblock_for_attention_attribution():
    decisions = _valid_decisions()
    row = decisions["primary_loss_attribution"][0]
    row.update(
        {
            "primary_component": "attention_output",
            "evidence_strength": "registered_confirmatory",
            "fold1_transition_reproduced": True,
            "attention_branch_registered_support": True,
            "transfer_failure_causal_scope": "target_margin_only",
            "transfer_explanation_level": "target_margin_component_sufficiency",
        }
    )
    with pytest.raises(DeepDiveReportContractError, match="conflicts"):
        validate_deep_dive_report_decisions(decisions)
    row["postblock_registered_support"] = True
    with pytest.raises(
        DeepDiveReportContractError, match="contradicts component evidence matrix"
    ):
        validate_deep_dive_report_decisions(decisions)
    component = next(
        item
        for item in decisions["component_evidence_matrix"]
        if item["component_id"] == "attention_output"
    )
    component["status"] = "supported"
    component["evidence_deliverables"] = ["d2_selected_branches"]
    _sync_cross_model(decisions)
    validate_deep_dive_report_decisions(decisions)


def test_primary_attribution_requires_component_support_for_same_model():
    decisions = _valid_decisions()
    row = decisions["primary_loss_attribution"][0]
    row.update(
        {
            "primary_component": "attention_output",
            "evidence_strength": "registered_confirmatory",
            "fold1_transition_reproduced": True,
            "attention_branch_registered_support": True,
            "postblock_registered_support": True,
            "transfer_failure_causal_scope": "target_margin_only",
            "transfer_explanation_level": "target_margin_component_sufficiency",
        }
    )
    component = next(
        item
        for item in decisions["component_evidence_matrix"]
        if item["component_id"] == "attention_output"
    )
    component["status"] = "supported"
    component["model_scope"] = ["q3_tallrec_generalqwen"]
    component["evidence_deliverables"] = ["d2_selected_branches"]
    with pytest.raises(
        DeepDiveReportContractError, match="contradicts component evidence matrix"
    ):
        validate_deep_dive_report_decisions(decisions)


def test_deep_dive_report_contract_requires_all_19_deliverables():
    admitted = list(EXPECTED_DELIVERABLES)[:-1]
    with pytest.raises(DeepDiveReportContractError, match="all 19"):
        validate_deep_dive_report_decisions(
            _valid_decisions(), admitted_deliverables=admitted
        )


def test_every_admitted_mechanical_failure_requires_component_assignment():
    failure = "runs/example/mechanical_failure_record.json"
    with pytest.raises(
        DeepDiveReportContractError,
        match="lack component assignments",
    ):
        validate_deep_dive_report_decisions(
            _valid_decisions(), admitted_failure_records=[failure]
        )


def test_admitted_mechanical_failure_may_coexist_with_completed_evidence():
    failure = "runs/example/mechanical_failure_record.json"
    decisions = _valid_decisions()
    decisions["component_evidence_matrix"][0][
        "mechanical_failure_records"
    ] = [failure]
    validate_deep_dive_report_decisions(
        decisions, admitted_failure_records=[failure]
    )


def test_deep_dive_report_contract_refuses_pending_closeout(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "myrec.mechanism.deep_dive_report_contract.audit_deep_dive_closeout",
        lambda _root: {"status": "pending"},
    )
    with pytest.raises(DeepDiveReportContractError, match="must be completed"):
        validate_deep_dive_report_against_closeout(tmp_path, _valid_decisions())


def test_deep_dive_report_contract_accepts_only_completed_closeout(
    monkeypatch, tmp_path
):
    closeout = {
        "status": "completed",
        "deliverables": {
            name: {"status": "completed"} for name in EXPECTED_DELIVERABLES
        },
        "mechanical_failure_records": [],
    }
    monkeypatch.setattr(
        "myrec.mechanism.deep_dive_report_contract.audit_deep_dive_closeout",
        lambda _root: closeout,
    )
    monkeypatch.setattr(
        "myrec.mechanism.deep_dive_report_contract._validate_primary_attribution_against_evidence",
        lambda _root, _decisions: None,
    )
    assert (
        validate_deep_dive_report_against_closeout(
            tmp_path, _valid_decisions()
        )
        is closeout
    )


def test_primary_attribution_flags_are_crosschecked_against_d2_evidence(tmp_path):
    postblock_path = (
        tmp_path
        / "runs/20260718_kuaisearch_mech_d2_postblock_synthesis_v1/metrics.json"
    )
    selected_path = (
        tmp_path
        / "runs/20260718_kuaisearch_mech_d2_selected_branch_synthesis_v1/metrics.json"
    )
    postblock_path.parent.mkdir(parents=True)
    selected_path.parent.mkdir(parents=True)
    postblock_path.write_text(
        json.dumps(
            {
                "analysis_type": "transformer_deep_dive_d2_postblock_synthesis",
                "status": "completed",
                "localization": {
                    "q2_recranker_generalqwen": {"resolved": True},
                    "q3_tallrec_generalqwen": {"resolved": False},
                },
            }
        )
    )
    node = "attention_o_projection"
    contrast_ids = (
        f"same__{node}",
        f"same_minus_cross__{node}",
        f"same_minus_wrong__{node}",
        f"norm__{node}",
        f"direction__{node}",
        f"random__{node}",
    )
    selected_path.write_text(
        json.dumps(
            {
                "analysis_type": "transformer_deep_dive_d2_selected_branch_synthesis",
                "status": "completed",
                "fold_scope": dict(PRIMARY_ATTRIBUTION_FOLD_SCOPE),
                "rows": [
                    {
                        "method_id": "q2_recranker_generalqwen",
                        "endpoint": "target_margin",
                        "contrast_id": contrast_id,
                        "registered_support": True,
                        "missing": False,
                        "evidence_role": "registered_confirmatory_branch_localization",
                    }
                    for contrast_id in contrast_ids
                ],
            }
        )
    )
    decisions = _valid_decisions()
    q2 = decisions["primary_loss_attribution"][0]
    q2.update(
        {
            "primary_component": "attention_output",
            "evidence_strength": "registered_confirmatory",
            "fold1_transition_reproduced": True,
            "attention_branch_registered_support": True,
            "postblock_registered_support": True,
            "transfer_failure_causal_scope": "target_margin_only",
            "transfer_explanation_level": "target_margin_component_sufficiency",
        }
    )
    _validate_primary_attribution_against_evidence(tmp_path, decisions)
    q2["evidence_strength"] = "exploratory_only"
    with pytest.raises(
        DeepDiveReportContractError,
        match="strength differs from deterministic D2 evidence",
    ):
        _validate_primary_attribution_against_evidence(tmp_path, decisions)
    q2["evidence_strength"] = "registered_confirmatory"
    q2["attention_branch_registered_support"] = False
    with pytest.raises(DeepDiveReportContractError, match="differs from D2 evidence"):
        _validate_primary_attribution_against_evidence(tmp_path, decisions)
    q2["attention_branch_registered_support"] = True
    q2["residual_composition_criterion_met"] = True
    with pytest.raises(DeepDiveReportContractError, match="differs from D2 evidence"):
        _validate_primary_attribution_against_evidence(tmp_path, decisions)
    q2["residual_composition_criterion_met"] = False
    q2["residual_norm_interaction_criterion_met"] = True
    with pytest.raises(DeepDiveReportContractError, match="differs from D2 evidence"):
        _validate_primary_attribution_against_evidence(tmp_path, decisions)
    q2["residual_norm_interaction_criterion_met"] = False
    q2["primary_component"] = "mixed_attention_mlp"
    with pytest.raises(
        DeepDiveReportContractError,
        match="label differs from deterministic D2 evidence",
    ):
        _validate_primary_attribution_against_evidence(tmp_path, decisions)


def test_primary_attribution_residual_norm_precedence_is_outcome_independent(tmp_path):
    postblock_path = (
        tmp_path
        / "runs/20260718_kuaisearch_mech_d2_postblock_synthesis_v1/metrics.json"
    )
    selected_path = (
        tmp_path
        / "runs/20260718_kuaisearch_mech_d2_selected_branch_synthesis_v1/metrics.json"
    )
    postblock_path.parent.mkdir(parents=True)
    selected_path.parent.mkdir(parents=True)
    postblock_path.write_text(
        json.dumps(
            {
                "analysis_type": "transformer_deep_dive_d2_postblock_synthesis",
                "status": "completed",
                "localization": {
                    "q2_recranker_generalqwen": {"resolved": True},
                    "q3_tallrec_generalqwen": {"resolved": False},
                },
            }
        )
    )

    def support_rows(node):
        return [
            {
                "method_id": "q2_recranker_generalqwen",
                "endpoint": "target_margin",
                "contrast_id": f"{prefix}__{node}",
                "registered_support": True,
                "missing": False,
                "evidence_role": "registered_confirmatory_branch_localization",
            }
            for prefix in (
                "same",
                "same_minus_cross",
                "same_minus_wrong",
                "norm",
                "direction",
                "random",
            )
        ]

    selected_path.write_text(
        json.dumps(
            {
                "analysis_type": "transformer_deep_dive_d2_selected_branch_synthesis",
                "status": "completed",
                "fold_scope": dict(PRIMARY_ATTRIBUTION_FOLD_SCOPE),
                "rows": support_rows("post_attention_residual")
                + support_rows("input_rmsnorm_output"),
            }
        )
    )
    decisions = _valid_decisions()
    q2 = decisions["primary_loss_attribution"][0]
    q2.update(
        {
            "primary_component": "residual_norm_interaction",
            "evidence_strength": "registered_confirmatory",
            "fold1_transition_reproduced": True,
            "postblock_registered_support": True,
            "residual_composition_criterion_met": False,
            "residual_norm_interaction_criterion_met": True,
            "transfer_failure_causal_scope": "target_margin_only",
            "transfer_explanation_level": "target_margin_component_sufficiency",
        }
    )
    _validate_primary_attribution_against_evidence(tmp_path, decisions)

    q2["primary_component"] = "residual_composition"
    q2["residual_composition_criterion_met"] = True
    q2["residual_norm_interaction_criterion_met"] = False
    with pytest.raises(DeepDiveReportContractError, match="differs from D2 evidence"):
        _validate_primary_attribution_against_evidence(tmp_path, decisions)


def test_post_norm_support_requires_paired_boundary_isolation_for_norm_label():
    def support_rows(node):
        return [
            {
                "method_id": "q2_recranker_generalqwen",
                "endpoint": "target_margin",
                "contrast_id": f"{prefix}__{node}",
                "registered_support": True,
                "missing": False,
                "evidence_role": "registered_confirmatory_branch_localization",
            }
            for prefix in (
                "same",
                "same_minus_cross",
                "same_minus_wrong",
                "norm",
                "direction",
                "random",
            )
        ]

    postblock = {
        "localization": {
            "q2_recranker_generalqwen": {"status": "resolved", "resolved": True},
            "q3_tallrec_generalqwen": {
                "status": "fold0_no_negative_transition",
                "resolved": False,
            },
        }
    }
    isolated = derive_primary_attribution_evidence(
        postblock,
        {
            "fold_scope": dict(PRIMARY_ATTRIBUTION_FOLD_SCOPE),
            "rows": support_rows("post_attention_rmsnorm_output"),
        },
    )[0]
    assert isolated["normalization_boundary_isolated_registered_support"] is True
    assert isolated["normalization_boundary_isolated_nodes"] == [
        "post_attention_rmsnorm_output"
    ]
    assert isolated["normalization_state_support_without_boundary_isolation"] is False
    assert isolated["residual_norm_interaction_criterion_met"] is True
    assert isolated["derived_primary_component"] == "residual_norm_interaction"

    carried = derive_primary_attribution_evidence(
        postblock,
        {
            "fold_scope": dict(PRIMARY_ATTRIBUTION_FOLD_SCOPE),
            "rows": support_rows("post_attention_residual")
            + support_rows("post_attention_rmsnorm_output"),
        },
    )[0]
    assert carried["normalization_boundary_isolated_registered_support"] is False
    assert carried["normalization_state_support_without_boundary_isolation"] is True
    assert carried["normalization_nodes_without_boundary_isolation"] == [
        "post_attention_rmsnorm_output"
    ]
    assert carried["residual_norm_interaction_criterion_met"] is False
    assert carried["residual_composition_criterion_met"] is True
    assert carried["derived_primary_component"] == "residual_composition"


def test_component_matrix_residual_and_norm_routes_share_primary_guards():
    def support_rows(*nodes):
        return [
            {
                "method_id": "q2_recranker_generalqwen",
                "endpoint": "target_margin",
                "contrast_id": f"{prefix}__{node}",
                "registered_support": True,
                "missing": False,
                "evidence_role": "registered_confirmatory_branch_localization",
            }
            for node in nodes
            for prefix in (
                "same",
                "same_minus_cross",
                "same_minus_wrong",
                "norm",
                "direction",
                "random",
            )
        ]

    method_id = "q2_recranker_generalqwen"
    residual_only = {"rows": support_rows("post_attention_residual")}
    assert component_result_support(
        "residual_composition",
        method_id,
        "d2_selected_branches",
        residual_only,
    ) is True

    incoming_carried = {
        "rows": support_rows(
            "block_input_residual", "post_attention_residual"
        )
    }
    assert component_result_support(
        "residual_composition",
        method_id,
        "d2_selected_branches",
        incoming_carried,
    ) is False

    attention_explains = {
        "rows": support_rows(
            "attention_o_projection", "post_attention_residual"
        )
    }
    assert component_result_support(
        "residual_composition",
        method_id,
        "d2_selected_branches",
        attention_explains,
    ) is False

    isolated_norm = {"rows": support_rows("post_attention_rmsnorm_output")}
    assert component_result_support(
        "normalization",
        method_id,
        "d2_selected_branches",
        isolated_norm,
    ) is True

    norm_carried = {
        "rows": support_rows(
            "post_attention_residual", "post_attention_rmsnorm_output"
        )
    }
    assert component_result_support(
        "normalization",
        method_id,
        "d2_selected_branches",
        norm_carried,
    ) is False
    assert component_result_support(
        "residual_composition",
        method_id,
        "d2_selected_branches",
        norm_carried,
    ) is True

    isolated_norm_with_attention = {
        "rows": support_rows(
            "attention_o_projection", "post_attention_rmsnorm_output"
        )
    }
    assert component_result_support(
        "normalization",
        method_id,
        "d2_selected_branches",
        isolated_norm_with_attention,
    ) is False


def test_practical_equivalence_routes_require_complete_registered_sesoi_cells():
    def equivalent_rows():
        return [
            {
                "normalized_query_fold": fold,
                "mean": 0.0,
                "ci95": [-0.001, 0.001],
            }
            for fold in ("all", "0", "1")
        ]

    q2_results = {
        node: {
            comparison: {"ndcg@10": equivalent_rows()}
            for comparison in (
                "same_minus_null",
                "same_minus_full",
                "same_minus_cross",
            )
        }
        for node in ("final_rmsnorm_input", "final_rmsnorm_output")
    }
    q2_metrics = {
        "method_id": "q2_recranker_generalqwen",
        "results": q2_results,
    }
    assert component_result_practical_equivalence(
        "native_readout",
        "q2_recranker_generalqwen",
        "d6_q2_native_readout",
        q2_metrics,
    ) is True
    q2_results["final_rmsnorm_output"]["same_minus_cross"]["ndcg@10"][0][
        "ci95"
    ] = [-0.006, 0.001]
    assert component_result_practical_equivalence(
        "native_readout",
        "q2_recranker_generalqwen",
        "d6_q2_native_readout",
        q2_metrics,
    ) is False

    rope_results = {
        str(block): {
            contrast: {
                "ndcg@10": {
                    "registered_compression_minus_expansion": equivalent_rows(),
                    "registered_compression_minus_baseline_support_gate": (
                        equivalent_rows()
                    ),
                }
            }
            for contrast in ("readout_q", "history_k", "paired_qk")
        }
        for block in (13, 20, 27)
    }
    rope_metrics = {
        "position_support_gate": {
            "active_ci95_equivalence_band": [-0.005, 0.005]
        },
        "results": {"q2_recranker_generalqwen": rope_results},
    }
    assert component_result_practical_equivalence(
        "positional_encoding_rope",
        "q2_recranker_generalqwen",
        "d5_rope",
        rope_metrics,
    ) is True
    rope_results["20"]["readout_q"]["ndcg@10"][
        "registered_compression_minus_expansion"
    ][0]["ci95"] = [-0.001, 0.006]
    assert component_result_practical_equivalence(
        "positional_encoding_rope",
        "q2_recranker_generalqwen",
        "d5_rope",
        rope_metrics,
    ) is False

    family_rows = [
        {
            "state": state,
            "surface": surface,
            "endpoint": endpoint,
            "practical_equivalence_within_sesoi": (
                True if endpoint == "ranknet_listnet_cosine" else None
            ),
        }
        for state in D7_OBJECTIVE_FAMILY_STATES
        for surface in D7_OBJECTIVE_FAMILY_SURFACES
        for endpoint in D7_OBJECTIVE_FAMILY_ENDPOINTS
    ]
    objective_metrics = {"family_rows": family_rows}
    assert component_result_practical_equivalence(
        "loss_gradient",
        "q2_recranker_generalqwen",
        "d7_q2_objective",
        objective_metrics,
    ) is True
    next(
        row
        for row in family_rows
        if row["endpoint"] == "ranknet_listnet_cosine"
    )["practical_equivalence_within_sesoi"] = False
    assert component_result_practical_equivalence(
        "loss_gradient",
        "q2_recranker_generalqwen",
        "d7_q2_objective",
        objective_metrics,
    ) is False


def test_practical_equivalence_rejects_malformed_cells_and_route_mismatch():
    def equivalent_rows():
        return [
            {
                "normalized_query_fold": fold,
                "mean": 0.0,
                "ci95": [-0.001, 0.001],
            }
            for fold in ("all", "0", "1")
        ]

    q2_metrics = {
        "method_id": "q2_recranker_generalqwen",
        "results": {
            node: {
                comparison: {"ndcg@10": equivalent_rows()}
                for comparison in (
                    "same_minus_null",
                    "same_minus_full",
                    "same_minus_cross",
                )
            }
            for node in ("final_rmsnorm_input", "final_rmsnorm_output")
        },
    }
    duplicate_fold = deepcopy(q2_metrics)
    duplicate_rows = duplicate_fold["results"]["final_rmsnorm_input"][
        "same_minus_null"
    ]["ndcg@10"]
    duplicate_rows[2]["normalized_query_fold"] = "0"
    assert component_result_practical_equivalence(
        "native_readout",
        "q2_recranker_generalqwen",
        "d6_q2_native_readout",
        duplicate_fold,
    ) is False

    inverted_ci = deepcopy(q2_metrics)
    inverted_ci["results"]["final_rmsnorm_input"]["same_minus_null"][
        "ndcg@10"
    ][0]["ci95"] = [0.004, -0.004]
    assert component_result_practical_equivalence(
        "native_readout",
        "q2_recranker_generalqwen",
        "d6_q2_native_readout",
        inverted_ci,
    ) is False

    q3_metrics = {
        "method_id": "q3_tallrec_generalqwen",
        "results": {
            scope: {
                comparison: {"ndcg@10": equivalent_rows()}
                for comparison in (
                    "same_minus_null",
                    "same_minus_full",
                    "same_minus_cross",
                )
            }
            for scope in ("shared_prompt", "yes_context", "no_context", "joint")
        },
    }
    assert component_result_practical_equivalence(
        "native_readout",
        "q3_tallrec_generalqwen",
        "d6_q2_native_readout",
        q3_metrics,
    ) is False

    missing_registered_comparison = deepcopy(q2_metrics)
    del missing_registered_comparison["results"]["final_rmsnorm_input"][
        "same_minus_full"
    ]
    assert component_result_practical_equivalence(
        "native_readout",
        "q2_recranker_generalqwen",
        "d6_q2_native_readout",
        missing_registered_comparison,
    ) is False

    family_rows = [
        {
            "state": state,
            "surface": surface,
            "endpoint": endpoint,
            "practical_equivalence_within_sesoi": (
                endpoint == "ranknet_listnet_cosine"
            ),
        }
        for state in D7_OBJECTIVE_FAMILY_STATES
        for surface in D7_OBJECTIVE_FAMILY_SURFACES
        for endpoint in D7_OBJECTIVE_FAMILY_ENDPOINTS
    ]
    objective_with_extra_row = {
        "family_rows": family_rows + [{"unexpected": "extra row"}]
    }
    assert component_result_practical_equivalence(
        "loss_gradient",
        "q2_recranker_generalqwen",
        "d7_q2_objective",
        objective_with_extra_row,
    ) is False


def test_residual_node_support_is_not_composition_when_attention_is_sufficient():
    def support_rows(node):
        return [
            {
                "method_id": "q2_recranker_generalqwen",
                "endpoint": "target_margin",
                "contrast_id": f"{prefix}__{node}",
                "registered_support": True,
                "missing": False,
                "evidence_role": (
                    "registered_confirmatory_branch_localization"
                ),
            }
            for prefix in (
                "same",
                "same_minus_cross",
                "same_minus_wrong",
                "norm",
                "direction",
                "random",
            )
        ]

    rows = derive_primary_attribution_evidence(
        {
            "localization": {
                "q2_recranker_generalqwen": {"resolved": True},
                "q3_tallrec_generalqwen": {"resolved": False},
            }
        },
        {
            "fold_scope": dict(PRIMARY_ATTRIBUTION_FOLD_SCOPE),
            "rows": support_rows("attention_o_projection")
            + support_rows("block_output_residual")
        },
    )
    q2 = next(
        row for row in rows if row["method_id"] == "q2_recranker_generalqwen"
    )
    assert q2["attention_branch_registered_support"] is True
    assert q2["residual_node_registered_support"] is True
    assert q2["residual_composition_criterion_met"] is False
    assert q2["derived_primary_component"] == "attention_output"
    assert PRIMARY_ATTRIBUTION_DERIVATION_PRECEDENCE.index(
        "attention_output"
    ) < PRIMARY_ATTRIBUTION_DERIVATION_PRECEDENCE.index(
        "residual_composition"
    )


def test_incoming_block_state_support_prevents_residual_composition_overclaim():
    def support_rows(node):
        return [
            {
                "method_id": "q2_recranker_generalqwen",
                "endpoint": "target_margin",
                "contrast_id": f"{prefix}__{node}",
                "registered_support": True,
                "missing": False,
                "evidence_role": (
                    "registered_confirmatory_branch_localization"
                ),
            }
            for prefix in (
                "same",
                "same_minus_cross",
                "same_minus_wrong",
                "norm",
                "direction",
                "random",
            )
        ]

    rows = derive_primary_attribution_evidence(
        {
            "localization": {
                "q2_recranker_generalqwen": {
                    "status": "resolved",
                    "resolved": True,
                },
                "q3_tallrec_generalqwen": {
                    "status": "fold0_no_negative_transition",
                    "resolved": False,
                },
            }
        },
        {
            "fold_scope": dict(PRIMARY_ATTRIBUTION_FOLD_SCOPE),
            "rows": support_rows("block_input_residual")
            + support_rows("post_attention_residual"),
        },
    )
    q2 = rows[0]
    assert q2["incoming_block_state_registered_support"] is True
    assert q2["incoming_state_confounds_residual_or_norm_attribution"] is True
    assert q2["residual_node_registered_support"] is True
    assert q2["residual_composition_criterion_met"] is False
    assert q2["derived_primary_component"] == "unresolved"
    assert q2["derived_evidence_strength"] == "unresolved"


def test_primary_attribution_requires_same_component_ndcg_corroboration():
    def support_rows(node, endpoint, *, ci95=None):
        return [
            {
                "method_id": "q2_recranker_generalqwen",
                "endpoint": endpoint,
                "contrast_id": f"{prefix}__{node}",
                "registered_support": True,
                "missing": False,
                "ci95": ci95,
                "evidence_role": "registered_confirmatory_branch_localization",
            }
            for prefix in (
                "same",
                "same_minus_cross",
                "same_minus_wrong",
                "norm",
                "direction",
                "random",
            )
        ]

    postblock = {
        "localization": {
            "q2_recranker_generalqwen": {
                "status": "resolved",
                "resolved": True,
            },
            "q3_tallrec_generalqwen": {
                "status": "fold0_no_negative_transition",
                "resolved": False,
            },
        }
    }
    target_only = derive_primary_attribution_evidence(
        postblock,
        {
            "fold_scope": dict(PRIMARY_ATTRIBUTION_FOLD_SCOPE),
            "rows": support_rows("attention_o_projection", "target_margin"),
        },
    )[0]
    assert target_only["derived_primary_component"] == "attention_output"
    assert target_only["strict_transfer_ndcg_component_corroborated"] is False
    assert target_only["transfer_failure_causal_scope"] == "target_margin_only"
    assert target_only["transfer_explanation_level"] == (
        "target_margin_component_sufficiency"
    )

    corroborated = derive_primary_attribution_evidence(
        postblock,
        {
            "fold_scope": dict(PRIMARY_ATTRIBUTION_FOLD_SCOPE),
            "rows": support_rows("attention_o_projection", "target_margin")
            + support_rows("attention_o_projection", "ndcg@10"),
        },
    )[0]
    assert corroborated["strict_transfer_ndcg_component_corroborated"] is True
    assert corroborated["transfer_failure_causal_scope"] == (
        "target_margin_primary_with_statistical_ndcg_corroboration_only"
    )
    assert corroborated["transfer_explanation_level"] == (
        "target_margin_component_with_statistical_ndcg_corroboration"
    )
    assert corroborated[
        "strict_transfer_ndcg_beyond_equivalence_component_corroborated"
    ] is False

    utility_corroborated = derive_primary_attribution_evidence(
        postblock,
        {
            "fold_scope": dict(PRIMARY_ATTRIBUTION_FOLD_SCOPE),
            "rows": support_rows("attention_o_projection", "target_margin")
            + support_rows(
                "attention_o_projection",
                "ndcg@10",
                ci95=[-0.02, -0.01],
            ),
        },
    )[0]
    assert utility_corroborated[
        "strict_transfer_ndcg_beyond_equivalence_component_corroborated"
    ] is True
    assert utility_corroborated["transfer_failure_causal_scope"] == (
        "target_margin_primary_with_utility_relevant_ndcg_corroboration"
    )
    assert utility_corroborated["transfer_explanation_level"] == (
        "target_margin_component_with_utility_relevant_ndcg_corroboration"
    )

    different_residual_nodes = derive_primary_attribution_evidence(
        postblock,
        {
            "fold_scope": dict(PRIMARY_ATTRIBUTION_FOLD_SCOPE),
            "rows": support_rows("post_attention_residual", "target_margin")
            + support_rows("block_output_residual", "ndcg@10"),
        },
    )[0]
    assert different_residual_nodes["derived_primary_component"] == (
        "residual_composition"
    )
    assert different_residual_nodes[
        "strict_transfer_ndcg_component_corroborated"
    ] is False


def test_primary_attribution_strength_distinguishes_gate_stop_from_nonreproduction():
    rows = derive_primary_attribution_evidence(
        {
            "localization": {
                "q2_recranker_generalqwen": {
                    "status": "unresolved",
                    "resolved": False,
                },
                "q3_tallrec_generalqwen": {
                    "status": "fold0_no_negative_transition",
                    "resolved": False,
                },
            }
        },
        {
            "fold_scope": dict(PRIMARY_ATTRIBUTION_FOLD_SCOPE),
            "rows": [],
        },
    )
    by_model = {row["method_id"]: row for row in rows}
    assert by_model["q2_recranker_generalqwen"][
        "derived_evidence_strength"
    ] == "exploratory_only"
    assert by_model["q3_tallrec_generalqwen"][
        "derived_evidence_strength"
    ] == "gate_stopped"
    assert all(
        row["derived_primary_component"] == "unresolved" for row in rows
    )
    assert all(
        row["transfer_explanation_level"] == "unresolved_or_gate_stopped"
        for row in rows
    )


def test_reproduced_transition_without_component_has_second_explanation_level():
    rows = derive_primary_attribution_evidence(
        {
            "localization": {
                "q2_recranker_generalqwen": {
                    "status": "resolved",
                    "resolved": True,
                },
                "q3_tallrec_generalqwen": {
                    "status": "fold0_no_negative_transition",
                    "resolved": False,
                },
            }
        },
        {
            "fold_scope": dict(PRIMARY_ATTRIBUTION_FOLD_SCOPE),
            "rows": [],
        },
    )
    by_model = {row["method_id"]: row for row in rows}
    assert by_model["q2_recranker_generalqwen"]["derived_primary_component"] == (
        "unresolved"
    )
    assert by_model["q2_recranker_generalqwen"][
        "transfer_explanation_level"
    ] == "reproduced_layer_transition_without_unique_component"


def test_primary_attribution_rejects_duplicate_node_contrast_rows():
    node = "attention_o_projection"
    rows = [
        {
            "method_id": "q2_recranker_generalqwen",
            "endpoint": "target_margin",
            "contrast_id": f"{prefix}__{node}",
            "registered_support": True,
            "missing": False,
            "evidence_role": "registered_confirmatory_branch_localization",
        }
        for prefix in (
            "same",
            "same_minus_cross",
            "same_minus_wrong",
            "norm",
            "direction",
            "random",
        )
    ]
    rows.append(dict(rows[0]))
    with pytest.raises(
        DeepDiveReportContractError,
        match="duplicate selected-branch contrast",
    ):
        derive_primary_attribution_evidence(
            {
                "localization": {
                    "q2_recranker_generalqwen": {
                        "status": "resolved",
                        "resolved": True,
                    },
                    "q3_tallrec_generalqwen": {
                        "status": "fold0_no_negative_transition",
                        "resolved": False,
                    },
                }
            },
            {
                "fold_scope": dict(PRIMARY_ATTRIBUTION_FOLD_SCOPE),
                "rows": rows,
            },
        )


def test_primary_attribution_rejects_missing_fold_scope_boundary():
    with pytest.raises(
        DeepDiveReportContractError,
        match="selected-branch fold-scope boundary differs",
    ):
        derive_primary_attribution_evidence(
            {
                "localization": {
                    "q2_recranker_generalqwen": {
                        "status": "resolved",
                        "resolved": True,
                    },
                    "q3_tallrec_generalqwen": {
                        "status": "fold0_no_negative_transition",
                        "resolved": False,
                    },
                }
            },
            {"rows": []},
        )


def _write_result_deliverable(tmp_path, deliverable, payload):
    path = tmp_path / EXPECTED_DELIVERABLES[deliverable]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _registered_rows(*, mean=1.0, ci95=(0.5, 1.5)):
    return [
        {
            "normalized_query_fold": fold,
            "mean": fold_mean,
            "ci95": list(ci95),
        }
        for fold, fold_mean in (("all", mean), ("0", mean * 0.8), ("1", mean * 1.2))
    ]


def _d7_objective_family_rows():
    return [
        {
            "state": state,
            "surface": surface,
            "endpoint": endpoint,
            "conflict_beyond_sesoi": False,
            "bh_q_below_0.05": False,
        }
        for state in D7_OBJECTIVE_FAMILY_STATES
        for surface in D7_OBJECTIVE_FAMILY_SURFACES
        for endpoint in D7_OBJECTIVE_FAMILY_ENDPOINTS
    ]


def _supported_component(decisions, component_id, model_id, evidence):
    row = next(
        value
        for value in decisions["component_evidence_matrix"]
        if value["component_id"] == component_id
    )
    row["status"] = "supported"
    row["model_scope"] = [model_id]
    row["evidence_deliverables"] = [evidence]
    return row


def test_supported_context_component_requires_actual_registered_result(tmp_path):
    method_id = "q2_recranker_generalqwen"
    metrics = {
        "analysis_type": "transformer_deep_dive_d5_contextual_controls",
        "status": "completed",
        "family_rows": [
            {
                "method_id": method_id,
                "condition": "history_attention_null",
                "endpoint": "target_margin",
                "bh_q": 0.20,
            }
        ],
        "results": {
            method_id: {
                "history_attention_null": {
                    "target_margin": {"registered": _registered_rows()}
                }
            }
        },
    }
    _write_result_deliverable(tmp_path, "d5_context", metrics)
    decisions = _valid_decisions()
    _supported_component(decisions, "history_routing", method_id, "d5_context")
    with pytest.raises(
        DeepDiveReportContractError, match="differs from registered result evidence"
    ):
        _validate_supported_components_against_evidence(tmp_path, decisions)
    metrics["family_rows"][0]["bh_q"] = 0.01
    _write_result_deliverable(tmp_path, "d5_context", metrics)
    _validate_supported_components_against_evidence(tmp_path, decisions)


def test_attention_value_transport_cannot_borrow_logits_mask_result(tmp_path):
    method_id = "q2_recranker_generalqwen"
    metrics = {
        "analysis_type": "transformer_deep_dive_d3_attention_edges",
        "status": "completed",
        "family_rows": [
            {
                "method_id": method_id,
                "block_zero_based": 13,
                "condition": "history_logits_mask",
                "endpoint": "target_margin",
                "bh_q": 0.01,
            }
        ],
        "results": {
            method_id: {
                "13": {
                    "history_logits_mask": {
                        "target_margin": {"registered": _registered_rows()}
                    }
                }
            }
        },
    }
    _write_result_deliverable(tmp_path, "d3_attention_edges", metrics)
    decisions = _valid_decisions()
    _supported_component(
        decisions,
        "attention_value_transport",
        method_id,
        "d3_attention_edges",
    )
    with pytest.raises(
        DeepDiveReportContractError, match="differs from registered result evidence"
    ):
        _validate_supported_components_against_evidence(tmp_path, decisions)
    metrics["family_rows"].append(
        {
            "method_id": method_id,
            "block_zero_based": 13,
            "condition": "history_value_edge_zero",
            "endpoint": "target_margin",
            "bh_q": 0.01,
        }
    )
    metrics["results"][method_id]["13"]["history_value_edge_zero"] = {
        "target_margin": {"registered": _registered_rows()}
    }
    _write_result_deliverable(tmp_path, "d3_attention_edges", metrics)
    _validate_supported_components_against_evidence(tmp_path, decisions)


def test_attention_query_key_routing_cannot_borrow_value_transport_result(tmp_path):
    method_id = "q2_recranker_generalqwen"
    metrics = {
        "analysis_type": "transformer_deep_dive_d3_attention_edges",
        "status": "completed",
        "family_rows": [
            {
                "method_id": method_id,
                "block_zero_based": 13,
                "condition": "history_value_edge_zero",
                "endpoint": "target_margin",
                "bh_q": 0.01,
            }
        ],
        "results": {
            method_id: {
                "13": {
                    "history_value_edge_zero": {
                        "target_margin": {"registered": _registered_rows()}
                    }
                }
            }
        },
    }
    _write_result_deliverable(tmp_path, "d3_attention_edges", metrics)
    decisions = _valid_decisions()
    _supported_component(
        decisions,
        "attention_query_key_routing",
        method_id,
        "d3_attention_edges",
    )
    with pytest.raises(
        DeepDiveReportContractError, match="differs from registered result evidence"
    ):
        _validate_supported_components_against_evidence(tmp_path, decisions)
    metrics["family_rows"].append(
        {
            "method_id": method_id,
            "block_zero_based": 13,
            "condition": "history_logits_mask",
            "endpoint": "target_margin",
            "bh_q": 0.01,
        }
    )
    metrics["results"][method_id]["13"]["history_logits_mask"] = {
        "target_margin": {"registered": _registered_rows()}
    }
    _write_result_deliverable(tmp_path, "d3_attention_edges", metrics)
    _validate_supported_components_against_evidence(tmp_path, decisions)


def test_attention_output_cannot_borrow_upstream_edge_causality():
    decisions = _valid_decisions()
    _supported_component(
        decisions,
        "attention_output",
        "q2_recranker_generalqwen",
        "d3_attention_edges",
    )
    with pytest.raises(
        DeepDiveReportContractError, match="causal evidence: attention_output"
    ):
        validate_deep_dive_report_decisions(decisions)


def test_rope_support_requires_active_band_and_compression_contrast(tmp_path):
    method_id = "q2_recranker_generalqwen"
    active = _registered_rows(mean=0.006, ci95=(-0.004, 0.012))
    metrics = {
        "analysis_type": "transformer_deep_dive_d5_rope",
        "status": "completed",
        "position_support_gate": {
            "active_contrast": "compression_minus_baseline",
            "active_endpoint": "ndcg@10",
            "active_ci95_equivalence_band": [-0.005, 0.005],
            "requires_compression_minus_expansion_bh_q_below_alpha_0p05": True,
            "requires_all_fold0_fold1_same_nonzero_direction": True,
            "active_contrast_is_confirmatory_family_member": False,
        },
        "family_rows": [
            {
                "method_id": method_id,
                "block_zero_based": 13,
                "contrast": "readout_q",
                "endpoint": "ndcg@10",
                "bh_q": 0.01,
            }
        ],
        "results": {
            method_id: {
                "13": {
                    "readout_q": {
                        "ndcg@10": {
                            "registered_compression_minus_expansion": (
                                _registered_rows()
                            ),
                            "registered_compression_minus_baseline_support_gate": active,
                        }
                    }
                }
            }
        },
    }
    _write_result_deliverable(tmp_path, "d5_rope", metrics)
    decisions = _valid_decisions()
    _supported_component(
        decisions, "positional_encoding_rope", method_id, "d5_rope"
    )
    with pytest.raises(
        DeepDiveReportContractError, match="differs from registered result evidence"
    ):
        _validate_supported_components_against_evidence(tmp_path, decisions)
    for row in active:
        row["ci95"] = [0.006, 0.015]
    _write_result_deliverable(tmp_path, "d5_rope", metrics)
    _validate_supported_components_against_evidence(tmp_path, decisions)


def test_native_readout_support_requires_same_user_specificity(tmp_path):
    method_id = "q2_recranker_generalqwen"
    metrics = {
        "analysis_type": "transformer_deep_dive_d6_q2_native_readout",
        "status": "completed",
        "method_id": method_id,
        "family_rows": [
            {
                "node": "final_rmsnorm_output",
                "comparison": comparison,
                "endpoint": "target_margin",
                "bh_q": q_value,
            }
            for comparison, q_value in (
                ("same_minus_null", 0.01),
                ("same_minus_cross", 0.20),
            )
        ],
        "results": {
            "final_rmsnorm_output": {
                comparison: {"target_margin": _registered_rows()}
                for comparison in ("same_minus_null", "same_minus_cross")
            }
        },
    }
    _write_result_deliverable(tmp_path, "d6_q2_native_readout", metrics)
    decisions = _valid_decisions()
    _supported_component(
        decisions, "native_readout", method_id, "d6_q2_native_readout"
    )
    with pytest.raises(
        DeepDiveReportContractError, match="differs from registered result evidence"
    ):
        _validate_supported_components_against_evidence(tmp_path, decisions)
    metrics["family_rows"][1]["bh_q"] = 0.01
    _write_result_deliverable(tmp_path, "d6_q2_native_readout", metrics)
    _validate_supported_components_against_evidence(tmp_path, decisions)


def test_native_readout_cannot_join_specificity_across_endpoints(tmp_path):
    method_id = "q2_recranker_generalqwen"
    metrics = {
        "analysis_type": "transformer_deep_dive_d6_q2_native_readout",
        "status": "completed",
        "method_id": method_id,
        "family_rows": [
            {
                "node": "final_rmsnorm_output",
                "comparison": "same_minus_null",
                "endpoint": "target_margin",
                "bh_q": 0.01,
            },
            {
                "node": "final_rmsnorm_output",
                "comparison": "same_minus_cross",
                "endpoint": "ndcg@10",
                "bh_q": 0.01,
            },
        ],
        "results": {
            "final_rmsnorm_output": {
                "same_minus_null": {"target_margin": _registered_rows()},
                "same_minus_cross": {"ndcg@10": _registered_rows()},
            }
        },
    }
    _write_result_deliverable(tmp_path, "d6_q2_native_readout", metrics)
    decisions = _valid_decisions()
    _supported_component(
        decisions, "native_readout", method_id, "d6_q2_native_readout"
    )
    with pytest.raises(
        DeepDiveReportContractError, match="differs from registered result evidence"
    ):
        _validate_supported_components_against_evidence(tmp_path, decisions)


def test_score_nullspace_support_uses_exact_algebra_not_readout_significance(tmp_path):
    method_id = "q2_recranker_generalqwen"
    metrics = {
        "analysis_type": "transformer_deep_dive_d6_q2_native_readout",
        "status": "completed",
        "method_id": method_id,
        "readout_decomposition": {
            "algebra": {
                "score_identity": "score_ij = common_i + relative_ij",
                "common_definition": "mean_j(score_ij)",
                "relative_definition": "score_ij - common_i",
                "maximum_recomposition_abs_error": 0.0,
                "maximum_relative_sum_abs_error": 0.0,
            },
            "qrels_read": False,
            "confirmatory_family_membership": False,
        },
    }
    _write_result_deliverable(tmp_path, "d6_q2_native_readout", metrics)
    decisions = _valid_decisions()
    _supported_component(
        decisions,
        "score_calibration_nullspace",
        method_id,
        "d6_q2_native_readout",
    )
    _validate_supported_components_against_evidence(tmp_path, decisions)
    metrics["readout_decomposition"]["algebra"][
        "maximum_recomposition_abs_error"
    ] = 1.0e-4
    _write_result_deliverable(tmp_path, "d6_q2_native_readout", metrics)
    with pytest.raises(
        DeepDiveReportContractError, match="differs from registered result evidence"
    ):
        _validate_supported_components_against_evidence(tmp_path, decisions)


def test_loss_gradient_support_requires_registered_sesoi_and_fdr(tmp_path):
    method_id = "q2_recranker_generalqwen"
    metrics = {
        "analysis_type": "transformer_deep_dive_d7_q2_objective_conflict",
        "status": "completed",
        "family_rows": _d7_objective_family_rows(),
    }
    active = next(
        row
        for row in metrics["family_rows"]
        if row["endpoint"] == "ranknet_listnet_cosine"
    )
    active["conflict_beyond_sesoi"] = True
    _write_result_deliverable(tmp_path, "d7_q2_objective", metrics)
    decisions = _valid_decisions()
    _supported_component(
        decisions, "loss_gradient", method_id, "d7_q2_objective"
    )
    with pytest.raises(
        DeepDiveReportContractError, match="differs from registered result evidence"
    ):
        _validate_supported_components_against_evidence(tmp_path, decisions)
    active["bh_q_below_0.05"] = True
    _write_result_deliverable(tmp_path, "d7_q2_objective", metrics)
    _validate_supported_components_against_evidence(tmp_path, decisions)
    metrics["family_rows"].append(dict(active))
    _write_result_deliverable(tmp_path, "d7_q2_objective", metrics)
    with pytest.raises(
        DeepDiveReportContractError,
        match="duplicate D7 objective-conflict family key",
    ):
        _validate_supported_components_against_evidence(tmp_path, decisions)
    metrics["family_rows"] = _d7_objective_family_rows()
    metrics["family_rows"][0]["surface"] = "unregistered_surface"
    _write_result_deliverable(tmp_path, "d7_q2_objective", metrics)
    with pytest.raises(
        DeepDiveReportContractError,
        match="family key coverage differs",
    ):
        _validate_supported_components_against_evidence(tmp_path, decisions)


def test_descriptive_breadth_readout_cannot_promote_native_readout_support():
    decisions = _valid_decisions()
    _supported_component(
        decisions,
        "native_readout",
        "q0_qwen3_reranker_06b",
        "d6_q0_q1_readouts",
    )
    with pytest.raises(
        DeepDiveReportContractError, match="causal evidence: native_readout"
    ):
        validate_deep_dive_report_decisions(decisions)


def test_native_readout_cannot_be_relabelled_as_normalization_support():
    decisions = _valid_decisions()
    _supported_component(
        decisions,
        "normalization",
        "q2_recranker_generalqwen",
        "d6_q2_native_readout",
    )
    with pytest.raises(
        DeepDiveReportContractError, match="causal evidence: normalization"
    ):
        validate_deep_dive_report_decisions(decisions)


def test_q0_block_output_patch_cannot_promote_residual_composition():
    decisions = _valid_decisions()
    _supported_component(
        decisions,
        "residual_composition",
        "q0_qwen3_reranker_06b",
        "d6_q0_q1_branches",
    )
    with pytest.raises(
        DeepDiveReportContractError, match="causal evidence: residual_composition"
    ):
        validate_deep_dive_report_decisions(decisions)
