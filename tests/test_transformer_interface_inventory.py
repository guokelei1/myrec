from __future__ import annotations

from pathlib import Path

import pytest

from myrec.mechanism.deep_dive_closeout_audit import EXPECTED_DELIVERABLES
from myrec.mechanism.deep_dive_evidence_topology import MODEL_IDS
from myrec.mechanism.deep_dive_report_contract import COMPONENT_IDS
from myrec.mechanism.supplemental_evidence_registry import (
    EXPECTED_SUPPLEMENT_IDS,
    audit_supplemental_evidence_registry,
)
from myrec.mechanism.transformer_instrumentation import BLOCK_NODE_IDS, FINAL_NODE_IDS
from myrec.mechanism.transformer_interface_inventory import (
    OPERATOR_CAUSAL_DEBT_CONTRACT,
    TRANSFORMER_INTERFACE_INVENTORY,
    build_transformer_interface_coverage,
)


ROOT = Path(__file__).resolve().parents[1]


def _supplement_scopes() -> dict[str, set[str]]:
    return {evidence_id: set(MODEL_IDS) for evidence_id in EXPECTED_SUPPLEMENT_IDS}


def _supplement_component_scopes() -> dict[str, set[str]]:
    return {
        evidence_id: set(COMPONENT_IDS) for evidence_id in EXPECTED_SUPPLEMENT_IDS
    }


def test_exact_transformer_interface_inventory_is_exhaustive_and_effect_blind() -> None:
    coverage = build_transformer_interface_coverage(
        completed_formal=set(EXPECTED_DELIVERABLES),
        completed_supplements=set(EXPECTED_SUPPLEMENT_IDS),
        supplement_model_scopes=_supplement_scopes(),
        supplement_component_scopes=_supplement_component_scopes(),
    )

    assert coverage["interface_count"] == len(TRANSFORMER_INTERFACE_INVENTORY) == 63
    assert coverage["interfaces_with_any_completed_evidence"] == 60
    assert coverage["interfaces_with_all_registered_evidence_completed"] == 60
    assert coverage["interfaces_with_registered_causal_role_evidence"] == 25
    assert coverage["interfaces_with_completed_causal_role_evidence"] == 25
    assert coverage["operator_causal_debt_count"] == 38
    assert coverage["operator_causal_debt_class_counts"] == {
        "inference_operator": 18,
        "training_mechanism": 20,
    }
    assert {
        row["interface_id"] for row in coverage["operator_causal_debt"]
    } == set(OPERATOR_CAUSAL_DEBT_CONTRACT)
    assert all(
        row["current_stage_disposition"]
        == "unresolved_no_operator_causal_claim"
        and row["active_experiment_authorized"] is False
        and row["can_rank_architecture_from_current_evidence"] is False
        and row["scientific_support_inferred_from_completion"] is False
        and row["smallest_falsification_gate"]
        for row in coverage["operator_causal_debt"]
    )
    assert coverage["new_experiment_family_authorized_by_debt_ledger"] is False
    assert coverage["operator_causal_debt_is_lower_bound"] is True
    assert coverage["operator_attribution_inferred_for_other_interfaces"] is False
    assert coverage[
        "operator_attribution_inferred_from_artifact_availability_count"
    ] == 0
    assert coverage[
        "operator_attribution_unresolved_from_artifact_availability_count"
    ] == 63
    assert len(
        coverage[
            "interfaces_with_functional_causal_role_but_no_operator_attribution_inferred"
        ]
    ) == 25
    assert coverage["functional_causal_role_is_operator_attribution"] is False
    assert coverage["registered_claim_ceiling_counts"] == {
        "none": 3,
        "M": 0,
        "D": 35,
        "S": 21,
        "N": 1,
        "G": 3,
    }
    assert coverage["completed_artifact_claim_ceiling_counts"] == coverage[
        "registered_claim_ceiling_counts"
    ]
    assert coverage["claim_ceilings_are_artifact_availability_only"] is True
    assert coverage["actual_scientific_evidence_levels_inferred"] is False
    layer_coverage = coverage["system_layer_coverage"]
    assert set(layer_coverage) == {
        "input",
        "representation",
        "routing",
        "readout",
        "training",
    }
    assert sum(row["interface_count"] for row in layer_coverage.values()) == 63
    assert sum(
        row["operator_causal_debt_count"] for row in layer_coverage.values()
    ) == 38
    assert layer_coverage["input"]["interface_count"] == 2
    assert layer_coverage["input"]["operator_causal_debt_count"] == 1
    assert layer_coverage["readout"]["interface_count"] == 9
    assert layer_coverage["representation"]["interface_count"] == 14
    assert layer_coverage["routing"]["interface_count"] == 18
    assert layer_coverage["routing"]["operator_causal_debt_count"] == 8
    assert layer_coverage["training"]["interface_count"] == 20
    assert layer_coverage["training"]["operator_causal_debt_count"] == 20
    assert coverage["direct_interface_evidence_count"] == 31
    assert coverage["cross_interface_evidence_count"] == 9
    assert coverage["registered_evidence_count"] == 40
    assert coverage[
        "all_registered_evidence_has_exact_or_cross_interface_disposition"
    ] is True
    assert {row["evidence_id"] for row in coverage["cross_interface_evidence"]} == {
        "d1_activation_anisotropy",
        "d1_candidate_block_flow",
        "d1_candidate_residual_geometry",
        "d1_preference_subspace_geometry",
        "d1_query_causal_floor",
        "d2_q3_native_gate",
        "d6_q0_q1_branches",
        "d6_q0_trajectory",
        "d7_objective_common_nullspace",
    }
    assert all(
        row["scientific_support_inferred_from_completion"] is False
        for row in coverage["cross_interface_evidence"]
    )
    assert coverage["scientific_support_inferred_from_completion"] is False
    by_id = {row["interface_id"]: row for row in coverage["interfaces"]}
    assert set(by_id) == {
        "serialization_tokenization",
        "token_embedding_lookup",
        "tied_lm_head_rows",
        "autoregressive_causal_attention_mask",
        "kv_cache_phase_boundary",
        "block_input_residual",
        "input_rmsnorm_output",
        "input_rmsnorm_variance_rescale_and_gain",
            "q_pre_norm",
            "q3_q_lora_scaled_adapter_injection",
        "k_pre_norm",
        "q_post_norm_pre_rope",
        "k_post_norm_pre_rope",
        "q_head_rmsnorm_variance_rescale_and_gain",
        "k_head_rmsnorm_variance_rescale_and_gain",
        "q_post_rope",
        "k_post_rope",
            "v_projection",
            "q3_v_lora_scaled_adapter_injection",
        "attention_scaled_qk_logits",
        "attention_softmax_edge_weights",
        "attention_head_output_pre_o",
        "gqa_query_to_kv_grouping",
        "attention_o_projection",
        "post_attention_residual",
        "attention_residual_addition",
        "post_attention_rmsnorm_output",
        "post_attention_rmsnorm_variance_rescale_and_gain",
        "mlp_gate_projection",
        "mlp_up_projection",
        "mlp_silu_gate",
        "mlp_swiglu_product",
        "mlp_down_projection",
        "block_output_residual",
        "mlp_residual_addition",
        "final_rmsnorm_input",
        "final_rmsnorm_output",
        "final_rmsnorm_variance_rescale_and_gain",
        "q_head_rmsnorm_variance_rescale_and_gain",
        "k_head_rmsnorm_variance_rescale_and_gain",
        "candidate_readout_positions",
        "q0_next_token_yes_no_logit_difference",
        "q1_candidate_response_mean_log_likelihood",
        "q2_next_token_yes_no_logit_difference",
        "q3_two_path_mean_log_likelihood_difference",
        "q0_pointwise_bce_loss",
        "q1_normalized_response_nll",
        "q2_pairwise_ranknet_loss",
        "q2_listwise_listnet_loss",
            "q3_alignment_nll_loss",
            "bfloat16_autocast_training_forward",
            "nonreentrant_gradient_checkpoint_recomputation",
            "q3_input_activation_requires_grad_bridge",
            "q3_fp32_lora_bf16_base_cast_boundary",
            "gradient_accumulation_and_global_clip",
        "adam_moment_preconditioned_direction",
        "decoupled_weight_decay_term",
            "learning_rate_scaled_effective_parameter_delta",
            "lora_training_input_dropout",
            "lora_q_low_rank_a_factor",
        "lora_q_low_rank_b_factor",
        "lora_q_effective_delta_weight",
        "lora_v_low_rank_a_factor",
        "lora_v_low_rank_b_factor",
        "lora_v_effective_delta_weight",
    }
    assert set(BLOCK_NODE_IDS) | set(FINAL_NODE_IDS) <= set(by_id)
    assert {row["system_layer"] for row in coverage["interfaces"]} == {
        "input",
        "representation",
        "routing",
        "readout",
        "training",
    }
    assert {
        component_id
        for row in coverage["interfaces"]
        for component_id in row["component_ids"]
    } == {
        "serialization_tokenization",
        "token_embedding",
        "positional_encoding_rope",
        "attention_query_key_routing",
        "attention_value_transport",
        "attention_output",
        "mlp_feature_formation",
        "mlp_output",
        "residual_composition",
        "normalization",
        "layerwise_representation",
        "history_routing",
        "candidate_conditioned_interaction",
        "native_readout",
        "score_calibration_nullspace",
        "loss_gradient",
        "optimizer_effective_update",
        "lora_parameterization",
    }
    assert by_id["mlp_silu_gate"]["causal_role_registered"] is False
    assert by_id["token_embedding_lookup"]["causal_role_registered"] is False
    assert by_id["tied_lm_head_rows"]["causal_role_registered"] is True
    assert by_id["token_embedding_lookup"]["registered_claim_ceiling"] == "D"
    assert by_id["tied_lm_head_rows"]["registered_claim_ceiling"] == "S"
    assert by_id["attention_o_projection"]["registered_claim_ceiling"] == "G"
    assert by_id["block_output_residual"]["registered_claim_ceiling"] == "N"
    assert by_id["q0_pointwise_bce_loss"]["registered_claim_ceiling"] == "none"
    assert by_id["q1_normalized_response_nll"]["registered_claim_ceiling"] == "none"
    assert by_id["q2_pairwise_ranknet_loss"]["registered_claim_ceiling"] == "D"
    assert by_id["q0_pointwise_bce_loss"]["implementation_model_scope"] == [
        MODEL_IDS[0]
    ]
    assert by_id["q1_normalized_response_nll"]["implementation_model_scope"] == [
        MODEL_IDS[1]
    ]
    assert by_id["q0_next_token_yes_no_logit_difference"]["implementation_model_scope"] == [MODEL_IDS[0]]
    assert by_id["q1_candidate_response_mean_log_likelihood"]["implementation_model_scope"] == [MODEL_IDS[1]]
    assert by_id["q2_next_token_yes_no_logit_difference"]["implementation_model_scope"] == [MODEL_IDS[2]]
    assert by_id["q3_two_path_mean_log_likelihood_difference"]["implementation_model_scope"] == [MODEL_IDS[3]]
    assert by_id["final_rmsnorm_input"]["model_scope_registered"] == list(MODEL_IDS)
    assert by_id["final_rmsnorm_output"]["model_scope_registered"] == list(MODEL_IDS)
    for interface_id in (
        "lora_q_low_rank_a_factor",
        "lora_q_low_rank_b_factor",
        "lora_q_effective_delta_weight",
        "lora_v_low_rank_a_factor",
        "lora_v_low_rank_b_factor",
        "lora_v_effective_delta_weight",
    ):
        assert by_id[interface_id]["implementation_model_scope"] == [MODEL_IDS[3]]
    assert all(
        row["claim_ceiling_is_artifact_availability_only"] is True
        and row["actual_scientific_evidence_level_inferred"] is False
        and row["operator_attribution_inferred_from_artifact_availability"] is False
        for row in coverage["interfaces"]
    )
    assert (
        by_id["attention_o_projection"][
            "operator_attribution_status_from_artifact_availability"
        ]
        == "not_inferred_functional_causal_role_available"
    )
    assert (
        by_id["attention_scaled_qk_logits"][
            "operator_attribution_status_from_artifact_availability"
        ]
        == "not_inferred_no_functional_causal_role_artifact"
    )
    assert by_id["attention_o_projection"]["causal_role_registered"] is True
    assert by_id["attention_o_projection"]["causal_role_completed"] is True
    assert coverage["interfaces_without_any_completed_evidence"] == [
        "nonreentrant_gradient_checkpoint_recomputation",
        "q0_pointwise_bce_loss",
        "q1_normalized_response_nll",
    ]
    assert "mlp_silu_gate" in coverage[
        "interfaces_without_registered_causal_role_evidence"
    ]
    for interface_id in (
        "input_rmsnorm_variance_rescale_and_gain",
        "post_attention_rmsnorm_variance_rescale_and_gain",
        "final_rmsnorm_variance_rescale_and_gain",
        "attention_residual_addition",
        "mlp_residual_addition",
    ):
        assert by_id[interface_id]["causal_role_registered"] is False
        assert interface_id in coverage[
            "interfaces_without_registered_causal_role_evidence"
        ]
    assert "kv_cache_phase_boundary" in coverage[
        "interfaces_without_registered_causal_role_evidence"
    ]


def test_interface_inventory_keeps_pending_and_causal_completion_separate() -> None:
    coverage = build_transformer_interface_coverage(
        completed_formal={"d1_representation", "d3_attention_heads"},
        completed_supplements={"d3_attention_pattern_synthesis"},
        supplement_model_scopes=_supplement_scopes(),
        supplement_component_scopes=_supplement_component_scopes(),
    )
    by_id = {row["interface_id"]: row for row in coverage["interfaces"]}

    assert by_id["attention_softmax_edge_weights"]["any_evidence_completed"] is True
    assert by_id["attention_softmax_edge_weights"]["causal_role_registered"] is True
    assert by_id["attention_softmax_edge_weights"]["causal_role_completed"] is False
    assert by_id["mlp_silu_gate"]["any_evidence_completed"] is False
    assert by_id["mlp_silu_gate"]["scientific_support_inferred_from_completion"] is False
    assert "attention_softmax_edge_weights" in coverage[
        "interfaces_with_registered_but_pending_causal_role_evidence"
    ]
    assert sum(coverage["completed_artifact_claim_ceiling_counts"].values()) == 63
    assert coverage["completed_artifact_claim_ceiling_counts"]["none"] > 0


def test_model_specific_native_scores_do_not_borrow_q2_readout_completion() -> None:
    coverage = build_transformer_interface_coverage(
        completed_formal={"d6_q2_native_readout"},
        completed_supplements=set(),
        supplement_model_scopes=_supplement_scopes(),
        supplement_component_scopes=_supplement_component_scopes(),
    )
    by_id = {row["interface_id"]: row for row in coverage["interfaces"]}

    q0 = by_id["q0_next_token_yes_no_logit_difference"]
    q1 = by_id["q1_candidate_response_mean_log_likelihood"]
    q2 = by_id["q2_next_token_yes_no_logit_difference"]
    q3 = by_id["q3_two_path_mean_log_likelihood_difference"]
    assert q2["causal_role_completed"] is True
    assert q2["completed_artifact_claim_ceiling"] == "S"
    assert all(
        row["causal_role_completed"] is False
        and row["completed_artifact_claim_ceiling"] == "none"
        for row in (q0, q1, q3)
    )


def test_interface_inventory_rejects_incomplete_supplement_scope_map() -> None:
    scopes = _supplement_scopes()
    scopes.pop(next(iter(scopes)))
    with pytest.raises(ValueError, match="supplement model-scope coverage drift"):
        build_transformer_interface_coverage(
            completed_formal=set(),
            completed_supplements=set(),
            supplement_model_scopes=scopes,
            supplement_component_scopes=_supplement_component_scopes(),
        )


def test_real_frozen_registry_components_semantically_bind_every_interface() -> None:
    registry = audit_supplemental_evidence_registry(ROOT)
    assert registry["status"] in {"pending", "completed"}
    rows = {row["evidence_id"]: row for row in registry["entries"]}
    coverage = build_transformer_interface_coverage(
        completed_formal=set(),
        completed_supplements=set(),
        supplement_model_scopes={
            evidence_id: set(row["model_scope"])
            for evidence_id, row in rows.items()
        },
        supplement_component_scopes={
            evidence_id: set(row["components"])
            for evidence_id, row in rows.items()
        },
    )
    assert coverage["interface_count"] == 63
    assert coverage["registered_evidence_count"] == 40
    assert coverage["scientific_support_inferred_from_completion"] is False
