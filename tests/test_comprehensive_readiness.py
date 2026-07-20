from __future__ import annotations

from pathlib import Path

from myrec.mechanism.comprehensive_readiness import (
    build_comprehensive_readiness,
)


ROOT = Path(__file__).resolve().parents[1]


def test_comprehensive_readiness_keeps_formal_supplement_and_d2_progress_separate():
    result = build_comprehensive_readiness(ROOT)
    assert result["status"] in {"pending", "completed"}
    assert result["formal"]["registered"] == 19
    assert result["supplements"]["registered"] == 21
    assert result["combined_artifact_completion"]["registered"] == 40
    assert "not compute progress" in result["combined_artifact_completion"][
        "interpretation"
    ]
    assert result["d2_causal_core"]["maximum_total_scientific_bundles"] == 62
    assert result["component_count"] == 18
    assert result["components_with_completed_causal_role_artifact"] <= result[
        "components_with_any_completed_artifact"
    ]
    assert result["components_with_completed_q2_q3_causal_role_artifacts"] <= result[
        "components_with_completed_causal_role_artifact"
    ]
    assert set(result["components_without_registered_causal_role_artifact"]) == {
        "token_embedding",
        "mlp_feature_formation",
        "optimizer_effective_update",
        "lora_parameterization",
    }
    assert "descriptive geometry" in result["component_coverage_interpretation"]
    interfaces = result["transformer_internal_interface_coverage"]
    assert interfaces["interface_count"] == 63
    assert len(interfaces["interfaces"]) == 63
    assert len({row["interface_id"] for row in interfaces["interfaces"]}) == 63
    assert interfaces["interfaces_with_all_registered_evidence_completed"] <= interfaces[
        "interfaces_with_any_completed_evidence"
    ]
    assert interfaces["interfaces_with_completed_causal_role_evidence"] <= interfaces[
        "interfaces_with_registered_causal_role_evidence"
    ]
    assert interfaces["scientific_support_inferred_from_completion"] is False
    assert interfaces["direct_interface_evidence_count"] == 31
    assert interfaces["cross_interface_evidence_count"] == 9
    assert interfaces["registered_evidence_count"] == 40
    assert interfaces[
        "all_registered_evidence_has_exact_or_cross_interface_disposition"
    ] is True
    assert interfaces["registered_claim_ceiling_counts"] == {
        "none": 3,
        "M": 0,
        "D": 35,
        "S": 21,
        "N": 1,
        "G": 3,
    }
    assert sum(interfaces["completed_artifact_claim_ceiling_counts"].values()) == 63
    assert interfaces["claim_ceilings_are_artifact_availability_only"] is True
    assert interfaces["actual_scientific_evidence_levels_inferred"] is False
    architecture = result["frozen_model_architecture_audit"]
    assert architecture["status"] == "completed"
    assert architecture["failures"] == []
    assert architecture["frozen_topology"]["num_hidden_layers"] == 28
    assert architecture["frozen_topology"]["num_attention_heads"] == 16
    assert architecture["frozen_topology"]["num_key_value_heads"] == 8
    assert architecture["config_backed_interface_count"] == 53
    assert architecture["config_backed_interfaces_present_in_inventory"] is True
    assert architecture["dynamic_runtime_or_source_backed_interface_count"] == 10
    assert architecture["implementation_provenance_covered_interface_count"] == 63
    assert architecture["forward_primitive_count"] == 40
    assert architecture["forward_inference_interface_count"] == 43
    assert architecture["forward_mapped_interface_count"] == 43
    assert architecture["forward_missing_interface_ids"] == []
    assert architecture["forward_extraneous_interface_ids"] == []
    assert architecture["forward_training_interface_count"] == 20
    assert architecture["forward_primitive_interface_coverage_complete"] is True
    assert architecture["forward_source_binding_count"] == 11
    assert architecture["forward_peft_version"] == "0.19.1"
    assert architecture[
        "forward_source_environment_is_frozen_checkpoint_environment"
    ] is True
    assert architecture["inactive_architecture_path_count"] == 9
    assert architecture["all_inactive_architecture_paths_verified"] is True
    assert architecture["operator_attribution_inferred_from_forward_coverage"] is False
    assert architecture["training_primitive_count"] == 27
    assert architecture["training_exact_interface_count"] == 20
    assert architecture["training_mapped_interface_count"] == 20
    assert architecture["training_missing_interface_ids"] == []
    assert architecture["training_extraneous_interface_ids"] == []
    assert architecture["training_primitive_interface_coverage_complete"] is True
    assert architecture["training_source_binding_count"] == 12
    assert architecture["training_artifact_binding_count"] == 3
    assert architecture["training_peft_version"] == "0.19.1"
    assert architecture["training_update_graph_failures"] == []
    assert architecture[
        "training_coverage_is_single_step_semantic_primitive_census"
    ] is True
    assert architecture["training_coverage_is_multiseed_causal_attribution"] is False
    assert architecture[
        "operator_attribution_inferred_from_training_coverage"
    ] is False
    assert architecture["inactive_training_path_count"] == 7
    assert architecture["all_inactive_training_paths_verified"] is True
    assert architecture["runtime_identity_smoke_count"] == 4
    assert architecture["runtime_hook_node_count"] == 18
    assert architecture["runtime_identity_and_recomposition_validated"] is True
    assert architecture["superseded_runtime_lineage_count"] == 1
    assert architecture["superseded_runtime_lineage_retained"] is True
    assert architecture["frozen_base_artifact_count"] == 2
    assert architecture["frozen_model_pathway_count"] == 4
    assert set(architecture["model_pathways"]) == {
        "q0_qwen3_reranker_06b",
        "q1_instructrec_generalqwen",
        "q2_recranker_generalqwen",
        "q3_tallrec_generalqwen",
    }
    assert architecture["scientific_effect_values_read"] is False
    assert result["readiness_gates"][
        "frozen_model_architecture_audit_completed"
    ] is True
    assert {
        "attention_scaled_qk_logits",
        "attention_softmax_edge_weights",
        "autoregressive_causal_attention_mask",
        "gqa_query_to_kv_grouping",
        "mlp_silu_gate",
        "kv_cache_phase_boundary",
        "final_rmsnorm_output",
    }.issubset(row["interface_id"] for row in interfaces["interfaces"])
    assert result["mechanical_nonresults"]["retained"] == len(
        result["mechanical_nonresults"]["run_ids"]
    )
    assert len(result["mechanical_nonresults"]["run_ids"]) == len(
        set(result["mechanical_nonresults"]["run_ids"])
    )
    assert "neither scientific null" in result["mechanical_nonresults"][
        "interpretation"
    ]
    assert result["readiness_gates"]["producer_topology_complete"] is True
    assert result["producer_topology"] == {
        "status": "completed",
        "formal_covered": 19,
        "formal_registered": 19,
        "supplements_covered": 21,
        "supplements_registered": 21,
        "queued_or_watched": 18,
        "failures": [],
        "scientific_effect_values_read": False,
    }
    assert result["effect_values_used_for_readiness"] is False
    assert result["scientific_support_inferred_from_completion"] is False
    assert result["qrels_files_opened_by_this_readiness"] is False


def test_every_component_retains_separate_formal_and_supplement_debt():
    result = build_comprehensive_readiness(ROOT)
    rows = result["component_artifact_coverage"]
    assert len(rows) == 18
    assert {row["component_id"] for row in rows} == {
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
    assert all(
        row["scientific_support_inferred_from_completion"] is False
        for row in rows
    )
    mlp = next(row for row in rows if row["component_id"] == "mlp_feature_formation")
    assert "d4_mlp_feature_formation_extension" in mlp["supplements_registered"]
    assert mlp["causal_role_artifact_registered"] is False
    assert mlp["causal_role_artifact_completed"] is False

    attention = next(row for row in rows if row["component_id"] == "attention_output")
    assert "d2_selected_branches" in attention["causal_role_formal_registered"]
    assert {
        "component_state_reverse_necessity_v2",
        "component_functional_design_gate_synthesis",
    }.issubset(attention["causal_role_supplements_registered"])
    assert attention["causal_role_artifact_registered"] is True
    assert attention["q2_q3_causal_role_artifacts_registered"] is True
    assert "requires its registered sign" in attention["artifact_role_interpretation"]


def test_causal_role_coverage_never_borrows_descriptive_artifacts():
    result = build_comprehensive_readiness(ROOT)
    rows = {
        row["component_id"]: row for row in result["component_artifact_coverage"]
    }
    for row in rows.values():
        assert set(row["causal_role_formal_completed"]).issubset(
            row["causal_role_formal_registered"]
        )
        assert set(row["causal_role_supplements_completed"]).issubset(
            row["causal_role_supplements_registered"]
        )
        assert set(row["causal_role_model_scope_completed"]).issubset(
            row["causal_role_model_scope_registered"]
        )
        assert row["scientific_support_inferred_from_completion"] is False

    for component_id in (
        "token_embedding",
        "mlp_feature_formation",
        "optimizer_effective_update",
        "lora_parameterization",
    ):
        assert rows[component_id]["causal_role_artifact_registered"] is False

    assert rows["native_readout"]["causal_role_model_scope_completed"] == [
        "q2_recranker_generalqwen"
    ]
    assert rows["native_readout"]["q2_q3_causal_role_artifacts_completed"] is False
