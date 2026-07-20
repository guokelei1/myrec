from __future__ import annotations

import json
from pathlib import Path

import myrec.mechanism.frozen_model_architecture_audit as architecture_audit_module
from myrec.mechanism.frozen_model_architecture_audit import (
    FORWARD_PRIMITIVE_CONTRACTS,
    IDENTITY_SMOKES,
    Q3_ADAPTER_CONFIG,
    SUPERSEDED_IDENTITY_SMOKES,
    TRAINING_PRIMITIVE_CONTRACTS,
    audit_frozen_model_architecture,
)


ROOT = Path(__file__).resolve().parents[1]


def test_real_frozen_model_architecture_is_exact_and_effect_blind() -> None:
    result = audit_frozen_model_architecture(ROOT)
    assert result["status"] == "completed"
    assert result["failures"] == []
    topology = result["frozen_topology"]
    assert topology["num_hidden_layers"] == 28
    assert topology["hidden_size"] == 1024
    assert topology["intermediate_size"] == 3072
    assert topology["num_attention_heads"] == 16
    assert topology["num_key_value_heads"] == 8
    assert topology["query_heads_per_kv_head"] == 2
    assert topology["hidden_act"] == "silu"
    assert topology["rope_theta"] == 1_000_000
    assert topology["attention_dropout"] == 0.0
    assert topology["sliding_window"] is None
    assert topology["tie_word_embeddings"] is True
    assert result["model_pathways"]["q2_recranker_generalqwen"][
        "adaptation"
    ] == "full_parameter"
    assert result["frozen_base_artifact_count"] == 2
    assert result["frozen_model_pathway_count"] == 4
    assert set(result["model_pathways"]) == {
        "q0_qwen3_reranker_06b",
        "q1_instructrec_generalqwen",
        "q2_recranker_generalqwen",
        "q3_tallrec_generalqwen",
    }
    assert result["model_pathways"]["q0_qwen3_reranker_06b"] == {
        "base_artifact": "qwen3_reranker_06b",
        "adaptation": "full_parameter",
        "objective": "pointwise_binary_cross_entropy_on_yes_no_logits",
        "optimizer_steps": 967,
    }
    assert result["model_pathways"]["q1_instructrec_generalqwen"] == {
        "base_artifact": "qwen3_general_06b",
        "adaptation": "full_parameter",
        "objective": "output_only_normalized_candidate_response_nll",
        "optimizer_steps": 967,
    }
    assert {
        row["optimizer_steps"] for row in result["model_pathways"].values()
    } == {967}
    q3 = result["model_pathways"]["q3_tallrec_generalqwen"]
    assert q3["adaptation"] == "lora"
    assert q3["lora_rank"] == 8
    assert q3["lora_targets"] == ["q_proj", "v_proj"]
    assert q3["trainable_parameters"] == 1_146_880
    assert result["config_backed_interface_count"] == 53
    assert result["exact_interface_inventory_count"] == 63
    assert result["config_backed_interfaces_present_in_inventory"] is True
    assert result["dynamic_runtime_or_source_backed_interface_count"] == 10
    assert result["implementation_provenance_covered_interface_count"] == 63
    assert result[
        "all_exact_interfaces_have_config_or_runtime_source_provenance"
    ] is True
    assert result["forward_primitive_count"] == len(FORWARD_PRIMITIVE_CONTRACTS) == 40
    assert result["forward_inference_interface_count"] == 43
    assert result["forward_mapped_interface_count"] == 43
    assert result["forward_missing_interface_ids"] == []
    assert result["forward_extraneous_interface_ids"] == []
    assert result["forward_training_interface_count"] == 20
    assert len(result["forward_training_interfaces_excluded_by_design"]) == 20
    assert result["forward_primitive_interface_coverage_complete"] is True
    assert result["forward_graph_failures"] == []
    assert result["forward_source_binding_count"] == 11
    assert result["transformers_version"] == "5.12.1"
    assert result["forward_peft_version"] == "0.19.1"
    assert result["forward_source_environment_is_frozen_checkpoint_environment"] is True
    assert all(
        row["status"] == "completed"
        and row["missing_fragments"] == []
        and row["source_file_sha256"]
        and row["object_source_sha256"]
        and row["scientific_support_inferred"] is False
        and row["operator_attribution_inferred"] is False
        for row in result["forward_source_bindings"]
    )
    assert {
        "peft_lora_linear_forward",
        "peft_lora_layer_construction",
    }.issubset({row["source_id"] for row in result["forward_source_bindings"]})
    assert "project_q3_peft_loader" in {
        row["source_id"] for row in result["forward_source_bindings"]
    }
    assert all(
        row["forbidden_fragments_present"] == []
        for row in result["forward_source_bindings"]
    )
    assert result["inactive_architecture_path_count"] == 9
    assert result["all_inactive_architecture_paths_verified"] is True
    assert {
        "attention_dropout_stochasticity",
        "sliding_window_attention",
        "sliding_window_dispatch",
        "attention_projection_bias",
        "untied_language_model_head",
        "non_default_or_dynamic_rope_scaling",
        "mlp_projection_bias",
        "native_attention_weight_materialization",
        "alternative_attention_backend",
    } == {row["path_id"] for row in result["inactive_architecture_paths"]}
    assert all(
        row["inactive_verified"] is True
        and row["failures"] == []
        and row["scientific_support_inferred"] is False
        and row["operator_attribution_inferred"] is False
        for row in result["inactive_architecture_paths"]
    )
    assert result["forward_coverage_is_semantic_primitive_census"] is True
    assert result["forward_coverage_is_kernel_instruction_census"] is False
    assert result["operator_attribution_inferred_from_forward_coverage"] is False
    assert result["training_primitive_count"] == len(
        TRAINING_PRIMITIVE_CONTRACTS
    ) == 27
    assert result["training_exact_interface_count"] == 20
    assert result["training_mapped_interface_count"] == 20
    assert result["training_missing_interface_ids"] == []
    assert result["training_extraneous_interface_ids"] == []
    assert result["training_nontraining_interface_count"] == 43
    assert len(result["training_nontraining_interfaces_excluded_by_design"]) == 43
    assert result["training_primitive_interface_coverage_complete"] is True
    assert result["training_update_graph_failures"] == []
    assert result["training_source_binding_count"] == 12
    assert result["training_artifact_binding_count"] == 3
    assert result["training_torch_version"] == "2.6.0+cu124"
    assert result["training_transformers_version"] == "5.12.1"
    assert result["training_peft_version"] == "0.19.1"
    assert all(
        row["status"] == "completed"
        and row["missing_fragments"] == []
        and row["source_file_sha256"]
        and row["object_source_sha256"]
        and row["scientific_support_inferred"] is False
        and row["operator_attribution_inferred"] is False
        for row in result["training_source_bindings"]
    )
    assert all(
        row["status"] == "completed"
        and row["failures"] == []
        and row["scientific_support_inferred"] is False
        and row["operator_attribution_inferred"] is False
        for row in result["training_primitives"]
    )
    adapter_binding = result["training_artifact_bindings"][0]
    assert adapter_binding["observed"]["lora_dropout"] == 0.05
    assert adapter_binding["dropout_executes_before_a_down_projection"] is True
    assert adapter_binding["dropout_identity_at_evaluation"] is True
    implementation_binding = result["training_artifact_bindings"][1]
    assert implementation_binding[
        "current_project_source_matches_frozen_training_identity"
    ] is True
    dtype_binding = result["training_artifact_bindings"][2]
    assert dtype_binding["q0_q2_trainable_master_parameters_are_fp32"] is True
    assert dtype_binding["q3_lora_trainable_parameters_are_fp32"] is True
    assert dtype_binding["q3_frozen_base_parameters_are_bfloat16"] is True
    assert dtype_binding["observed"] == dtype_binding["expected"]
    assert result[
        "training_coverage_is_single_step_semantic_primitive_census"
    ] is True
    assert result["training_coverage_is_multiseed_causal_attribution"] is False
    assert result["operator_attribution_inferred_from_training_coverage"] is False
    assert result["inactive_training_path_count"] == 7
    assert result["all_inactive_training_paths_verified"] is True
    assert {
        "fp16_autocast_and_dynamic_loss_scaling",
        "history_dropout_augmentation",
        "reentrant_gradient_checkpoint_engine",
        "q3_full_parameter_optimization",
        "q3_lora_bias_parameters",
        "q3_nonvanilla_lora_variants",
        "q3_merged_or_disabled_adapter_training",
    } == {row["path_id"] for row in result["inactive_training_paths"]}
    assert all(
        row["inactive_verified"] is True
        and row["failures"] == []
        and row["scientific_support_inferred"] is False
        and row["operator_attribution_inferred"] is False
        for row in result["inactive_training_paths"]
    )
    assert all(
        row["status"] == "completed"
        and row["failures"] == []
        and row["scientific_support_inferred"] is False
        and row["operator_attribution_inferred"] is False
        for row in result["forward_primitives"]
    )
    dynamic = {
        row["interface_id"]: row
        for row in result["dynamic_runtime_or_source_backed_interfaces"]
    }
    assert set(dynamic) == {
        "serialization_tokenization",
        "kv_cache_phase_boundary",
        "q_head_rmsnorm_variance_rescale_and_gain",
        "k_head_rmsnorm_variance_rescale_and_gain",
        "block_input_residual",
        "post_attention_residual",
        "attention_residual_addition",
        "block_output_residual",
        "mlp_residual_addition",
        "candidate_readout_positions",
    }
    assert all(
        row["status"] == "completed"
        and row["failures"] == []
        and all(source["sha256"] for source in row["source_identities"])
        and row["scientific_support_inferred"] is False
        for row in dynamic.values()
    )
    assert set(dynamic["post_attention_residual"]["runtime_identity_models"]) == set(
        IDENTITY_SMOKES
    )
    assert set(
        dynamic["q_head_rmsnorm_variance_rescale_and_gain"][
            "runtime_identity_models"
        ]
    ) == set(IDENTITY_SMOKES)
    assert dynamic["attention_residual_addition"]["runtime_algebra_key"] == (
        "post_attention_recomposition"
    )
    assert set(
        dynamic["attention_residual_addition"]["runtime_algebra_models"]
    ) == set(IDENTITY_SMOKES)
    assert dynamic["mlp_residual_addition"]["runtime_algebra_key"] == (
        "block_output_recomposition"
    )
    assert result["runtime_identity_smoke_count"] == 4
    assert result["runtime_hook_node_count"] == 18
    assert result["runtime_attention_backend"] == "sdpa"
    assert result["runtime_identity_and_recomposition_validated"] is True
    assert {
        row["method_id"] for row in result["runtime_identity_smokes"]
    } == set(IDENTITY_SMOKES)
    assert all(
        row["hook_nodes_validated"] == 18
        and row["maximum_identity_error"] == 0.0
        and row["algebra_recomposition_passed"] is True
        and row["native_attention_wrapper_noop_max_abs_score_delta"] == 0.0
        and row["qrels_read"] is False
        and row["source_test_opened"] is False
        and row["failures"] == []
        for row in result["runtime_identity_smokes"]
    )
    assert result["superseded_runtime_lineage_count"] == 1
    assert result["superseded_runtime_lineage_retained"] is True
    superseded = result["superseded_runtime_lineage"][0]
    assert superseded["run_id"] in SUPERSEDED_IDENTITY_SMOKES
    assert superseded["status"] == "failed_identity"
    assert superseded["maximum_identity_error"] > superseded["identity_tolerance"]
    assert superseded["retained_as_mechanical_lineage"] is True
    assert superseded["scientific_result_eligible"] is False
    assert superseded["failures"] == []
    assert result["scientific_effect_values_read"] is False
    assert result["scientific_support_inferred"] is False
    assert result["qrels_read_by_audit"] is False
    assert result["source_test_opened"] is False


def test_architecture_audit_fails_closed_on_adapter_topology_drift(
    tmp_path: Path,
) -> None:
    for relative in (
        "models/huggingface/Qwen3-0.6B/config.json",
        "models/huggingface/Qwen3-Reranker-0.6B/config.json",
        "configs/methods/kuaisearch_motivation_v12_q0_qwen3_reranker_06b.yaml",
        "configs/methods/kuaisearch_motivation_v12_q1_instructrec_generalqwen.yaml",
        "configs/methods/kuaisearch_motivation_v12_q2_recranker_generalqwen.yaml",
        "configs/methods/kuaisearch_motivation_v12_q3_tallrec_generalqwen.yaml",
        "artifacts/motivation_v1_2/checkpoints/q0_qwen3_reranker_06b_seed20260714/checkpoint_latest/model/config.json",
        "artifacts/motivation_v1_2/checkpoints/q1_instructrec_generalqwen_seed20260714/checkpoint_latest/model/config.json",
        "artifacts/motivation_v1_2/checkpoints/q2_recranker_generalqwen_seed20260714/checkpoint_latest/model/config.json",
        "artifacts/motivation_v1_2/checkpoints/q3_tallrec_generalqwen_seed20260714/checkpoint_latest/model/adapter_config.json",
        "artifacts/motivation_v1_2/checkpoints/q0_qwen3_reranker_06b_seed20260714/training_metadata.json",
        "artifacts/motivation_v1_2/checkpoints/q1_instructrec_generalqwen_seed20260714/training_metadata.json",
        "artifacts/motivation_v1_2/checkpoints/q2_recranker_generalqwen_seed20260714/training_metadata.json",
        "artifacts/motivation_v1_2/checkpoints/q3_tallrec_generalqwen_seed20260714/training_metadata.json",
        *(path.as_posix() for path in IDENTITY_SMOKES.values()),
        *(path.as_posix() for path in SUPERSEDED_IDENTITY_SMOKES.values()),
    ):
        source = ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    adapter_path = tmp_path / Q3_ADAPTER_CONFIG
    adapter = json.loads(adapter_path.read_text(encoding="utf-8"))
    adapter["target_modules"] = ["o_proj"]
    adapter_path.write_text(json.dumps(adapter), encoding="utf-8")
    metadata_path = tmp_path / (
        "artifacts/motivation_v1_2/checkpoints/"
        "q3_tallrec_generalqwen_seed20260714/training_metadata.json"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["package_versions"]["peft"] = "0.0.0"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    result = audit_frozen_model_architecture(tmp_path)
    assert result["status"] == "failed"
    assert "Q3 adapter target_modules mismatch" in result["failures"]
    assert any(
        "frozen runtime PEFT version is not 0.19.1" in failure
        for failure in result["failures"]
    )
    assert result["scientific_effect_values_read"] is False


def test_architecture_audit_fails_closed_when_forward_primitive_mapping_is_missing(
    monkeypatch,
) -> None:
    reduced = tuple(
        row
        for row in FORWARD_PRIMITIVE_CONTRACTS
        if row["primitive_id"] != "mlp_silu_nonlinearity"
    )
    reduced = tuple(
        {**row, "execution_order": index}
        for index, row in enumerate(reduced, start=1)
    )
    monkeypatch.setattr(
        architecture_audit_module,
        "FORWARD_PRIMITIVE_CONTRACTS",
        reduced,
    )
    result = audit_frozen_model_architecture(ROOT)
    assert result["status"] == "failed"
    assert result["forward_primitive_interface_coverage_complete"] is False
    assert result["forward_missing_interface_ids"] == ["mlp_silu_gate"]
    assert any("mlp_silu_gate" in failure for failure in result["failures"])
    assert result["scientific_effect_values_read"] is False


def test_architecture_audit_fails_closed_when_training_primitive_mapping_is_missing(
    monkeypatch,
) -> None:
    reduced = tuple(
        row
        for row in TRAINING_PRIMITIVE_CONTRACTS
        if row["primitive_id"] != "q3_lora_input_dropout"
    )
    reduced = tuple(
        {**row, "execution_order": index}
        for index, row in enumerate(reduced, start=1)
    )
    monkeypatch.setattr(
        architecture_audit_module,
        "TRAINING_PRIMITIVE_CONTRACTS",
        reduced,
    )
    result = audit_frozen_model_architecture(ROOT)
    assert result["status"] == "failed"
    assert result["training_primitive_interface_coverage_complete"] is False
    assert result["training_missing_interface_ids"] == [
        "lora_training_input_dropout"
    ]
    assert any(
        "lora_training_input_dropout" in failure for failure in result["failures"]
    )
    assert result["scientific_effect_values_read"] is False


def test_architecture_audit_fails_closed_when_q3_adapter_forward_is_omitted(
    monkeypatch,
) -> None:
    reduced = tuple(
        row
        for row in FORWARD_PRIMITIVE_CONTRACTS
        if row["primitive_id"] != "q3_query_lora_scaled_adapter_injection"
    )
    reduced = tuple(
        {**row, "execution_order": index}
        for index, row in enumerate(reduced, start=1)
    )
    monkeypatch.setattr(
        architecture_audit_module,
        "FORWARD_PRIMITIVE_CONTRACTS",
        reduced,
    )
    result = audit_frozen_model_architecture(ROOT)
    assert result["status"] == "failed"
    assert result["forward_primitive_interface_coverage_complete"] is False
    assert result["forward_missing_interface_ids"] == [
        "q3_q_lora_scaled_adapter_injection"
    ]
    assert any(
        "q3_q_lora_scaled_adapter_injection" in failure
        for failure in result["failures"]
    )
    assert result["scientific_effect_values_read"] is False


def test_architecture_audit_fails_closed_on_q3_adapter_dtype_drift(
    monkeypatch,
) -> None:
    original = architecture_audit_module._safetensors_dtype_summary

    def drifted(directory: Path) -> dict[str, int]:
        observed = original(directory)
        if directory == ROOT / Q3_ADAPTER_CONFIG.parent:
            return {"BF16": sum(observed.values())}
        return observed

    monkeypatch.setattr(
        architecture_audit_module,
        "_safetensors_dtype_summary",
        drifted,
    )
    result = audit_frozen_model_architecture(ROOT)
    assert result["status"] == "failed"
    dtype_binding = result["training_artifact_bindings"][2]
    assert dtype_binding["status"] == "failed"
    assert "frozen checkpoint tensor dtype/count contract mismatch" in dtype_binding[
        "failures"
    ]
    assert result["scientific_effect_values_read"] is False
