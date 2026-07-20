"""Effect-blind audit of the frozen Qwen topology used by the deep dive."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from myrec.mechanism.deep_dive_evidence_topology import MODEL_IDS
from myrec.mechanism.transformer_interface_inventory import (
    TRANSFORMER_INTERFACE_INVENTORY,
)
from myrec.mechanism.transformer_instrumentation import (
    BLOCK_NODE_IDS,
    FINAL_NODE_IDS,
)
from myrec.utils.hashing import sha256_file


BASE_CONFIG = Path("models/huggingface/Qwen3-0.6B/config.json")
Q0_BASE_CONFIG = Path("models/huggingface/Qwen3-Reranker-0.6B/config.json")
Q0_METHOD_CONFIG = Path(
    "configs/methods/kuaisearch_motivation_v12_q0_qwen3_reranker_06b.yaml"
)
Q1_METHOD_CONFIG = Path(
    "configs/methods/kuaisearch_motivation_v12_q1_instructrec_generalqwen.yaml"
)
Q2_METHOD_CONFIG = Path(
    "configs/methods/kuaisearch_motivation_v12_q2_recranker_generalqwen.yaml"
)
Q3_METHOD_CONFIG = Path(
    "configs/methods/kuaisearch_motivation_v12_q3_tallrec_generalqwen.yaml"
)
Q0_SAVED_CONFIG = Path(
    "artifacts/motivation_v1_2/checkpoints/q0_qwen3_reranker_06b_seed20260714/"
    "checkpoint_latest/model/config.json"
)
Q1_SAVED_CONFIG = Path(
    "artifacts/motivation_v1_2/checkpoints/q1_instructrec_generalqwen_seed20260714/"
    "checkpoint_latest/model/config.json"
)
Q2_SAVED_CONFIG = Path(
    "artifacts/motivation_v1_2/checkpoints/q2_recranker_generalqwen_seed20260714/"
    "checkpoint_latest/model/config.json"
)
Q3_ADAPTER_CONFIG = Path(
    "artifacts/motivation_v1_2/checkpoints/q3_tallrec_generalqwen_seed20260714/"
    "checkpoint_latest/model/adapter_config.json"
)
Q0_TRAINING_METADATA = Path(
    "artifacts/motivation_v1_2/checkpoints/q0_qwen3_reranker_06b_seed20260714/"
    "training_metadata.json"
)
Q1_TRAINING_METADATA = Path(
    "artifacts/motivation_v1_2/checkpoints/q1_instructrec_generalqwen_seed20260714/"
    "training_metadata.json"
)
Q2_TRAINING_METADATA = Path(
    "artifacts/motivation_v1_2/checkpoints/q2_recranker_generalqwen_seed20260714/"
    "training_metadata.json"
)
Q3_TRAINING_METADATA = Path(
    "artifacts/motivation_v1_2/checkpoints/q3_tallrec_generalqwen_seed20260714/"
    "training_metadata.json"
)
IDENTITY_SMOKES = {
    "q0_qwen3_reranker_06b": Path(
        "runs/20260718_kuaisearch_mech_d0_q0_identity_smoke_v3/metadata.json"
    ),
    "q1_instructrec_generalqwen": Path(
        "runs/20260718_kuaisearch_mech_d0_q1_identity_smoke_v3/metadata.json"
    ),
    "q2_recranker_generalqwen": Path(
        "runs/20260718_kuaisearch_mech_d0_q2_identity_smoke_v3/metadata.json"
    ),
    "q3_tallrec_generalqwen": Path(
        "runs/20260718_kuaisearch_mech_d0_q3_identity_smoke_v3/metadata.json"
    ),
}
SUPERSEDED_IDENTITY_SMOKES = {
    "20260718_kuaisearch_mech_d0_q1_identity_smoke_v2": Path(
        "runs/20260718_kuaisearch_mech_d0_q1_identity_smoke_v2/metadata.json"
    )
}

EXPECTED_BASE_TOPOLOGY = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "hidden_size": 1024,
    "intermediate_size": 3072,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-6,
    "rope_theta": 1_000_000,
    "attention_dropout": 0.0,
    "sliding_window": None,
    "use_sliding_window": False,
    "max_position_embeddings": 40_960,
    "tie_word_embeddings": True,
    "attention_bias": False,
}

CONFIG_BACKED_INTERFACES = {
    "token_embedding_lookup",
    "tied_lm_head_rows",
    "autoregressive_causal_attention_mask",
    "q_pre_norm",
    "q3_q_lora_scaled_adapter_injection",
    "k_pre_norm",
    "q_post_norm_pre_rope",
    "k_post_norm_pre_rope",
    "q_post_rope",
    "k_post_rope",
    "v_projection",
    "q3_v_lora_scaled_adapter_injection",
    "attention_scaled_qk_logits",
    "attention_softmax_edge_weights",
    "attention_head_output_pre_o",
    "gqa_query_to_kv_grouping",
    "attention_o_projection",
    "input_rmsnorm_output",
    "input_rmsnorm_variance_rescale_and_gain",
    "post_attention_rmsnorm_output",
    "post_attention_rmsnorm_variance_rescale_and_gain",
    "mlp_gate_projection",
    "mlp_up_projection",
    "mlp_silu_gate",
    "mlp_swiglu_product",
    "mlp_down_projection",
    "final_rmsnorm_input",
    "final_rmsnorm_output",
    "final_rmsnorm_variance_rescale_and_gain",
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

# These ten interfaces are dynamic execution contracts rather than scalar
# architecture/config fields.  Keep their source and, where applicable,
# runtime-hook provenance explicit so that ``53 config-backed`` is never read
# as ``ten unaudited``.
DYNAMIC_INTERFACE_CONTRACTS = {
    "serialization_tokenization": {
        "binding_kind": "project_owned_serialization_source",
        "source_paths": (
            Path("src/myrec/baselines/motivation_v12_ranker.py"),
            Path("src/myrec/mechanism/representation_probe.py"),
        ),
        "runtime_identity_node": None,
    },
    "kv_cache_phase_boundary": {
        "binding_kind": "project_owned_q1_cache_phase_source",
        "source_paths": (
            Path("src/myrec/baselines/motivation_v12_ranker.py"),
            Path("src/myrec/mechanism/q1_kv_trajectory.py"),
        ),
        "runtime_identity_node": None,
    },
    "q_head_rmsnorm_variance_rescale_and_gain": {
        "binding_kind": "runtime_qk_norm_hook_identity",
        "source_paths": (
            Path("src/myrec/mechanism/transformer_instrumentation.py"),
        ),
        "runtime_identity_node": "q_post_norm_pre_rope",
    },
    "k_head_rmsnorm_variance_rescale_and_gain": {
        "binding_kind": "runtime_qk_norm_hook_identity",
        "source_paths": (
            Path("src/myrec/mechanism/transformer_instrumentation.py"),
        ),
        "runtime_identity_node": "k_post_norm_pre_rope",
    },
    "block_input_residual": {
        "binding_kind": "runtime_hook_and_recomposition_identity",
        "source_paths": (
            Path("src/myrec/mechanism/transformer_instrumentation.py"),
        ),
        "runtime_identity_node": "block_input_residual",
    },
    "post_attention_residual": {
        "binding_kind": "runtime_hook_and_recomposition_identity",
        "source_paths": (
            Path("src/myrec/mechanism/transformer_instrumentation.py"),
        ),
        "runtime_identity_node": "post_attention_residual",
    },
    "attention_residual_addition": {
        "binding_kind": "runtime_algebra_recomposition_identity",
        "source_paths": (
            Path("src/myrec/mechanism/transformer_instrumentation.py"),
            Path("src/myrec/mechanism/deep_dive_smoke.py"),
        ),
        "runtime_identity_node": "post_attention_residual",
        "runtime_algebra_key": "post_attention_recomposition",
    },
    "block_output_residual": {
        "binding_kind": "runtime_hook_and_recomposition_identity",
        "source_paths": (
            Path("src/myrec/mechanism/transformer_instrumentation.py"),
        ),
        "runtime_identity_node": "block_output_residual",
    },
    "mlp_residual_addition": {
        "binding_kind": "runtime_algebra_recomposition_identity",
        "source_paths": (
            Path("src/myrec/mechanism/transformer_instrumentation.py"),
            Path("src/myrec/mechanism/deep_dive_smoke.py"),
        ),
        "runtime_identity_node": "block_output_residual",
        "runtime_algebra_key": "block_output_recomposition",
    },
    "candidate_readout_positions": {
        "binding_kind": "model_specific_prompt_and_cache_position_source",
        "source_paths": (
            Path("src/myrec/mechanism/representation_probe.py"),
            Path("src/myrec/mechanism/q0_representation_prompt.py"),
            Path("src/myrec/mechanism/q1_kv_trajectory.py"),
        ),
        "runtime_identity_node": None,
    },
}


def _primitive(
    primitive_id: str,
    execution_order: int,
    interface_ids: tuple[str, ...],
    implementation_step: str,
    *,
    model_scope: tuple[str, ...] = MODEL_IDS,
) -> dict[str, Any]:
    return {
        "primitive_id": primitive_id,
        "execution_order": execution_order,
        "interface_ids": interface_ids,
        "implementation_step": implementation_step,
        "model_scope": model_scope,
    }


def _ordered_primitives(
    contracts: tuple[dict[str, Any], ...]
) -> tuple[dict[str, Any], ...]:
    """Make tuple position the single source of truth for execution order."""

    return tuple(
        {**contract, "execution_order": index}
        for index, contract in enumerate(contracts, start=1)
    )


# This is a semantic census of the frozen inference graph, not an attempt to
# enumerate every tensor view or kernel instruction.  It deliberately starts
# at project-owned serialization and ends at each native ranking score so that
# a complete decoder-only inventory cannot silently omit the task readout.
FORWARD_PRIMITIVE_CONTRACTS = _ordered_primitives((
    _primitive(
        "project_input_serialization",
        1,
        ("serialization_tokenization",),
        "field whitelist, prompt construction, truncation, and tokenization",
    ),
    _primitive(
        "candidate_readout_position_binding",
        2,
        ("candidate_readout_positions",),
        "bind model-specific prompt/cache positions consumed by the native score",
    ),
    _primitive(
        "token_embedding_lookup",
        3,
        ("token_embedding_lookup",),
        "Qwen3Model.embed_tokens(input_ids)",
    ),
    _primitive(
        "q1_prefix_continuation_cache_phase",
        4,
        ("kv_cache_phase_boundary",),
        "Q1 prefix cache and continuation/answer-token phase boundary",
        model_scope=(MODEL_IDS[1],),
    ),
    _primitive(
        "position_id_and_rotary_basis_construction",
        5,
        ("q_post_rope", "k_post_rope"),
        "position_ids plus Qwen3RotaryEmbedding cosine/sine basis",
    ),
    _primitive(
        "causal_mask_construction",
        6,
        ("autoregressive_causal_attention_mask",),
        "create_causal_mask for the frozen full-attention layer type",
    ),
    _primitive(
        "decoder_block_input_residual",
        7,
        ("block_input_residual",),
        "incoming residual captured before each decoder block",
    ),
    _primitive(
        "input_rmsnorm",
        8,
        ("input_rmsnorm_variance_rescale_and_gain", "input_rmsnorm_output"),
        "FP32 RMS variance rescaling and learned gain before self-attention",
    ),
    _primitive(
        "query_projection",
        9,
        ("q_pre_norm",),
        "frozen base q_proj branch and query-head reshape; Q3 adapter contribution is enumerated separately",
    ),
    _primitive(
        "q3_query_lora_scaled_adapter_injection",
        10,
        ("q3_q_lora_scaled_adapter_injection",),
        "Q3 eval identity-dropout then A/B low-rank query branch, alpha/r scaling, and addition to the frozen base q_proj output",
        model_scope=(MODEL_IDS[3],),
    ),
    _primitive(
        "query_head_rmsnorm",
        10,
        (
            "q_head_rmsnorm_variance_rescale_and_gain",
            "q_post_norm_pre_rope",
        ),
        "head-dimensional q_norm between q_proj and RoPE",
    ),
    _primitive(
        "key_projection",
        11,
        ("k_pre_norm",),
        "self_attn.k_proj and key-head reshape",
    ),
    _primitive(
        "key_head_rmsnorm",
        12,
        (
            "k_head_rmsnorm_variance_rescale_and_gain",
            "k_post_norm_pre_rope",
        ),
        "head-dimensional k_norm between k_proj and RoPE",
    ),
    _primitive(
        "value_projection",
        13,
        ("v_projection",),
        "frozen base v_proj branch and value-head reshape; Q3 adapter contribution is enumerated separately",
    ),
    _primitive(
        "q3_value_lora_scaled_adapter_injection",
        15,
        ("q3_v_lora_scaled_adapter_injection",),
        "Q3 eval identity-dropout then A/B low-rank value branch, alpha/r scaling, and addition to the frozen base v_proj output",
        model_scope=(MODEL_IDS[3],),
    ),
    _primitive(
        "query_rotary_phase_application",
        14,
        ("q_post_rope",),
        "apply_rotary_pos_emb to normalized query heads",
    ),
    _primitive(
        "key_rotary_phase_application",
        15,
        ("k_post_rope",),
        "apply_rotary_pos_emb to normalized key heads",
    ),
    _primitive(
        "q1_key_value_cache_update",
        16,
        ("kv_cache_phase_boundary",),
        "past_key_values.update on Q1 cached prefix/continuation forwards",
        model_scope=(MODEL_IDS[1],),
    ),
    _primitive(
        "grouped_query_key_value_binding",
        17,
        ("gqa_query_to_kv_grouping",),
        "SDPA enable_gqa or exact K/V repeat across two query heads per KV head",
    ),
    _primitive(
        "scaled_query_key_dot_product",
        18,
        ("attention_scaled_qk_logits",),
        "head_dim^-0.5 scaled QK dot-product inside SDPA",
    ),
    _primitive(
        "causal_mask_application",
        19,
        ("autoregressive_causal_attention_mask",),
        "apply the additive/causal visibility mask before attention normalization",
    ),
    _primitive(
        "attention_softmax",
        20,
        ("attention_softmax_edge_weights",),
        "normalize masked scaled-QK logits over source positions",
    ),
    _primitive(
        "attention_probability_weighted_value_sum",
        21,
        ("v_projection", "attention_head_output_pre_o"),
        "probability-weighted value transport for every query head",
    ),
    _primitive(
        "attention_head_merge",
        22,
        ("attention_head_output_pre_o",),
        "transpose/contiguous reshape of sixteen query-head outputs",
    ),
    _primitive(
        "attention_output_projection",
        23,
        ("attention_o_projection",),
        "self_attn.o_proj before residual composition",
    ),
    _primitive(
        "attention_residual_composition",
        24,
        ("attention_residual_addition", "post_attention_residual"),
        "elementwise incoming-residual plus attention-output composition",
    ),
    _primitive(
        "post_attention_rmsnorm",
        25,
        (
            "post_attention_rmsnorm_variance_rescale_and_gain",
            "post_attention_rmsnorm_output",
        ),
        "FP32 RMS variance rescaling and learned gain before the MLP",
    ),
    _primitive(
        "mlp_gate_projection",
        26,
        ("mlp_gate_projection",),
        "bias-free SwiGLU gate_proj",
    ),
    _primitive(
        "mlp_up_projection",
        27,
        ("mlp_up_projection",),
        "bias-free SwiGLU up_proj",
    ),
    _primitive(
        "mlp_silu_nonlinearity",
        28,
        ("mlp_silu_gate",),
        "SiLU applied to the gate projection",
    ),
    _primitive(
        "mlp_swiglu_product",
        29,
        ("mlp_swiglu_product",),
        "elementwise SiLU(gate_proj) times up_proj",
    ),
    _primitive(
        "mlp_down_projection",
        30,
        ("mlp_down_projection",),
        "bias-free down_proj back to the residual width",
    ),
    _primitive(
        "mlp_residual_composition",
        31,
        ("mlp_residual_addition", "block_output_residual"),
        "elementwise post-attention residual plus MLP-output composition",
    ),
    _primitive(
        "final_rmsnorm_input_boundary",
        32,
        ("final_rmsnorm_input",),
        "block-27 output supplied to the final model norm",
    ),
    _primitive(
        "final_rmsnorm",
        33,
        ("final_rmsnorm_variance_rescale_and_gain", "final_rmsnorm_output"),
        "final FP32 RMS variance rescaling and learned gain",
    ),
    _primitive(
        "tied_language_model_head_projection",
        34,
        ("tied_lm_head_rows",),
        "tied lm_head projection at registered candidate/answer positions",
    ),
    _primitive(
        "q0_native_score_formula",
        35,
        ("q0_next_token_yes_no_logit_difference",),
        "Q0 next-token Yes-minus-No logit difference",
        model_scope=(MODEL_IDS[0],),
    ),
    _primitive(
        "q1_native_score_formula",
        36,
        ("q1_candidate_response_mean_log_likelihood",),
        "Q1 normalized candidate-response mean log likelihood",
        model_scope=(MODEL_IDS[1],),
    ),
    _primitive(
        "q2_native_score_formula",
        37,
        ("q2_next_token_yes_no_logit_difference",),
        "Q2 next-token Yes-minus-No logit difference",
        model_scope=(MODEL_IDS[2],),
    ),
    _primitive(
        "q3_native_score_formula",
        38,
        ("q3_two_path_mean_log_likelihood_difference",),
        "Q3 two teacher-forced Yes/No path mean-log-likelihood difference",
        model_scope=(MODEL_IDS[3],),
    ),
))


# The training census is the update-side counterpart of the frozen forward
# census.  Alternative Q0--Q3 objective branches share one post-loss update
# chain; repeated interface mappings deliberately expose substeps that the
# exact-interface table groups into one scientific claim boundary.
TRAINING_PRIMITIVE_CONTRACTS = _ordered_primitives((
    _primitive(
        "bfloat16_autocast_training_forward",
        1,
        ("bfloat16_autocast_training_forward",),
        "run each frozen model-specific forward and objective inside CUDA BF16 autocast; FP16 GradScaler remains disabled",
    ),
    _primitive(
        "nonreentrant_decoder_checkpoint_recomputation",
        2,
        ("nonreentrant_gradient_checkpoint_recomputation",),
        "recompute checkpointed decoder activations with use_reentrant=False during autograd",
    ),
    _primitive(
        "q3_frozen_embedding_output_gradient_bridge",
        3,
        ("q3_input_activation_requires_grad_bridge",),
        "mark frozen embedding outputs requires-grad so non-reentrant checkpointed q/v LoRA modules receive backward signal",
        model_scope=(MODEL_IDS[3],),
    ),
    _primitive(
        "q3_bf16_base_to_fp32_adapter_input_cast",
        4,
        ("q3_fp32_lora_bf16_base_cast_boundary",),
        "cast the Q3 BF16 base-projection input activation to the FP32 LoRA A parameter dtype",
        model_scope=(MODEL_IDS[3],),
    ),
    _primitive(
        "q3_fp32_adapter_to_bf16_base_result_cast",
        5,
        ("q3_fp32_lora_bf16_base_cast_boundary",),
        "cast the scaled FP32 LoRA branch result back to the BF16 base projection result dtype",
        model_scope=(MODEL_IDS[3],),
    ),
    _primitive(
        "q3_lora_input_dropout",
        1,
        ("lora_training_input_dropout",),
        "training-only Bernoulli dropout with p=0.05 on the q/v adapter input",
        model_scope=(MODEL_IDS[3],),
    ),
    _primitive(
        "q3_q_lora_a_down_projection",
        2,
        ("lora_q_low_rank_a_factor",),
        "q_proj adapter A maps the dropout output from width 1024 to rank 8",
        model_scope=(MODEL_IDS[3],),
    ),
    _primitive(
        "q3_q_lora_b_up_projection",
        3,
        ("lora_q_low_rank_b_factor",),
        "q_proj adapter B maps rank 8 to the query projection width",
        model_scope=(MODEL_IDS[3],),
    ),
    _primitive(
        "q3_v_lora_a_down_projection",
        4,
        ("lora_v_low_rank_a_factor",),
        "v_proj adapter A maps the dropout output from width 1024 to rank 8",
        model_scope=(MODEL_IDS[3],),
    ),
    _primitive(
        "q3_v_lora_b_up_projection",
        5,
        ("lora_v_low_rank_b_factor",),
        "v_proj adapter B maps rank 8 to the value projection width",
        model_scope=(MODEL_IDS[3],),
    ),
    _primitive(
        "q0_pointwise_bce_objective",
        6,
        ("q0_pointwise_bce_loss",),
        "binary_cross_entropy_with_logits over frozen Yes-minus-No scores",
        model_scope=(MODEL_IDS[0],),
    ),
    _primitive(
        "q1_response_sequence_nll_objective",
        7,
        ("q1_normalized_response_nll",),
        "mean token cross entropy over each normalized candidate response",
        model_scope=(MODEL_IDS[1],),
    ),
    _primitive(
        "q2_ranknet_objective_term",
        8,
        ("q2_pairwise_ranknet_loss",),
        "mean softplus of every frozen grade-different negative score margin",
        model_scope=(MODEL_IDS[2],),
    ),
    _primitive(
        "q2_listnet_objective_term",
        9,
        ("q2_listwise_listnet_loss",),
        "tie-aware target distribution cross entropy against score log-softmax",
        model_scope=(MODEL_IDS[2],),
    ),
    _primitive(
        "q2_half_ranknet_half_listnet_composition",
        10,
        ("q2_pairwise_ranknet_loss", "q2_listwise_listnet_loss"),
        "frozen 0.5 RankNet plus 0.5 ListNet scalar objective",
        model_scope=(MODEL_IDS[2],),
    ),
    _primitive(
        "q3_alignment_sequence_nll_objective",
        11,
        ("q3_alignment_nll_loss",),
        "mean token cross entropy over output-only Yes/No alignment targets",
        model_scope=(MODEL_IDS[3],),
    ),
    _primitive(
        "microbatch_loss_scaling_and_autograd_backward",
        12,
        ("gradient_accumulation_and_global_clip",),
        "divide by the realized accumulation target, scale if FP16, and backpropagate",
    ),
    _primitive(
        "microbatch_gradient_accumulation",
        13,
        ("gradient_accumulation_and_global_clip",),
        "sum gradients until the frozen model-specific accumulation boundary",
    ),
    _primitive(
        "gradient_unscale_and_global_norm_clip",
        14,
        ("gradient_accumulation_and_global_clip",),
        "unscale and apply one global L2-norm coefficient with max_norm=1.0",
    ),
    _primitive(
        "adam_first_moment_update",
        15,
        ("adam_moment_preconditioned_direction",),
        "update the exponential first moment from the clipped gradient",
    ),
    _primitive(
        "adam_second_moment_update",
        16,
        ("adam_moment_preconditioned_direction",),
        "update the exponential squared-gradient moment",
    ),
    _primitive(
        "adam_bias_correction_and_preconditioning",
        17,
        ("adam_moment_preconditioned_direction",),
        "bias-correct both moments and divide by root second moment plus epsilon",
    ),
    _primitive(
        "adamw_decoupled_weight_decay",
        18,
        ("decoupled_weight_decay_term",),
        "form the decoupled parameter-proportional decay delta",
    ),
    _primitive(
        "linear_warmup_decay_learning_rate",
        19,
        ("learning_rate_scaled_effective_parameter_delta",),
        "apply the frozen linear warmup then linear-decay learning-rate schedule",
    ),
    _primitive(
        "effective_parameter_delta_application",
        20,
        ("learning_rate_scaled_effective_parameter_delta",),
        "apply the scheduler-scaled moment-plus-decay delta at an optimizer boundary",
    ),
    _primitive(
        "q3_q_gauge_invariant_function_delta",
        21,
        ("lora_q_effective_delta_weight",),
        "compose q_proj alpha/r times B@A and its exact A/B/interaction update",
        model_scope=(MODEL_IDS[3],),
    ),
    _primitive(
        "q3_v_gauge_invariant_function_delta",
        22,
        ("lora_v_effective_delta_weight",),
        "compose v_proj alpha/r times B@A and its exact A/B/interaction update",
        model_scope=(MODEL_IDS[3],),
    ),
))


FORWARD_SOURCE_SENTINELS = {
    "qwen3_model": {
        "object": "Qwen3Model",
        "fragments": (
            "inputs_embeds = self.embed_tokens(input_ids)",
            "create_causal_mask",
            "position_embeddings = self.rotary_emb",
            "for i, decoder_layer in enumerate",
            "hidden_states = self.norm(hidden_states)",
        ),
    },
    "qwen3_decoder_layer": {
        "object": "Qwen3DecoderLayer",
        "fragments": (
            "self.input_layernorm",
            "self.self_attn",
            "hidden_states = residual + hidden_states",
            "self.post_attention_layernorm",
            "self.mlp",
        ),
    },
    "qwen3_attention": {
        "object": "Qwen3Attention",
        "fragments": (
            "bias=config.attention_bias",
            "self.q_norm(self.q_proj",
            "self.k_norm(self.k_proj",
            "self.v_proj(hidden_states)",
            "apply_rotary_pos_emb",
            "past_key_values.update",
            "attention_interface",
            "self.o_proj(attn_output)",
        ),
    },
    "qwen3_mlp": {
        "object": "Qwen3MLP",
        "fragments": (
            "bias=False",
            "self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))",
        ),
    },
    "qwen3_rmsnorm": {
        "object": "Qwen3RMSNorm",
        "fragments": (
            "hidden_states.pow(2).mean",
            "torch.rsqrt",
            "self.weight * hidden_states",
        ),
    },
    "qwen3_rotary_embedding": {
        "object": "Qwen3RotaryEmbedding",
        "fragments": (
            "@dynamic_rope_update",
            "freqs =",
            "emb.cos()",
            "emb.sin()",
        ),
    },
    "qwen3_causal_lm": {
        "object": "Qwen3ForCausalLM",
        "fragments": ("outputs: BaseModelOutputWithPast = self.model", "self.lm_head"),
    },
    "transformers_sdpa": {
        "object": "sdpa_attention_forward",
        "fragments": (
            "repeat_kv",
            '"enable_gqa": True',
            "scaled_dot_product_attention",
            "attn_output.transpose(1, 2).contiguous()",
            "return attn_output, None",
        ),
    },
    "peft_lora_linear_forward": {
        "object": "Linear.forward",
        "fragments": (
            "result = self.base_layer",
            "dropout = self.lora_dropout[active_adapter]",
            "scaling = self.scaling[active_adapter]",
            "result = result + lora_B(lora_A(dropout(x))) * scaling",
            "result = result.to(torch_result_dtype)",
        ),
    },
    "peft_lora_layer_construction": {
        "object": "LoraLayer.update_layer",
        "fragments": (
            "lora_dropout_layer = nn.Dropout(p=lora_dropout)",
            "self.lora_A[adapter_name] = nn.Linear",
            "self.lora_B[adapter_name] = nn.Linear",
            "self.scaling[adapter_name] = lora_alpha / r",
        ),
    },
    "project_q3_peft_loader": {
        "object": "_load_model_and_tokenizer",
        "fragments": (
            "LoraConfig(",
            "lora_dropout=float(method[\"lora_dropout\"])",
            "model = PeftModel.from_pretrained",
            "is_trainable=training",
            "model.train(training)",
        ),
        "forbidden_fragments": ("merge_and_unload", "disable_adapter"),
    },
}


FROZEN_FORWARD_SOURCE_FILES = {
    "Qwen3Model": "transformers/models/qwen3/modeling_qwen3.py",
    "Qwen3DecoderLayer": "transformers/models/qwen3/modeling_qwen3.py",
    "Qwen3Attention": "transformers/models/qwen3/modeling_qwen3.py",
    "Qwen3MLP": "transformers/models/qwen3/modeling_qwen3.py",
    "Qwen3RMSNorm": "transformers/models/qwen3/modeling_qwen3.py",
    "Qwen3RotaryEmbedding": "transformers/models/qwen3/modeling_qwen3.py",
    "Qwen3ForCausalLM": "transformers/models/qwen3/modeling_qwen3.py",
    "sdpa_attention_forward": "transformers/integrations/sdpa_attention.py",
    "Linear.forward": "peft/tuners/lora/layer.py",
    "LoraLayer.update_layer": "peft/tuners/lora/layer.py",
}


TRAINING_SOURCE_SENTINELS = {
    "project_training_loop": {
        "object": "train_motivation_v12_ranker",
        "fragments": (
            'scaler = torch.amp.GradScaler("cuda", enabled=dtype == "float16")',
            "with torch.autocast(",
            "dtype=autocast_dtype",
            "scaler.scale(raw_loss / accumulation_target).backward()",
            "scaler.unscale_(optimizer)",
            "torch.nn.utils.clip_grad_norm_",
            "scaler.step(optimizer)",
            "scheduler.step()",
        ),
    },
    "project_training_objective_dispatch": {
        "object": "_training_batch_loss",
        "fragments": (
            "F.binary_cross_entropy_with_logits",
            "pairwise_ranknet_loss",
            "listwise_softmax_loss",
            "pairwise_loss_weight",
            "listwise_loss_weight",
        ),
    },
    "project_model_training_loader": {
        "object": "_load_model_and_tokenizer",
        "fragments": (
            "gradient_checkpointing_enable",
            'gradient_checkpointing_kwargs={"use_reentrant": False}',
            "model.enable_input_require_grads()",
            "model.train(training)",
        ),
        "forbidden_fragments": ("merge_and_unload", "disable_adapter"),
    },
    "project_ranknet_loss": {
        "object": "pairwise_ranknet_loss",
        "fragments": ("F.softplus(-margins).mean()",),
    },
    "project_listnet_loss": {
        "object": "listwise_softmax_loss",
        "fragments": ("target * F.log_softmax(scores, dim=0)",),
    },
    "project_sequence_nll": {
        "object": "_mean_target_sequence_nll",
        "fragments": (
            "F.cross_entropy(prediction_logits.float(), target_tensor)",
            "torch.stack(losses).mean()",
        ),
    },
    "project_exact_gradient_clip": {
        "object": "clip_gradients",
        "fragments": (
            "max_norm / (norm + 1.0e-6)",
            "gradient * coefficient",
        ),
    },
    "project_exact_adamw_replay": {
        "object": "adamw_exact_delta",
        "fragments": (
            "next_exp_avg =",
            "next_exp_avg_sq =",
            "bias_correction1 =",
            "weight_decay_delta =",
            "total_delta =",
        ),
    },
    "project_lora_function_delta": {
        "object": "lora_function_delta",
        "fragments": (
            "b @ delta_a",
            "delta_b @ a",
            "delta_b @ delta_a",
            "(b + delta_b) @ (a + delta_a) - b @ a",
        ),
    },
    "torch_adamw_step": {
        "object": "AdamW.step",
        "fragments": (
            "params_with_grad",
            "exp_avgs",
            "exp_avg_sqs",
            "weight_decay=group[\"weight_decay\"]",
        ),
    },
    "transformers_linear_schedule": {
        "object": "get_linear_schedule_with_warmup",
        "fragments": (
            "_get_linear_schedule_with_warmup_lr_lambda",
            "num_warmup_steps",
            "num_training_steps",
            "LambdaLR",
        ),
    },
    "transformers_linear_schedule_lambda": {
        "object": "_get_linear_schedule_with_warmup_lr_lambda",
        "fragments": (
            "current_step < num_warmup_steps",
            "num_training_steps - current_step",
        ),
    },
}


FROZEN_TRAINING_SOURCE_FILES = {
    "AdamW.step": "torch/optim/adamw.py",
    "get_linear_schedule_with_warmup": "transformers/optimization.py",
    "_get_linear_schedule_with_warmup_lr_lambda": "transformers/optimization.py",
}


def audit_frozen_model_architecture(root: str | Path = ".") -> dict[str, Any]:
    """Bind exact-interface coverage to immutable model and training configs."""

    root_path = Path(root).resolve()
    paths = {
        "base_config": BASE_CONFIG,
        "q0_base_config": Q0_BASE_CONFIG,
        "q0_method_config": Q0_METHOD_CONFIG,
        "q1_method_config": Q1_METHOD_CONFIG,
        "q2_method_config": Q2_METHOD_CONFIG,
        "q3_method_config": Q3_METHOD_CONFIG,
        "q0_saved_config": Q0_SAVED_CONFIG,
        "q1_saved_config": Q1_SAVED_CONFIG,
        "q2_saved_config": Q2_SAVED_CONFIG,
        "q3_adapter_config": Q3_ADAPTER_CONFIG,
        "q0_training_metadata": Q0_TRAINING_METADATA,
        "q1_training_metadata": Q1_TRAINING_METADATA,
        "q2_training_metadata": Q2_TRAINING_METADATA,
        "q3_training_metadata": Q3_TRAINING_METADATA,
        **{
            f"identity_smoke_{method_id}": path
            for method_id, path in IDENTITY_SMOKES.items()
        },
        **{
            f"superseded_identity_smoke_{run_id}": path
            for run_id, path in SUPERSEDED_IDENTITY_SMOKES.items()
        },
    }
    failures = []
    loaded: dict[str, dict[str, Any]] = {}
    for key, relative in paths.items():
        path = root_path / relative
        try:
            loaded[key] = (
                _read_yaml(path) if path.suffix in {".yaml", ".yml"} else _read_json(path)
            )
        except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
            failures.append(f"{key} unreadable: {relative.as_posix()} ({exc})")

    if failures:
        return _failed_payload(root_path, paths, failures)

    base = loaded["base_config"]
    q0_base = loaded["q0_base_config"]
    q0_method = loaded["q0_method_config"]
    q1_method = loaded["q1_method_config"]
    q2_method = loaded["q2_method_config"]
    q3_method = loaded["q3_method_config"]
    q0_saved = loaded["q0_saved_config"]
    q1_saved = loaded["q1_saved_config"]
    q2_saved = loaded["q2_saved_config"]
    q3_adapter = loaded["q3_adapter_config"]
    q0_metadata = loaded["q0_training_metadata"]
    q1_metadata = loaded["q1_training_metadata"]
    q2_metadata = loaded["q2_training_metadata"]
    q3_metadata = loaded["q3_training_metadata"]

    for base_name, base_config in (("general base", base), ("Q0 base", q0_base)):
        _audit_topology(base_config, base_name, failures, saved=False)

    for saved_name, saved_config in (
        ("Q0 saved", q0_saved),
        ("Q1 saved", q1_saved),
        ("Q2 saved", q2_saved),
    ):
        _audit_topology(saved_config, saved_name, failures, saved=True)

    q0_model = _mapping(q0_method.get("model"), "Q0 model")
    q1_model = _mapping(q1_method.get("model"), "Q1 model")
    q2_model = _mapping(q2_method.get("model"), "Q2 model")
    q3_model = _mapping(q3_method.get("model"), "Q3 model")
    q0_training = _mapping(q0_method.get("training"), "Q0 training")
    q1_training = _mapping(q1_method.get("training"), "Q1 training")
    q2_training = _mapping(q2_method.get("training"), "Q2 training")
    q3_training = _mapping(q3_method.get("training"), "Q3 training")
    q0_method_body = _mapping(q0_method.get("method"), "Q0 method")
    q1_method_body = _mapping(q1_method.get("method"), "Q1 method")
    q2_method_body = _mapping(q2_method.get("method"), "Q2 method")
    q3_method_body = _mapping(q3_method.get("method"), "Q3 method")

    shared_model_expected = {
        "base_model_path": "models/huggingface/Qwen3-0.6B",
        "base_weights_sha256": "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b",
        "tokenizer_sha256": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        "base_artifact_manifest_sha256": "12a7e5ae8ecc02a88453fc1daec97853151f320a6ad1f242615594f2b8dd1663",
    }
    q0_model_expected = {
        "base_model_path": "models/huggingface/Qwen3-Reranker-0.6B",
        "base_weights_sha256": "27cd75a405b9c1b46b59abfd88aaa209e6fed2a1972cde9b70e7659537c5e65b",
        "tokenizer_sha256": shared_model_expected["tokenizer_sha256"],
        "base_artifact_manifest_sha256": "3727b9e53efc2cb5cb40cbc02fc722ed09998187c208930e3c0a3289bcdfbd78",
    }
    for model_name, model, expected_identity in (
        ("Q0", q0_model, q0_model_expected),
        ("Q1", q1_model, shared_model_expected),
        ("Q2", q2_model, shared_model_expected),
        ("Q3", q3_model, shared_model_expected),
    ):
        for key, expected in expected_identity.items():
            if model.get(key) != expected:
                failures.append(f"{model_name} frozen model mismatch: {key}")
    for model_name, model in (("Q0", q0_model), ("Q1", q1_model), ("Q2", q2_model)):
        if model.get("adaptation") != "full_parameter":
            failures.append(f"{model_name} adaptation is not full_parameter")
    if q3_model.get("adaptation") != "lora":
        failures.append("Q3 adaptation is not lora")

    expected_adapter = {
        "base_model_name_or_path": shared_model_expected["base_model_path"],
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "peft_type": "LORA",
        "peft_version": "0.19.1",
        "inference_mode": True,
    }
    for key, expected in expected_adapter.items():
        if q3_adapter.get(key) != expected:
            failures.append(f"Q3 adapter mismatch: {key}")
    if set(q3_adapter.get("target_modules", [])) != {"q_proj", "v_proj"}:
        failures.append("Q3 adapter target_modules mismatch")
    if q3_method_body.get("lora_targets") != ["q_proj", "v_proj"]:
        failures.append("Q3 method LoRA targets mismatch")

    for method_name, method, metadata, training, expected_identity in (
        ("Q0", q0_method, q0_metadata, q0_training, q0_model_expected),
        ("Q1", q1_method, q1_metadata, q1_training, shared_model_expected),
        ("Q2", q2_method, q2_metadata, q2_training, shared_model_expected),
        ("Q3", q3_method, q3_metadata, q3_training, shared_model_expected),
    ):
        if metadata.get("method_id") != method.get("method_id"):
            failures.append(f"{method_name} method_id mismatch")
        if metadata.get("base_model_path") != expected_identity["base_model_path"]:
            failures.append(f"{method_name} metadata base path mismatch")
        if metadata.get("base_weights_sha256") != expected_identity[
            "base_weights_sha256"
        ]:
            failures.append(f"{method_name} metadata base weights mismatch")
        metadata_training = metadata.get("training")
        if not isinstance(metadata_training, Mapping) or dict(metadata_training) != dict(
            training
        ):
            failures.append(f"{method_name} training metadata differs from config")
        if metadata.get("status") != "completed":
            failures.append(f"{method_name} frozen checkpoint is not completed")

    if q0_method_body.get("mechanism") != "specialized_reranker_pointwise_binary_cross_entropy":
        failures.append("Q0 method mechanism mismatch")
    if q1_method_body.get("mechanism") != "recommendation_and_search_instruction_templates_with_normalized_candidate_response_likelihood":
        failures.append("Q1 method mechanism mismatch")
    if q2_method_body.get("pairwise_loss_weight") != 0.5 or q2_method_body.get(
        "listwise_loss_weight"
    ) != 0.5:
        failures.append("Q2 registered loss weights mismatch")
    if q0_metadata.get("objective") != "pointwise_binary_cross_entropy_on_yes_no_logits":
        failures.append("Q0 objective metadata mismatch")
    if q1_metadata.get("objective") != "output_only_normalized_candidate_response_nll":
        failures.append("Q1 objective metadata mismatch")
    if q2_metadata.get("objective") != "0.5_ranknet_plus_0.5_tie_aware_listnet":
        failures.append("Q2 objective metadata mismatch")
    if q3_metadata.get("objective") != "output_only_yes_no_recommendation_alignment_nll_with_lora":
        failures.append("Q3 objective metadata mismatch")

    interface_ids = {
        str(row["interface_id"]) for row in TRANSFORMER_INTERFACE_INVENTORY
    }
    if not CONFIG_BACKED_INTERFACES.issubset(interface_ids):
        failures.append("config-backed interface is absent from exact inventory")
    dynamic_ids = set(DYNAMIC_INTERFACE_CONTRACTS)
    if CONFIG_BACKED_INTERFACES & dynamic_ids:
        failures.append("config and dynamic interface provenance overlap")
    if CONFIG_BACKED_INTERFACES | dynamic_ids != interface_ids:
        failures.append("exact interface implementation provenance is not exhaustive")

    expected_identity_nodes = {
        *(f"block_13.{node_id}" for node_id in BLOCK_NODE_IDS),
        *FINAL_NODE_IDS,
    }
    runtime_identity_smokes = []
    for method_id, relative in IDENTITY_SMOKES.items():
        smoke = loaded[f"identity_smoke_{method_id}"]
        smoke_failures = []
        expected = {
            "method_id": method_id,
            "status": "completed",
            "result_eligible": False,
            "actual_attention_backend": "sdpa",
            "identity_tolerance": 1e-5,
            "maximum_identity_error": 0.0,
            "capture_noop_max_abs_score_delta": 0.0,
            "attention_wrapper_noop_max_abs_score_delta": 0.0,
            "native_attention_wrapper_noop_max_abs_score_delta": 0.0,
            "algebra_recomposition_passed": True,
            "qrels_read": False,
            "source_test_opened": False,
        }
        for key, expected_value in expected.items():
            if smoke.get(key) != expected_value:
                smoke_failures.append(f"{key} mismatch")
        node_identity = smoke.get("node_identity_max_abs_score_delta")
        if not isinstance(node_identity, Mapping) or set(node_identity) != expected_identity_nodes:
            smoke_failures.append("node identity coverage mismatch")
        elif any(value != 0.0 for value in node_identity.values()):
            smoke_failures.append("node identity is not exact")
        algebra_error = smoke.get("algebra_max_abs_error")
        algebra_allowed = smoke.get("algebra_max_allowed_error")
        if (
            not isinstance(algebra_error, Mapping)
            or not isinstance(algebra_allowed, Mapping)
            or set(algebra_error) != set(algebra_allowed)
            or any(
                float(algebra_error[key]) > float(algebra_allowed[key])
                for key in algebra_error
            )
        ):
            smoke_failures.append("algebra recomposition bounds mismatch")
        if smoke.get("detailed_blocks") != [13, 20, 27]:
            smoke_failures.append("fixed detailed blocks mismatch")
        if smoke.get("patched_node_count") != len(expected_identity_nodes):
            smoke_failures.append("patched node count mismatch")
        if smoke_failures:
            failures.extend(
                f"identity smoke {method_id}: {message}"
                for message in smoke_failures
            )
        runtime_identity_smokes.append(
            {
                "method_id": method_id,
                "path": relative.as_posix(),
                "sha256": sha256_file(root_path / relative),
                "status": smoke.get("status"),
                "attention_backend": smoke.get("actual_attention_backend"),
                "identity_tolerance": smoke.get("identity_tolerance"),
                "maximum_identity_error": smoke.get("maximum_identity_error"),
                "hook_nodes_validated": len(node_identity)
                if isinstance(node_identity, Mapping)
                else 0,
                "algebra_recomposition_passed": smoke.get(
                    "algebra_recomposition_passed"
                ),
                "native_attention_wrapper_noop_max_abs_score_delta": smoke.get(
                    "native_attention_wrapper_noop_max_abs_score_delta"
                ),
                "qrels_read": smoke.get("qrels_read"),
                "source_test_opened": smoke.get("source_test_opened"),
                "failures": smoke_failures,
            }
        )

    superseded_runtime_lineage = []
    for run_id, relative in SUPERSEDED_IDENTITY_SMOKES.items():
        smoke = loaded[f"superseded_identity_smoke_{run_id}"]
        lineage_failures = []
        expected = {
            "run_id": run_id,
            "method_id": "q1_instructrec_generalqwen",
            "status": "failed_identity",
            "result_eligible": False,
            "qrels_read": False,
            "source_test_opened": False,
        }
        for key, expected_value in expected.items():
            if smoke.get(key) != expected_value:
                lineage_failures.append(f"{key} mismatch")
        tolerance = smoke.get("identity_tolerance")
        maximum_error = smoke.get("maximum_identity_error")
        wrapper_error = smoke.get("native_attention_wrapper_noop_max_abs_score_delta")
        if not all(isinstance(value, (int, float)) for value in (tolerance, maximum_error, wrapper_error)):
            lineage_failures.append("identity failure values invalid")
        elif not float(maximum_error) > float(tolerance) or float(wrapper_error) != float(
            maximum_error
        ):
            lineage_failures.append("identity failure is not reproduced mechanically")
        if lineage_failures:
            failures.extend(
                f"superseded identity smoke {run_id}: {message}"
                for message in lineage_failures
            )
        superseded_runtime_lineage.append(
            {
                "run_id": run_id,
                "method_id": smoke.get("method_id"),
                "path": relative.as_posix(),
                "sha256": sha256_file(root_path / relative),
                "status": smoke.get("status"),
                "maximum_identity_error": maximum_error,
                "identity_tolerance": tolerance,
                "failure_kind": "native_attention_wrapper_changed_q1_execution",
                "canonical_replacement": IDENTITY_SMOKES[
                    "q1_instructrec_generalqwen"
                ].as_posix(),
                "retained_as_mechanical_lineage": True,
                "scientific_result_eligible": False,
                "failures": lineage_failures,
            }
        )

    dynamic_interface_contracts = []
    for interface_id, contract in sorted(DYNAMIC_INTERFACE_CONTRACTS.items()):
        contract_failures = []
        source_identities = []
        for relative in contract["source_paths"]:
            source_path = root_path / relative
            if not source_path.is_file():
                contract_failures.append(f"source missing: {relative.as_posix()}")
                source_sha = None
            else:
                source_sha = sha256_file(source_path)
            source_identities.append(
                {"path": relative.as_posix(), "sha256": source_sha}
            )
        runtime_node = contract["runtime_identity_node"]
        runtime_models = []
        if runtime_node is not None:
            expected_runtime_key = f"block_13.{runtime_node}"
            for method_id in IDENTITY_SMOKES:
                smoke = loaded[f"identity_smoke_{method_id}"]
                node_identity = smoke.get("node_identity_max_abs_score_delta")
                if (
                    not isinstance(node_identity, Mapping)
                    or node_identity.get(expected_runtime_key) != 0.0
                ):
                    contract_failures.append(
                        f"runtime identity missing/nonzero: {method_id}/{expected_runtime_key}"
                    )
                else:
                    runtime_models.append(method_id)
        runtime_algebra_key = contract.get("runtime_algebra_key")
        runtime_algebra_models = []
        if runtime_algebra_key is not None:
            for method_id in IDENTITY_SMOKES:
                smoke = loaded[f"identity_smoke_{method_id}"]
                algebra_error = smoke.get("algebra_max_abs_error")
                algebra_allowed = smoke.get("algebra_max_allowed_error")
                if (
                    not isinstance(algebra_error, Mapping)
                    or not isinstance(algebra_allowed, Mapping)
                    or runtime_algebra_key not in algebra_error
                    or runtime_algebra_key not in algebra_allowed
                    or float(algebra_error[runtime_algebra_key])
                    > float(algebra_allowed[runtime_algebra_key])
                ):
                    contract_failures.append(
                        "runtime algebra missing/out-of-bound: "
                        f"{method_id}/{runtime_algebra_key}"
                    )
                else:
                    runtime_algebra_models.append(method_id)
        if contract_failures:
            failures.extend(
                f"dynamic interface {interface_id}: {message}"
                for message in contract_failures
            )
        dynamic_interface_contracts.append(
            {
                "interface_id": interface_id,
                "binding_kind": contract["binding_kind"],
                "source_identities": source_identities,
                "runtime_identity_node": runtime_node,
                "runtime_identity_models": sorted(runtime_models),
                "runtime_algebra_key": runtime_algebra_key,
                "runtime_algebra_models": sorted(runtime_algebra_models),
                "status": "completed" if not contract_failures else "failed",
                "failures": contract_failures,
                "scientific_support_inferred": False,
            }
        )

    forward_graph = _audit_forward_primitive_coverage(
        interface_ids=interface_ids,
        base_configs={"general_qwen3": base, "q0_reranker_qwen3": q0_base},
        runtime_identity_smokes=runtime_identity_smokes,
        frozen_runtime_metadata=q3_metadata,
    )
    failures.extend(
        f"frozen forward graph: {message}" for message in forward_graph["failures"]
    )
    training_graph = _audit_training_primitive_coverage(
        root_path=root_path,
        interface_ids=interface_ids,
        training_configs={
            MODEL_IDS[0]: q0_training,
            MODEL_IDS[1]: q1_training,
            MODEL_IDS[2]: q2_training,
            MODEL_IDS[3]: q3_training,
        },
        q3_adapter=q3_adapter,
        frozen_runtime_metadata=q3_metadata,
    )
    failures.extend(
        f"frozen training graph: {message}"
        for message in training_graph["failures"]
    )

    q3_trainable = _mapping(q3_metadata.get("trainable_parameters"), "Q3 trainable")
    total_parameters = q3_trainable.get("total")
    trainable_parameters = q3_trainable.get("trainable")
    if type(total_parameters) is not int or type(trainable_parameters) is not int:
        failures.append("Q3 trainable parameter counts invalid")
        trainable_fraction = None
    else:
        trainable_fraction = trainable_parameters / total_parameters

    return {
        "schema_version": 1,
        "analysis_type": "frozen_qwen_model_architecture_audit",
        "status": "completed" if not failures else "failed",
        "failures": failures,
        "frozen_topology": {
            **EXPECTED_BASE_TOPOLOGY,
            "query_heads_per_kv_head": (
                EXPECTED_BASE_TOPOLOGY["num_attention_heads"]
                // EXPECTED_BASE_TOPOLOGY["num_key_value_heads"]
            ),
            "mlp_expansion_ratio": (
                EXPECTED_BASE_TOPOLOGY["intermediate_size"]
                / EXPECTED_BASE_TOPOLOGY["hidden_size"]
            ),
        },
        "frozen_base_artifacts": {
            "qwen3_reranker_06b": dict(q0_model_expected),
            "qwen3_general_06b": dict(shared_model_expected),
        },
        "frozen_base_artifact_count": 2,
        "model_pathways": {
            "q0_qwen3_reranker_06b": {
                "base_artifact": "qwen3_reranker_06b",
                "adaptation": "full_parameter",
                "objective": q0_metadata.get("objective"),
                "optimizer_steps": q0_metadata.get("progress", {}).get(
                    "optimizer_steps"
                ),
            },
            "q1_instructrec_generalqwen": {
                "base_artifact": "qwen3_general_06b",
                "adaptation": "full_parameter",
                "objective": q1_metadata.get("objective"),
                "optimizer_steps": q1_metadata.get("progress", {}).get(
                    "optimizer_steps"
                ),
            },
            "q2_recranker_generalqwen": {
                "base_artifact": "qwen3_general_06b",
                "adaptation": "full_parameter",
                "objective": q2_metadata.get("objective"),
                "optimizer_steps": q2_metadata.get("progress", {}).get(
                    "optimizer_steps"
                ),
            },
            "q3_tallrec_generalqwen": {
                "base_artifact": "qwen3_general_06b",
                "adaptation": "lora",
                "lora_rank": q3_adapter.get("r"),
                "lora_alpha": q3_adapter.get("lora_alpha"),
                "lora_dropout": q3_adapter.get("lora_dropout"),
                "lora_targets": sorted(q3_adapter.get("target_modules", [])),
                "trainable_parameters": trainable_parameters,
                "total_parameters": total_parameters,
                "trainable_fraction": trainable_fraction,
                "objective": q3_metadata.get("objective"),
                "optimizer_steps": q3_metadata.get("progress", {}).get(
                    "optimizer_steps"
                ),
            },
        },
        "frozen_model_pathway_count": 4,
        "config_backed_interface_count": len(CONFIG_BACKED_INTERFACES),
        "config_backed_interfaces": sorted(CONFIG_BACKED_INTERFACES),
        "dynamic_runtime_or_source_backed_interface_count": len(
            dynamic_interface_contracts
        ),
        "dynamic_runtime_or_source_backed_interfaces": dynamic_interface_contracts,
        "exact_interface_inventory_count": len(interface_ids),
        "config_backed_interfaces_present_in_inventory": (
            CONFIG_BACKED_INTERFACES.issubset(interface_ids)
        ),
        "implementation_provenance_covered_interface_count": (
            len(CONFIG_BACKED_INTERFACES)
            + sum(row["status"] == "completed" for row in dynamic_interface_contracts)
        ),
        "all_exact_interfaces_have_config_or_runtime_source_provenance": (
            CONFIG_BACKED_INTERFACES | dynamic_ids == interface_ids
            and all(row["status"] == "completed" for row in dynamic_interface_contracts)
        ),
        "forward_primitive_count": forward_graph["forward_primitive_count"],
        "forward_primitives": forward_graph["forward_primitives"],
        "forward_inference_interface_count": forward_graph[
            "forward_inference_interface_count"
        ],
        "forward_mapped_interface_count": forward_graph[
            "forward_mapped_interface_count"
        ],
        "forward_missing_interface_ids": forward_graph[
            "forward_missing_interface_ids"
        ],
        "forward_extraneous_interface_ids": forward_graph[
            "forward_extraneous_interface_ids"
        ],
        "forward_training_interface_count": forward_graph[
            "forward_training_interface_count"
        ],
        "forward_training_interfaces_excluded_by_design": forward_graph[
            "forward_training_interfaces_excluded_by_design"
        ],
        "forward_primitive_interface_coverage_complete": forward_graph[
            "forward_primitive_interface_coverage_complete"
        ],
        "forward_source_binding_count": forward_graph[
            "forward_source_binding_count"
        ],
        "forward_source_bindings": forward_graph["forward_source_bindings"],
        "transformers_version": forward_graph["transformers_version"],
        "forward_peft_version": forward_graph["peft_version"],
        "forward_frozen_python_executable": forward_graph[
            "frozen_python_executable"
        ],
        "forward_source_environment_is_frozen_checkpoint_environment": forward_graph[
            "source_environment_is_frozen_checkpoint_environment"
        ],
        "inactive_architecture_path_count": forward_graph[
            "inactive_architecture_path_count"
        ],
        "inactive_architecture_paths": forward_graph[
            "inactive_architecture_paths"
        ],
        "all_inactive_architecture_paths_verified": forward_graph[
            "all_inactive_architecture_paths_verified"
        ],
        "forward_graph_failures": forward_graph["failures"],
        "forward_coverage_is_semantic_primitive_census": True,
        "forward_coverage_is_kernel_instruction_census": False,
        "operator_attribution_inferred_from_forward_coverage": False,
        "training_primitive_count": training_graph["training_primitive_count"],
        "training_primitives": training_graph["training_primitives"],
        "training_exact_interface_count": training_graph[
            "training_exact_interface_count"
        ],
        "training_mapped_interface_count": training_graph[
            "training_mapped_interface_count"
        ],
        "training_missing_interface_ids": training_graph[
            "training_missing_interface_ids"
        ],
        "training_extraneous_interface_ids": training_graph[
            "training_extraneous_interface_ids"
        ],
        "training_nontraining_interface_count": training_graph[
            "training_nontraining_interface_count"
        ],
        "training_nontraining_interfaces_excluded_by_design": training_graph[
            "training_nontraining_interfaces_excluded_by_design"
        ],
        "training_primitive_interface_coverage_complete": training_graph[
            "training_primitive_interface_coverage_complete"
        ],
        "training_source_binding_count": training_graph[
            "training_source_binding_count"
        ],
        "training_source_bindings": training_graph["training_source_bindings"],
        "training_artifact_binding_count": training_graph[
            "training_artifact_binding_count"
        ],
        "training_artifact_bindings": training_graph[
            "training_artifact_bindings"
        ],
        "training_torch_version": training_graph["torch_version"],
        "training_transformers_version": training_graph["transformers_version"],
        "training_peft_version": training_graph["peft_version"],
        "frozen_training_hyperparameters": training_graph[
            "frozen_training_hyperparameters"
        ],
        "inactive_training_path_count": training_graph[
            "inactive_training_path_count"
        ],
        "inactive_training_paths": training_graph["inactive_training_paths"],
        "all_inactive_training_paths_verified": training_graph[
            "all_inactive_training_paths_verified"
        ],
        "training_update_graph_failures": training_graph["failures"],
        "training_coverage_is_single_step_semantic_primitive_census": True,
        "training_coverage_is_multiseed_causal_attribution": False,
        "operator_attribution_inferred_from_training_coverage": False,
        "runtime_identity_smoke_count": len(runtime_identity_smokes),
        "runtime_identity_smokes": runtime_identity_smokes,
        "runtime_hook_node_count": len(expected_identity_nodes),
        "runtime_attention_backend": "sdpa",
        "runtime_identity_and_recomposition_validated": not any(
            row["failures"] for row in runtime_identity_smokes
        ),
        "superseded_runtime_lineage_count": len(superseded_runtime_lineage),
        "superseded_runtime_lineage": superseded_runtime_lineage,
        "superseded_runtime_lineage_retained": all(
            row["retained_as_mechanical_lineage"]
            and row["scientific_result_eligible"] is False
            for row in superseded_runtime_lineage
        ),
        "files": {
            key: {
                "path": relative.as_posix(),
                "sha256": sha256_file(root_path / relative),
            }
            for key, relative in paths.items()
        },
        "scientific_effect_values_read": False,
        "scientific_support_inferred": False,
        "qrels_read_by_audit": False,
        "source_test_opened": False,
    }


def _audit_forward_primitive_coverage(
    *,
    interface_ids: set[str],
    base_configs: Mapping[str, Mapping[str, Any]],
    runtime_identity_smokes: list[dict[str, Any]],
    frozen_runtime_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    failures: list[str] = []
    inventory_by_id = {
        str(row["interface_id"]): row for row in TRANSFORMER_INTERFACE_INVENTORY
    }
    inference_ids = {
        interface_id
        for interface_id, row in inventory_by_id.items()
        if row["system_layer"] != "training"
    }
    training_ids = interface_ids - inference_ids
    primitive_ids = [str(row["primitive_id"]) for row in FORWARD_PRIMITIVE_CONTRACTS]
    if len(primitive_ids) != len(set(primitive_ids)):
        failures.append("forward primitive IDs are not unique")
    execution_orders = [
        int(row["execution_order"]) for row in FORWARD_PRIMITIVE_CONTRACTS
    ]
    if execution_orders != list(range(1, len(FORWARD_PRIMITIVE_CONTRACTS) + 1)):
        failures.append("forward primitive execution order is not contiguous")

    mapped_ids: set[str] = set()
    primitive_rows = []
    for contract in FORWARD_PRIMITIVE_CONTRACTS:
        primitive_failures = []
        mapped = {str(item) for item in contract["interface_ids"]}
        model_scope = {str(item) for item in contract["model_scope"]}
        if not mapped:
            primitive_failures.append("no exact interface mapping")
        if mapped - interface_ids:
            primitive_failures.append(
                "unknown exact interface mapping: "
                + ",".join(sorted(mapped - interface_ids))
            )
        if mapped & training_ids:
            primitive_failures.append(
                "training interface mapped into inference-forward census"
            )
        if not model_scope or not model_scope.issubset(MODEL_IDS):
            primitive_failures.append("invalid frozen model scope")
        mapped_ids.update(mapped)
        if primitive_failures:
            failures.extend(
                f"primitive {contract['primitive_id']}: {message}"
                for message in primitive_failures
            )
        primitive_rows.append(
            {
                "primitive_id": contract["primitive_id"],
                "execution_order": contract["execution_order"],
                "implementation_step": contract["implementation_step"],
                "interface_ids": sorted(mapped),
                "model_scope": sorted(model_scope),
                "status": "completed" if not primitive_failures else "failed",
                "failures": primitive_failures,
                "scientific_support_inferred": False,
                "operator_attribution_inferred": False,
            }
        )

    missing_ids = sorted(inference_ids - mapped_ids)
    extraneous_ids = sorted(mapped_ids - inference_ids)
    if missing_ids:
        failures.append(
            "inference exact interfaces missing from forward graph: "
            + ",".join(missing_ids)
        )
    if extraneous_ids:
        failures.append(
            "non-inference interfaces present in forward graph: "
            + ",".join(extraneous_ids)
        )

    source_audit = _audit_frozen_forward_sources(frozen_runtime_metadata)
    failures.extend(source_audit["failures"])
    inactive_paths = _audit_inactive_architecture_paths(
        base_configs=base_configs,
        runtime_identity_smokes=runtime_identity_smokes,
        source_bindings=source_audit["bindings"],
    )
    failures.extend(inactive_paths["failures"])
    return {
        "forward_primitive_count": len(primitive_rows),
        "forward_primitives": primitive_rows,
        "forward_inference_interface_count": len(inference_ids),
        "forward_mapped_interface_count": len(mapped_ids & inference_ids),
        "forward_missing_interface_ids": missing_ids,
        "forward_extraneous_interface_ids": extraneous_ids,
        "forward_training_interface_count": len(training_ids),
        "forward_training_interfaces_excluded_by_design": sorted(training_ids),
        "forward_primitive_interface_coverage_complete": not failures,
        "forward_source_binding_count": len(source_audit["bindings"]),
        "forward_source_bindings": source_audit["bindings"],
        "transformers_version": source_audit["transformers_version"],
        "peft_version": source_audit["peft_version"],
        "frozen_python_executable": source_audit["python_executable"],
        "source_environment_is_frozen_checkpoint_environment": source_audit[
            "source_environment_is_frozen_checkpoint_environment"
        ],
        "inactive_architecture_path_count": len(inactive_paths["rows"]),
        "inactive_architecture_paths": inactive_paths["rows"],
        "all_inactive_architecture_paths_verified": not inactive_paths["failures"],
        "failures": failures,
    }


def _audit_training_primitive_coverage(
    *,
    root_path: Path,
    interface_ids: set[str],
    training_configs: Mapping[str, Mapping[str, Any]],
    q3_adapter: Mapping[str, Any],
    frozen_runtime_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    failures: list[str] = []
    inventory_by_id = {
        str(row["interface_id"]): row for row in TRANSFORMER_INTERFACE_INVENTORY
    }
    training_ids = {
        interface_id
        for interface_id, row in inventory_by_id.items()
        if row["system_layer"] == "training"
    }
    nontraining_ids = interface_ids - training_ids
    primitive_ids = [
        str(row["primitive_id"]) for row in TRAINING_PRIMITIVE_CONTRACTS
    ]
    if len(primitive_ids) != len(set(primitive_ids)):
        failures.append("training primitive IDs are not unique")
    execution_orders = [
        int(row["execution_order"]) for row in TRAINING_PRIMITIVE_CONTRACTS
    ]
    if execution_orders != list(range(1, len(TRAINING_PRIMITIVE_CONTRACTS) + 1)):
        failures.append("training primitive execution order is not contiguous")

    mapped_ids: set[str] = set()
    primitive_rows = []
    for contract in TRAINING_PRIMITIVE_CONTRACTS:
        primitive_failures = []
        mapped = {str(item) for item in contract["interface_ids"]}
        model_scope = {str(item) for item in contract["model_scope"]}
        if not mapped:
            primitive_failures.append("no exact interface mapping")
        if mapped - interface_ids:
            primitive_failures.append(
                "unknown exact interface mapping: "
                + ",".join(sorted(mapped - interface_ids))
            )
        if mapped & nontraining_ids:
            primitive_failures.append(
                "non-training interface mapped into training-update census"
            )
        if not model_scope or not model_scope.issubset(MODEL_IDS):
            primitive_failures.append("invalid frozen model scope")
        mapped_ids.update(mapped)
        if primitive_failures:
            failures.extend(
                f"primitive {contract['primitive_id']}: {message}"
                for message in primitive_failures
            )
        primitive_rows.append(
            {
                "primitive_id": contract["primitive_id"],
                "execution_order": contract["execution_order"],
                "implementation_step": contract["implementation_step"],
                "interface_ids": sorted(mapped),
                "model_scope": sorted(model_scope),
                "status": "completed" if not primitive_failures else "failed",
                "failures": primitive_failures,
                "scientific_support_inferred": False,
                "operator_attribution_inferred": False,
            }
        )

    missing_ids = sorted(training_ids - mapped_ids)
    extraneous_ids = sorted(mapped_ids - training_ids)
    if missing_ids:
        failures.append(
            "training exact interfaces missing from update graph: "
            + ",".join(missing_ids)
        )
    if extraneous_ids:
        failures.append(
            "non-training interfaces present in update graph: "
            + ",".join(extraneous_ids)
        )

    expected_training = {
        MODEL_IDS[0]: {
            "gradient_accumulation_steps": 2,
            "learning_rate": 1.0e-5,
        },
        MODEL_IDS[1]: {
            "gradient_accumulation_steps": 8,
            "learning_rate": 1.0e-5,
        },
        MODEL_IDS[2]: {
            "gradient_accumulation_steps": 16,
            "learning_rate": 1.0e-5,
        },
        MODEL_IDS[3]: {
            "gradient_accumulation_steps": 8,
            "learning_rate": 2.0e-4,
        },
    }
    shared_expected = {
        "dtype": "bfloat16",
        "epochs": 1,
        "gradient_checkpointing": True,
        "history_dropout_probability": 0.0,
        "max_grad_norm": 1.0,
        "seed": 20260714,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
    }
    frozen_training_hyperparameters = {}
    if set(training_configs) != set(MODEL_IDS):
        failures.append("training config model scope is not exactly Q0--Q3")
    for model_id in MODEL_IDS:
        config = training_configs.get(model_id, {})
        expected = {**shared_expected, **expected_training[model_id]}
        local_failures = [
            f"{key} differs from {value!r}"
            for key, value in expected.items()
            if config.get(key) != value
        ]
        if local_failures:
            failures.extend(
                f"training config {model_id}: {message}" for message in local_failures
            )
        frozen_training_hyperparameters[model_id] = {
            key: config.get(key) for key in sorted(expected)
        }

    source_audit = _audit_training_update_sources(frozen_runtime_metadata)
    failures.extend(source_audit["failures"])
    artifact_audit = _audit_training_artifact_bindings(
        root_path=root_path,
        q3_adapter=q3_adapter,
        frozen_runtime_metadata=frozen_runtime_metadata,
    )
    failures.extend(artifact_audit["failures"])
    inactive_paths = _audit_inactive_training_paths(
        training_configs=training_configs,
        q3_adapter=q3_adapter,
        frozen_runtime_metadata=frozen_runtime_metadata,
        source_bindings=source_audit["bindings"],
    )
    failures.extend(inactive_paths["failures"])
    return {
        "training_primitive_count": len(primitive_rows),
        "training_primitives": primitive_rows,
        "training_exact_interface_count": len(training_ids),
        "training_mapped_interface_count": len(mapped_ids & training_ids),
        "training_missing_interface_ids": missing_ids,
        "training_extraneous_interface_ids": extraneous_ids,
        "training_nontraining_interface_count": len(nontraining_ids),
        "training_nontraining_interfaces_excluded_by_design": sorted(
            nontraining_ids
        ),
        "training_primitive_interface_coverage_complete": not failures,
        "training_source_binding_count": len(source_audit["bindings"]),
        "training_source_bindings": source_audit["bindings"],
        "training_artifact_binding_count": len(artifact_audit["bindings"]),
        "training_artifact_bindings": artifact_audit["bindings"],
        "torch_version": source_audit["torch_version"],
        "transformers_version": source_audit["transformers_version"],
        "peft_version": q3_adapter.get("peft_version"),
        "frozen_training_hyperparameters": frozen_training_hyperparameters,
        "inactive_training_path_count": len(inactive_paths["rows"]),
        "inactive_training_paths": inactive_paths["rows"],
        "all_inactive_training_paths_verified": not inactive_paths["failures"],
        "failures": failures,
    }


def _audit_training_update_sources(
    frozen_runtime_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    failures: list[str] = []
    bindings = []
    try:
        from myrec.baselines import motivation_v12_ranker as ranker
        from myrec.mechanism import optimizer_replay_math
    except (ImportError, AttributeError) as exc:
        return {
            "torch_version": None,
            "transformers_version": None,
            "bindings": [],
            "failures": [f"training update source unavailable: {exc}"],
        }

    objects = {
        "train_motivation_v12_ranker": ranker.train_motivation_v12_ranker,
        "_training_batch_loss": ranker._training_batch_loss,
        "_load_model_and_tokenizer": ranker._load_model_and_tokenizer,
        "pairwise_ranknet_loss": ranker.pairwise_ranknet_loss,
        "listwise_softmax_loss": ranker.listwise_softmax_loss,
        "_mean_target_sequence_nll": ranker._mean_target_sequence_nll,
        "clip_gradients": optimizer_replay_math.clip_gradients,
        "adamw_exact_delta": optimizer_replay_math.adamw_exact_delta,
        "lora_function_delta": optimizer_replay_math.lora_function_delta,
    }
    package_versions = frozen_runtime_metadata.get("package_versions")
    if not isinstance(package_versions, Mapping):
        package_versions = {}
        failures.append("frozen training package versions are missing")
    torch_version = package_versions.get("torch")
    transformers_version = package_versions.get("transformers")
    if torch_version != "2.6.0+cu124":
        failures.append("frozen training Torch version is not 2.6.0+cu124")
    if transformers_version != "5.12.1":
        failures.append("frozen training Transformers version is not 5.12.1")
    python_executable = frozen_runtime_metadata.get("python_executable")
    if not isinstance(python_executable, str) or not python_executable:
        python_executable = None
        failures.append("frozen training Python executable is missing")
    frozen_source_cache: dict[Path, str] = {}

    for source_id, contract in TRAINING_SOURCE_SENTINELS.items():
        source_failures = []
        object_path = str(contract["object"])
        relative_source = FROZEN_TRAINING_SOURCE_FILES.get(object_path)
        runtime_package = "project"
        runtime_package_version = None
        if relative_source is None:
            obj = objects[object_path]
            try:
                source = inspect.getsource(obj)
                source_file_value = inspect.getsourcefile(obj)
            except (OSError, TypeError) as exc:
                source = ""
                source_file_value = None
                source_failures.append(f"source unreadable: {exc}")
            source_path = Path(source_file_value) if source_file_value else None
        else:
            runtime_package = (
                "torch" if relative_source.startswith("torch/") else "transformers"
            )
            runtime_package_version = package_versions.get(runtime_package)
            source_path = _find_frozen_site_package_source(
                python_executable, relative_source
            )
            if source_path is None:
                source = ""
                source_failures.append(
                    f"frozen package source unavailable: {relative_source}"
                )
            else:
                if source_path not in frozen_source_cache:
                    try:
                        frozen_source_cache[source_path] = source_path.read_text(
                            encoding="utf-8"
                        )
                    except (OSError, UnicodeError) as exc:
                        frozen_source_cache[source_path] = ""
                        source_failures.append(f"source file unreadable: {exc}")
                try:
                    source = _extract_python_object_source(
                        frozen_source_cache[source_path], object_path
                    )
                except (SyntaxError, ValueError) as exc:
                    source = ""
                    source_failures.append(f"object source unreadable: {exc}")
        missing_fragments = [
            fragment for fragment in contract["fragments"] if fragment not in source
        ]
        if missing_fragments:
            source_failures.append(
                "required execution fragments missing: "
                + ",".join(missing_fragments)
            )
        forbidden_fragments = [
            fragment
            for fragment in contract.get("forbidden_fragments", ())
            if fragment in source
        ]
        if forbidden_fragments:
            source_failures.append(
                "forbidden execution fragments present: "
                + ",".join(forbidden_fragments)
            )
        if source_path is None or not source_path.is_file():
            source_failures.append("source file identity unavailable")
            source_file_sha256 = None
            package_relative_path = None
        else:
            source_file_sha256 = sha256_file(source_path)
            package_relative_path = _package_relative_source_path(source_path)
        if source_failures:
            failures.extend(
                f"training source {source_id}: {message}"
                for message in source_failures
            )
        bindings.append(
            {
                "source_id": source_id,
                "object": object_path,
                "runtime_package": runtime_package,
                "runtime_package_version": runtime_package_version,
                "package_relative_path": package_relative_path,
                "source_file_sha256": source_file_sha256,
                "object_source_sha256": hashlib.sha256(
                    source.encode("utf-8")
                ).hexdigest()
                if source
                else None,
                "required_fragment_count": len(contract["fragments"]),
                "missing_fragments": missing_fragments,
                "forbidden_fragments_present": forbidden_fragments,
                "status": "completed" if not source_failures else "failed",
                "failures": source_failures,
                "scientific_support_inferred": False,
                "operator_attribution_inferred": False,
            }
        )
    return {
        "torch_version": torch_version,
        "transformers_version": transformers_version,
        "bindings": bindings,
        "failures": failures,
    }


def _audit_inactive_training_paths(
    *,
    training_configs: Mapping[str, Mapping[str, Any]],
    q3_adapter: Mapping[str, Any],
    frozen_runtime_metadata: Mapping[str, Any],
    source_bindings: list[dict[str, Any]],
) -> dict[str, Any]:
    """Prove which tempting training branches are absent from the frozen recipes."""

    rows = []
    failures: list[str] = []

    def add_row(
        path_id: str,
        evidence_kind: str,
        observed: Any,
        expected: Any,
        verified: bool,
        explanation: str,
    ) -> None:
        local_failures = [] if verified else ["inactive training path check failed"]
        if local_failures:
            failures.extend(
                f"inactive training path {path_id}: {message}"
                for message in local_failures
            )
        rows.append(
            {
                "path_id": path_id,
                "evidence_kind": evidence_kind,
                "observed": observed,
                "expected": expected,
                "inactive_verified": verified,
                "explanation": explanation,
                "failures": local_failures,
                "scientific_support_inferred": False,
                "operator_attribution_inferred": False,
            }
        )

    dtype_observed = {
        model_id: config.get("dtype") for model_id, config in training_configs.items()
    }
    add_row(
        "fp16_autocast_and_dynamic_loss_scaling",
        "four_model_training_config_and_project_loop_source",
        dtype_observed,
        {model_id: "bfloat16" for model_id in MODEL_IDS},
        set(dtype_observed) == set(MODEL_IDS)
        and all(value == "bfloat16" for value in dtype_observed.values()),
        "The FP16 autocast and enabled GradScaler branch is inactive; the frozen path is BF16 autocast with identity scaling.",
    )
    history_dropout_observed = {
        model_id: config.get("history_dropout_probability")
        for model_id, config in training_configs.items()
    }
    add_row(
        "history_dropout_augmentation",
        "four_model_training_config",
        history_dropout_observed,
        {model_id: 0.0 for model_id in MODEL_IDS},
        set(history_dropout_observed) == set(MODEL_IDS)
        and all(value == 0.0 for value in history_dropout_observed.values()),
        "No training recipe stochastically removes the serialized user history; LoRA dropout remains a separate internal Q3 mechanism.",
    )
    checkpoint_observed = {
        model_id: config.get("gradient_checkpointing")
        for model_id, config in training_configs.items()
    }
    loader_binding = next(
        (
            row
            for row in source_bindings
            if row["source_id"] == "project_model_training_loader"
        ),
        None,
    )
    add_row(
        "reentrant_gradient_checkpoint_engine",
        "four_model_training_config_and_project_loader_source",
        {
            "enabled": checkpoint_observed,
            "loader_source_status": loader_binding.get("status")
            if loader_binding
            else None,
        },
        {"enabled": True, "use_reentrant": False},
        set(checkpoint_observed) == set(MODEL_IDS)
        and all(value is True for value in checkpoint_observed.values())
        and loader_binding is not None
        and loader_binding["status"] == "completed",
        "Activation checkpointing is active, but the legacy reentrant engine is not; every frozen model requests use_reentrant=False.",
    )
    trainable = frozen_runtime_metadata.get("trainable_parameters")
    q3_trainable_observed = dict(trainable) if isinstance(trainable, Mapping) else {}
    add_row(
        "q3_full_parameter_optimization",
        "q3_frozen_training_metadata_and_adapter_targets",
        {
            "parameter_counts": q3_trainable_observed,
            "target_modules": sorted(q3_adapter.get("target_modules", [])),
        },
        {
            "total": 597_196_800,
            "trainable": 1_146_880,
            "target_modules": ["q_proj", "v_proj"],
        },
        q3_trainable_observed
        == {"total": 597_196_800, "trainable": 1_146_880}
        and sorted(q3_adapter.get("target_modules", [])) == ["q_proj", "v_proj"],
        "Q3 does not update the frozen base-model parameters; only q/v LoRA A/B tensors are trainable.",
    )
    lora_bias_observed = {
        "bias": q3_adapter.get("bias"),
        "lora_bias": q3_adapter.get("lora_bias"),
    }
    add_row(
        "q3_lora_bias_parameters",
        "q3_frozen_adapter_config",
        lora_bias_observed,
        {"bias": "none", "lora_bias": False},
        lora_bias_observed == {"bias": "none", "lora_bias": False},
        "No base or adapter bias parameter is introduced into the Q3 q/v LoRA branches.",
    )
    variant_observed = {
        key: q3_adapter.get(key)
        for key in (
            "alora_invocation_tokens",
            "use_bdlora",
            "use_dora",
            "use_qalora",
            "use_rslora",
        )
    }
    variant_expected = {
        "alora_invocation_tokens": None,
        "use_bdlora": None,
        "use_dora": False,
        "use_qalora": False,
        "use_rslora": False,
    }
    add_row(
        "q3_nonvanilla_lora_variants",
        "q3_frozen_adapter_config",
        variant_observed,
        variant_expected,
        variant_observed == variant_expected,
        "DoRA, RSLoRA, QALoRA, BDLora, and activated-LoRA invocation branches are all inactive; the checkpoint uses vanilla alpha/r LoRA.",
    )
    source_by_id = {str(row["source_id"]): row for row in source_bindings}
    forward_loader = source_by_id.get("project_model_training_loader")
    add_row(
        "q3_merged_or_disabled_adapter_training",
        "project_loader_and_frozen_peft_config",
        {
            "loader_status": forward_loader.get("status")
            if forward_loader
            else None,
            "adapter_inference_mode": q3_adapter.get("inference_mode"),
        },
        {"loader_status": "completed", "adapter_loaded_via_peft": True},
        forward_loader is not None
        and forward_loader["status"] == "completed"
        and forward_loader["forbidden_fragments_present"] == []
        and q3_adapter.get("inference_mode") is True,
        "The project loader constructs or loads an active PEFT model and does not merge or disable the adapter branch.",
    )
    return {"rows": rows, "failures": failures}


def _audit_training_artifact_bindings(
    *,
    root_path: Path,
    q3_adapter: Mapping[str, Any],
    frozen_runtime_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    failures = []
    expected = {
        "bias": "none",
        "inference_mode": True,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "peft_type": "LORA",
        "peft_version": "0.19.1",
        "r": 8,
        "target_modules": ["q_proj", "v_proj"],
    }
    observed = {
        **{key: q3_adapter.get(key) for key in expected if key != "target_modules"},
        "target_modules": sorted(q3_adapter.get("target_modules", [])),
    }
    for key, value in expected.items():
        if observed.get(key) != value:
            failures.append(f"Q3 PEFT adapter binding mismatch: {key}")
    path = root_path / Q3_ADAPTER_CONFIG
    if not path.is_file():
        failures.append("Q3 PEFT adapter config is missing")
        digest = None
    else:
        digest = sha256_file(path)
    binding_failures = list(failures)
    implementation_failures = []
    implementation = frozen_runtime_metadata.get("implementation_identity")
    expected_implementation_digest = (
        "ccae915c402151faee71204803062bd346ef28737ac2c1ee796ec987026b4e93"
    )
    expected_project_sources = {
        "motivation_v12_contracts.py": Path(
            "src/myrec/baselines/motivation_v12_contracts.py"
        ),
        "motivation_v12_ranker.py": Path(
            "src/myrec/baselines/motivation_v12_ranker.py"
        ),
    }
    if not isinstance(implementation, Mapping):
        implementation = {}
        implementation_failures.append("frozen training implementation identity missing")
    if implementation.get("digest") != expected_implementation_digest:
        implementation_failures.append("frozen training implementation digest mismatch")
    metadata_files = implementation.get("files")
    metadata_sha_by_name = {}
    if isinstance(metadata_files, list):
        for row in metadata_files:
            if isinstance(row, Mapping) and isinstance(row.get("path"), str):
                metadata_sha_by_name[Path(row["path"]).name] = row.get("sha256")
    implementation_sources = []
    for name, relative in expected_project_sources.items():
        source_path = root_path / relative
        current_sha = sha256_file(source_path) if source_path.is_file() else None
        frozen_sha = metadata_sha_by_name.get(name)
        if current_sha is None:
            implementation_failures.append(f"frozen training source missing: {relative}")
        elif current_sha != frozen_sha:
            implementation_failures.append(
                f"current project source differs from frozen training identity: {name}"
            )
        implementation_sources.append(
            {
                "path": relative.as_posix(),
                "current_sha256": current_sha,
                "frozen_sha256": frozen_sha,
            }
        )
    if set(metadata_sha_by_name) != set(expected_project_sources):
        implementation_failures.append(
            "frozen training implementation source set mismatch"
        )
    failures.extend(implementation_failures)
    dtype_failures = []
    dtype_roots = {
        "q0_trainable_checkpoint": root_path / Q0_SAVED_CONFIG.parent,
        "q1_trainable_checkpoint": root_path / Q1_SAVED_CONFIG.parent,
        "q2_trainable_checkpoint": root_path / Q2_SAVED_CONFIG.parent,
        "q3_fp32_adapter_checkpoint": root_path / Q3_ADAPTER_CONFIG.parent,
        "q3_bf16_frozen_base": root_path / BASE_CONFIG.parent,
    }
    dtype_observed = {}
    for label, directory in dtype_roots.items():
        try:
            dtype_observed[label] = _safetensors_dtype_summary(directory)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            dtype_observed[label] = {}
            dtype_failures.append(f"tensor dtype identity unavailable: {label} ({exc})")
    dtype_expected = {
        "q0_trainable_checkpoint": {"F32": 595_776_512},
        "q1_trainable_checkpoint": {"F32": 596_049_920},
        "q2_trainable_checkpoint": {"F32": 596_049_920},
        "q3_fp32_adapter_checkpoint": {"F32": 1_146_880},
        "q3_bf16_frozen_base": {"BF16": 751_632_384},
    }
    if dtype_observed != dtype_expected:
        dtype_failures.append("frozen checkpoint tensor dtype/count contract mismatch")
    failures.extend(dtype_failures)
    return {
        "bindings": [
            {
                "binding_id": "q3_peft_lora_adapter_config",
                "binding_kind": "frozen_checkpoint_adapter_config",
                "path": Q3_ADAPTER_CONFIG.as_posix(),
                "sha256": digest,
                "observed": observed,
                "expected": expected,
                "dropout_executes_before_a_down_projection": True,
                "dropout_identity_at_evaluation": True,
                "status": "completed" if not binding_failures else "failed",
                "failures": binding_failures,
                "scientific_support_inferred": False,
                "operator_attribution_inferred": False,
            },
            {
                "binding_id": "frozen_training_project_source_identity",
                "binding_kind": "training_metadata_implementation_identity",
                "path": Q3_TRAINING_METADATA.as_posix(),
                "sha256": sha256_file(root_path / Q3_TRAINING_METADATA),
                "implementation_digest": implementation.get("digest"),
                "expected_implementation_digest": expected_implementation_digest,
                "source_identities": implementation_sources,
                "current_project_source_matches_frozen_training_identity": not implementation_failures,
                "status": "completed" if not implementation_failures else "failed",
                "failures": implementation_failures,
                "scientific_support_inferred": False,
                "operator_attribution_inferred": False,
            },
            {
                "binding_id": "frozen_checkpoint_tensor_dtype_identity",
                "binding_kind": "safetensors_header_dtype_and_shape_census",
                "path": "artifacts/motivation_v1_2/checkpoints and models/huggingface/Qwen3-0.6B",
                "sha256": hashlib.sha256(
                    json.dumps(
                        dtype_observed, sort_keys=True, separators=(",", ":")
                    ).encode("utf-8")
                ).hexdigest(),
                "observed": dtype_observed,
                "expected": dtype_expected,
                "q0_q2_trainable_master_parameters_are_fp32": True,
                "q3_lora_trainable_parameters_are_fp32": True,
                "q3_frozen_base_parameters_are_bfloat16": True,
                "status": "completed" if not dtype_failures else "failed",
                "failures": dtype_failures,
                "scientific_support_inferred": False,
                "operator_attribution_inferred": False,
            },
        ],
        "failures": failures,
    }


def _safetensors_dtype_summary(directory: Path) -> dict[str, int]:
    files = sorted(directory.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors files: {directory}")
    totals: dict[str, int] = {}
    for path in files:
        with path.open("rb") as handle:
            header_size_raw = handle.read(8)
            if len(header_size_raw) != 8:
                raise ValueError(f"invalid safetensors header size: {path}")
            header_size = int.from_bytes(header_size_raw, "little", signed=False)
            if header_size <= 0 or header_size > 100_000_000:
                raise ValueError(f"invalid safetensors header length: {path}")
            header = json.loads(handle.read(header_size).decode("utf-8"))
        if not isinstance(header, Mapping):
            raise ValueError(f"invalid safetensors header mapping: {path}")
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(tensor, Mapping):
                raise ValueError(f"invalid safetensors tensor header: {path}/{name}")
            dtype = tensor.get("dtype")
            shape = tensor.get("shape")
            if not isinstance(dtype, str) or not isinstance(shape, list):
                raise ValueError(f"invalid tensor dtype/shape: {path}/{name}")
            elements = 1
            for dimension in shape:
                if type(dimension) is not int or dimension < 0:
                    raise ValueError(f"invalid tensor dimension: {path}/{name}")
                elements *= dimension
            totals[dtype] = totals.get(dtype, 0) + elements
    return dict(sorted(totals.items()))


def _audit_frozen_forward_sources(
    frozen_runtime_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind forward semantics to the checkpoint's actual package environment."""

    failures: list[str] = []
    bindings = []
    package_versions = frozen_runtime_metadata.get("package_versions")
    if not isinstance(package_versions, Mapping):
        package_versions = {}
        failures.append("frozen runtime package versions are missing")
    transformers_version = package_versions.get("transformers")
    peft_version = package_versions.get("peft")
    if transformers_version != "5.12.1":
        failures.append("frozen runtime Transformers version is not 5.12.1")
    if peft_version != "0.19.1":
        failures.append("frozen runtime PEFT version is not 0.19.1")
    python_executable = frozen_runtime_metadata.get("python_executable")
    if not isinstance(python_executable, str) or not python_executable:
        failures.append("frozen runtime Python executable is missing")
        python_executable = None

    source_cache: dict[Path, str] = {}
    try:
        from myrec.baselines import motivation_v12_ranker as ranker
    except ImportError as exc:
        ranker = None
        failures.append(f"project forward source unavailable: {exc}")
    project_objects = (
        {"_load_model_and_tokenizer": ranker._load_model_and_tokenizer}
        if ranker is not None
        else {}
    )
    for source_id, contract in FORWARD_SOURCE_SENTINELS.items():
        source_failures = []
        object_path = str(contract["object"])
        relative_source = FROZEN_FORWARD_SOURCE_FILES.get(object_path)
        if relative_source is None:
            obj = project_objects.get(object_path)
            if obj is None:
                source = ""
                source_path = None
                source_failures.append(
                    f"project source object unavailable: {object_path}"
                )
            else:
                try:
                    source = inspect.getsource(obj)
                    source_file_value = inspect.getsourcefile(obj)
                except (OSError, TypeError) as exc:
                    source = ""
                    source_file_value = None
                    source_failures.append(f"source unreadable: {exc}")
                source_path = Path(source_file_value) if source_file_value else None
            if source_path is None or not source_path.is_file():
                source_file_sha256 = None
                package_relative_path = None
                source_failures.append("project source file identity unavailable")
            else:
                source_file_sha256 = sha256_file(source_path)
                package_relative_path = _package_relative_source_path(source_path)
            runtime_package = "project"
            runtime_version = None
        else:
            source_path = _find_frozen_site_package_source(
                python_executable, relative_source
            )
            if source_path is None:
                source = ""
                source_file_sha256 = None
                package_relative_path = relative_source
                source_failures.append(
                    f"frozen package source unavailable: {relative_source}"
                )
            else:
                if source_path not in source_cache:
                    try:
                        source_cache[source_path] = source_path.read_text(encoding="utf-8")
                    except (OSError, UnicodeError) as exc:
                        source_cache[source_path] = ""
                        source_failures.append(f"source file unreadable: {exc}")
                file_source = source_cache[source_path]
                try:
                    source = _extract_python_object_source(file_source, object_path)
                except (SyntaxError, ValueError) as exc:
                    source = ""
                    source_failures.append(f"object source unreadable: {exc}")
                source_file_sha256 = sha256_file(source_path)
                package_relative_path = _package_relative_source_path(source_path)
            runtime_package = (
                "peft" if relative_source.startswith("peft/") else "transformers"
            )
            runtime_version = package_versions.get(runtime_package)
        missing_fragments = [
            fragment for fragment in contract["fragments"] if fragment not in source
        ]
        if missing_fragments:
            source_failures.append(
                "required execution fragments missing: " + ",".join(missing_fragments)
            )
        forbidden_fragments = [
            fragment
            for fragment in contract.get("forbidden_fragments", ())
            if fragment in source
        ]
        if forbidden_fragments:
            source_failures.append(
                "forbidden execution fragments present: "
                + ",".join(forbidden_fragments)
            )
        if source_failures:
            failures.extend(
                f"source {source_id}: {message}" for message in source_failures
            )
        bindings.append(
            {
                "source_id": source_id,
                "object": object_path,
                "runtime_package": runtime_package,
                "runtime_package_version": runtime_version,
                "package_relative_path": package_relative_path,
                "source_file_sha256": source_file_sha256,
                "object_source_sha256": hashlib.sha256(
                    source.encode("utf-8")
                ).hexdigest()
                if source
                else None,
                "required_fragment_count": len(contract["fragments"]),
                "missing_fragments": missing_fragments,
                "forbidden_fragments_present": forbidden_fragments,
                "status": "completed" if not source_failures else "failed",
                "failures": source_failures,
                "scientific_support_inferred": False,
                "operator_attribution_inferred": False,
            }
        )
    return {
        "transformers_version": transformers_version,
        "peft_version": peft_version,
        "python_executable": python_executable,
        "source_environment_is_frozen_checkpoint_environment": (
            not failures and python_executable == frozen_runtime_metadata.get("python_executable")
        ),
        "bindings": bindings,
        "failures": failures,
    }


def _find_frozen_site_package_source(
    python_executable: str | None, relative_source: str
) -> Path | None:
    if python_executable is None:
        return None
    executable = Path(python_executable)
    if not executable.is_file():
        return None
    environment_root = executable.resolve().parent.parent
    candidates = sorted({
        path.resolve()
        for path in (environment_root / "lib").glob(
            f"python*/site-packages/{relative_source}"
        )
        if path.is_file()
    })
    return candidates[0] if len(candidates) == 1 else None


def _extract_python_object_source(file_source: str, object_path: str) -> str:
    tree = ast.parse(file_source)
    parts = object_path.split(".")
    nodes: list[ast.AST] = list(tree.body)
    selected: ast.AST | None = None
    for part in parts:
        selected = next(
            (
                node
                for node in nodes
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == part
            ),
            None,
        )
        if selected is None:
            raise ValueError(f"object not found: {object_path}")
        nodes = list(getattr(selected, "body", []))
    source = ast.get_source_segment(file_source, selected) if selected else None
    if not source:
        raise ValueError(f"object source segment unavailable: {object_path}")
    return source


def _audit_inactive_architecture_paths(
    *,
    base_configs: Mapping[str, Mapping[str, Any]],
    runtime_identity_smokes: list[dict[str, Any]],
    source_bindings: list[dict[str, Any]],
) -> dict[str, Any]:
    source_status = {
        str(row["source_id"]): row["status"] for row in source_bindings
    }
    config_checks = (
        (
            "attention_dropout_stochasticity",
            "attention_dropout",
            0.0,
            "Self-attention dropout is numerically disabled in train and eval paths.",
        ),
        (
            "sliding_window_attention",
            "sliding_window",
            None,
            "All 28 decoder layers use full causal attention; no sliding window is active.",
        ),
        (
            "sliding_window_dispatch",
            "use_sliding_window",
            False,
            "The alternate sliding-attention mask/dispatch path is disabled.",
        ),
        (
            "attention_projection_bias",
            "attention_bias",
            False,
            "Q/K/V/O projections instantiate without additive bias.",
        ),
        (
            "untied_language_model_head",
            "tie_word_embeddings",
            True,
            "The independent untied lm-head parameter path is inactive.",
        ),
    )
    rows = []
    failures: list[str] = []
    for path_id, config_key, expected, explanation in config_checks:
        observed = {
            label: config.get(config_key) for label, config in base_configs.items()
        }
        verified = all(value == expected for value in observed.values())
        path_failures = [] if verified else [f"{config_key} differs from {expected!r}"]
        if path_failures:
            failures.extend(f"inactive path {path_id}: {item}" for item in path_failures)
        rows.append(
            {
                "path_id": path_id,
                "evidence_kind": "frozen_base_config",
                "observed": observed,
                "expected": expected,
                "inactive_verified": verified,
                "explanation": explanation,
                "failures": path_failures,
                "scientific_support_inferred": False,
                "operator_attribution_inferred": False,
            }
        )

    rope_observed = {
        label: {
            "rope_theta": config.get("rope_theta"),
            "rope_scaling": config.get("rope_scaling"),
            "rope_parameters": config.get("rope_parameters"),
        }
        for label, config in base_configs.items()
    }
    rope_expected = {
        "rope_theta": 1_000_000,
        "rope_scaling": None,
        "rope_parameters": None,
    }
    rope_verified = all(value == rope_expected for value in rope_observed.values())
    rope_failures = (
        []
        if rope_verified
        else ["non-default or dynamic RoPE parameters are active"]
    )
    if rope_failures:
        failures.extend(
            f"inactive path non_default_or_dynamic_rope_scaling: {item}"
            for item in rope_failures
        )
    rows.append(
        {
            "path_id": "non_default_or_dynamic_rope_scaling",
            "evidence_kind": "frozen_base_config_and_installed_source",
            "observed": rope_observed,
            "expected": rope_expected,
            "inactive_verified": rope_verified,
            "explanation": "The installed RotaryEmbedding supports dynamic updates, but both frozen bases use default fixed-theta RoPE without a scaling branch.",
            "failures": rope_failures,
            "scientific_support_inferred": False,
            "operator_attribution_inferred": False,
        }
    )

    source_only_checks = (
        (
            "mlp_projection_bias",
            "qwen3_mlp",
            "Qwen3MLP hard-codes gate/up/down Linear bias=False.",
        ),
        (
            "native_attention_weight_materialization",
            "transformers_sdpa",
            "The frozen SDPA score path returns no native attention-weight tensor; project-owned observations remain separate diagnostics.",
        ),
    )
    for path_id, source_id, explanation in source_only_checks:
        verified = source_status.get(source_id) == "completed"
        path_failures = [] if verified else [f"source binding {source_id} failed"]
        if path_failures:
            failures.extend(f"inactive path {path_id}: {item}" for item in path_failures)
        rows.append(
            {
                "path_id": path_id,
                "evidence_kind": "installed_transformers_source",
                "observed": {"source_binding": source_id},
                "expected": "source sentinel verified",
                "inactive_verified": verified,
                "explanation": explanation,
                "failures": path_failures,
                "scientific_support_inferred": False,
                "operator_attribution_inferred": False,
            }
        )

    backend_observed = {
        str(row["method_id"]): row.get("attention_backend")
        for row in runtime_identity_smokes
    }
    backend_verified = bool(backend_observed) and all(
        value == "sdpa" for value in backend_observed.values()
    )
    backend_failures = [] if backend_verified else ["not all frozen models used SDPA"]
    if backend_failures:
        failures.extend(
            f"inactive path alternative_attention_backend: {item}"
            for item in backend_failures
        )
    rows.append(
        {
            "path_id": "alternative_attention_backend",
            "evidence_kind": "four_model_runtime_identity",
            "observed": backend_observed,
            "expected": "sdpa",
            "inactive_verified": backend_verified,
            "explanation": "Eager/flash alternatives are not the frozen native scoring backend; eager is used only in registered diagnostic cross-checks.",
            "failures": backend_failures,
            "scientific_support_inferred": False,
            "operator_attribution_inferred": False,
        }
    )
    return {"rows": rows, "failures": failures}


def _package_relative_source_path(path: Path) -> str:
    parts = path.resolve().parts
    for package in ("myrec", "peft", "torch", "transformers"):
        if package in parts:
            index = max(i for i, value in enumerate(parts) if value == package)
            return Path(*parts[index:]).as_posix()
    return path.name


def _failed_payload(
    root: Path, paths: Mapping[str, Path], failures: list[str]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "analysis_type": "frozen_qwen_model_architecture_audit",
        "status": "failed",
        "failures": failures,
        "files": {
            key: {
                "path": relative.as_posix(),
                "sha256": sha256_file(root / relative)
                if (root / relative).is_file()
                else None,
            }
            for key, relative in paths.items()
        },
        "scientific_effect_values_read": False,
        "scientific_support_inferred": False,
        "qrels_read_by_audit": False,
        "source_test_opened": False,
    }


def _audit_topology(
    config: Mapping[str, Any],
    label: str,
    failures: list[str],
    *,
    saved: bool,
) -> None:
    for key, expected in EXPECTED_BASE_TOPOLOGY.items():
        if saved and key == "rope_theta":
            continue
        if config.get(key) != expected:
            failures.append(f"{label} topology mismatch: {key}")
    rope_theta = config.get("rope_theta")
    if saved and rope_theta is None and isinstance(
        config.get("rope_parameters"), Mapping
    ):
        rope_theta = config["rope_parameters"].get("rope_theta")
    if rope_theta != EXPECTED_BASE_TOPOLOGY["rope_theta"]:
        failures.append(f"{label} topology mismatch: rope_theta")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return _mapping(value, path.as_posix())


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _mapping(value, path.as_posix())


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is not a mapping")
    return dict(value)
