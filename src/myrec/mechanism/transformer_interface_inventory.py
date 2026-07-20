"""Outcome-independent inventory of exact Transformer implementation interfaces.

The comprehensive report uses eighteen scientific component classes, but those
classes intentionally aggregate several concrete Qwen interfaces.  This module
keeps a second, implementation-level census so that Q/K normalization, RoPE,
attention edges, every SwiGLU stage, residual boundaries, the native readout,
and training-only paths cannot disappear behind an aggregate component label.

The inventory only records preregistered evidence availability.  It never reads
effects and it does not infer scientific support from completion.
"""

from __future__ import annotations

from typing import Any, Mapping

from myrec.mechanism.deep_dive_closeout_audit import EXPECTED_DELIVERABLES
from myrec.mechanism.deep_dive_evidence_topology import (
    DELIVERABLE_MODEL_COVERAGE,
    MODEL_IDS,
)
from myrec.mechanism.deep_dive_report_contract import (
    COMPONENT_ALLOWED_DELIVERABLES,
    COMPONENT_IDS,
)
from myrec.mechanism.supplemental_evidence_registry import EXPECTED_SUPPLEMENT_IDS
from myrec.mechanism.transformer_instrumentation import BLOCK_NODE_IDS, FINAL_NODE_IDS


EVIDENCE_IDS = frozenset(EXPECTED_DELIVERABLES) | frozenset(
    EXPECTED_SUPPLEMENT_IDS
)
CAUSAL_ROLES = frozenset(
    {
        "causal_sufficiency",
        "causal_edge_intervention",
        "causal_context_intervention",
        "causal_position_intervention",
        "causal_readout_intervention",
        "causal_necessity",
        "causal_design_gate",
    }
)
SYSTEM_LAYERS = frozenset({"input", "representation", "routing", "readout", "training"})
ROLE_CLAIM_CEILINGS = {
    "measurement_confound_audit": "M",
    "descriptive": "D",
    "interventional_localization": "D",
    "training_dynamics": "D",
    "causal_context_intervention": "S",
    "causal_edge_intervention": "S",
    "causal_position_intervention": "S",
    "causal_readout_intervention": "S",
    "causal_sufficiency": "S",
    "causal_necessity": "N",
    "causal_design_gate": "G",
}
CLAIM_CEILING_ORDER = {"none": -1, "M": 0, "D": 1, "S": 2, "N": 3, "G": 4}


def _evidence(*rows: tuple[str, str]) -> tuple[dict[str, str], ...]:
    return tuple({"evidence_id": evidence_id, "role": role} for evidence_id, role in rows)


TRANSFORMER_INTERFACE_INVENTORY = (
    {
        "interface_id": "serialization_tokenization",
        "system_layer": "input",
        "implementation_surface": "field whitelist, prompt serialization, truncation, and token spans",
        "component_ids": ("serialization_tokenization",),
        "evidence": _evidence(
            ("d5_context", "causal_context_intervention"),
            ("d3_full_null_position_shift_audit", "measurement_confound_audit"),
        ),
        "claim_boundary": "Input controls can expose serialization or position confounds; they do not identify an internal operator bottleneck.",
    },
    {
        "interface_id": "token_embedding_lookup",
        "system_layer": "input",
        "implementation_surface": "embed_tokens lookup before the first decoder block",
        "component_ids": ("token_embedding",),
        "evidence": _evidence(
            ("d1_representation", "descriptive"),
            ("d0_embedding_readout_geometry", "descriptive"),
        ),
        "claim_boundary": "Shared-weight readout evidence cannot be borrowed as input-embedding causality; geometry alone does not establish an embedding operator bottleneck.",
    },
    {
        "interface_id": "tied_lm_head_rows",
        "system_layer": "readout",
        "implementation_surface": "tied lm_head token rows used by each model-specific native score",
        "component_ids": ("native_readout",),
        "evidence": _evidence(
            ("d0_embedding_readout_geometry", "descriptive"),
            ("d6_q0_q1_readouts", "causal_readout_intervention"),
            ("d6_q2_native_readout", "causal_readout_intervention"),
            ("d6_q3_native_readout", "causal_readout_intervention"),
        ),
        "claim_boundary": "Tied-row readout mediation is not evidence that the same shared parameter rows are causal at the input embedding lookup.",
    },
    {
        "interface_id": "autoregressive_causal_attention_mask",
        "system_layer": "routing",
        "implementation_surface": "frozen autoregressive visibility topology applied before attention softmax",
        "component_ids": ("attention_query_key_routing", "history_routing"),
        "evidence": _evidence(
            ("d3_attention_pattern_synthesis", "descriptive"),
        ),
        "claim_boundary": "The diagnostic history-edge mask is not an intervention on the frozen autoregressive topology; observed route use cannot establish that the base mask is optimal or causal for transfer failure.",
    },
    {
        "interface_id": "kv_cache_phase_boundary",
        "system_layer": "routing",
        "implementation_surface": "Q1 prefix cache, continuation calls, and answer-token phases",
        "component_ids": ("history_routing", "layerwise_representation"),
        "implementation_model_scope": (MODEL_IDS[1],),
        "evidence": _evidence(("d6_q1_trajectory", "descriptive")),
        "claim_boundary": "KV-phase coverage prevents single-forward assumptions; it does not make Q1 directly comparable to pointwise Q2/Q3 readout semantics.",
    },
    {
        "interface_id": "block_input_residual",
        "system_layer": "representation",
        "implementation_surface": "incoming residual stream before input RMSNorm",
        "component_ids": ("layerwise_representation", "history_routing"),
        "evidence": _evidence(
            ("d2_selected_branches", "causal_sufficiency"),
            ("component_state_reverse_necessity_v2", "causal_necessity"),
            ("component_functional_design_gate_synthesis", "causal_design_gate"),
        ),
        "claim_boundary": "Incoming-state mediation localizes an upstream carrier; it cannot be attributed to computation inside the current block.",
    },
    {
        "interface_id": "input_rmsnorm_output",
        "system_layer": "representation",
        "implementation_surface": "input_layernorm output before self-attention",
        "component_ids": ("normalization",),
        "evidence": _evidence(
            ("d2_selected_branches", "causal_sufficiency"),
            ("d2_rmsnorm_flow", "descriptive"),
        ),
        "claim_boundary": "Pre/post-state differences do not by themselves establish RMSNorm operator necessity.",
    },
    {
        "interface_id": "input_rmsnorm_variance_rescale_and_gain",
        "system_layer": "representation",
        "implementation_surface": "input RMSNorm variance rescaling and learned gain applied to the incoming residual",
        "component_ids": ("normalization",),
        "evidence": _evidence(("d2_rmsnorm_flow", "descriptive")),
        "claim_boundary": "Observed input/output geometry does not isolate the variance denominator or learned gain as the cause of a downstream state change.",
    },
    {
        "interface_id": "q_pre_norm",
        "system_layer": "routing",
        "implementation_surface": "q_proj output before QK normalization",
        "component_ids": ("attention_query_key_routing",),
        "evidence": _evidence(
            ("d5_rope", "causal_position_intervention"),
            ("d3_qk_stage_geometry_v3", "descriptive"),
        ),
        "claim_boundary": "Q-stage geometry brackets where routing changes; only registered interventions can support a causal routing claim.",
    },
    {
        "interface_id": "q3_q_lora_scaled_adapter_injection",
        "system_layer": "routing",
        "implementation_surface": "Q3 unmerged q_proj PEFT branch base_q(x) + (alpha/r) B_q(A_q(identity(x))) before q_norm at evaluation",
        "component_ids": ("lora_parameterization", "attention_query_key_routing"),
        "implementation_model_scope": (MODEL_IDS[3],),
        "evidence": _evidence(
            ("d7_q3_lora_path", "training_dynamics"),
            ("d7_optimizer_replay", "training_dynamics"),
            ("d7_q3_lora_head_geometry", "descriptive"),
        ),
        "claim_boundary": "The active q-adapter branch is implementation- and geometry-backed, but no branch-specific bypass or matched control attributes transfer failure or ranking utility to its A/B/scale/add composition.",
    },
    {
        "interface_id": "k_pre_norm",
        "system_layer": "routing",
        "implementation_surface": "k_proj output before QK normalization",
        "component_ids": ("attention_query_key_routing",),
        "evidence": _evidence(
            ("d5_rope", "causal_position_intervention"),
            ("d3_qk_stage_geometry_v3", "descriptive"),
        ),
        "claim_boundary": "K-stage geometry brackets where routing changes; it does not identify user-specific transport without registered controls.",
    },
    {
        "interface_id": "q_post_norm_pre_rope",
        "system_layer": "routing",
        "implementation_surface": "Q normalization output before rotary phase application",
        "component_ids": ("attention_query_key_routing", "normalization"),
        "evidence": _evidence(
            ("d5_rope", "causal_position_intervention"),
            ("d3_qk_stage_geometry_v3", "descriptive"),
        ),
        "claim_boundary": "Normalization and position effects remain distinct unless a registered contrast separates them.",
    },
    {
        "interface_id": "k_post_norm_pre_rope",
        "system_layer": "routing",
        "implementation_surface": "K normalization output before rotary phase application",
        "component_ids": ("attention_query_key_routing", "normalization"),
        "evidence": _evidence(
            ("d5_rope", "causal_position_intervention"),
            ("d3_qk_stage_geometry_v3", "descriptive"),
        ),
        "claim_boundary": "Normalization and position effects remain distinct unless a registered contrast separates them.",
    },
    {
        "interface_id": "q_head_rmsnorm_variance_rescale_and_gain",
        "system_layer": "routing",
        "implementation_surface": "head-dimensional q_norm RMS variance rescaling and learned gain between q_proj and RoPE",
        "component_ids": ("attention_query_key_routing", "normalization"),
        "evidence": _evidence(("d3_qk_stage_geometry_v3", "descriptive")),
        "claim_boundary": "Pre/post q_norm geometry brackets a routing change but does not isolate query-head normalization as its operator cause.",
    },
    {
        "interface_id": "k_head_rmsnorm_variance_rescale_and_gain",
        "system_layer": "routing",
        "implementation_surface": "head-dimensional k_norm RMS variance rescaling and learned gain between k_proj and RoPE",
        "component_ids": ("attention_query_key_routing", "normalization"),
        "evidence": _evidence(("d3_qk_stage_geometry_v3", "descriptive")),
        "claim_boundary": "Pre/post k_norm geometry brackets a routing change but does not isolate key-head normalization as its operator cause.",
    },
    {
        "interface_id": "q_post_rope",
        "system_layer": "routing",
        "implementation_surface": "rotated Q captured by the project-owned attention wrapper",
        "component_ids": ("positional_encoding_rope", "attention_query_key_routing"),
        "evidence": _evidence(
            ("d5_rope", "causal_position_intervention"),
            ("d5_rope_position_geometry", "descriptive"),
            ("d3_qk_stage_geometry_v3", "descriptive"),
        ),
        "claim_boundary": "RoPE support is scoped to registered phase interventions and cannot be inferred from norm preservation alone.",
    },
    {
        "interface_id": "k_post_rope",
        "system_layer": "routing",
        "implementation_surface": "rotated K captured by the project-owned attention wrapper",
        "component_ids": ("positional_encoding_rope", "attention_query_key_routing"),
        "evidence": _evidence(
            ("d5_rope", "causal_position_intervention"),
            ("d5_rope_position_geometry", "descriptive"),
            ("d3_qk_stage_geometry_v3", "descriptive"),
        ),
        "claim_boundary": "RoPE support is scoped to registered phase interventions and cannot be inferred from norm preservation alone.",
    },
    {
        "interface_id": "v_projection",
        "system_layer": "routing",
        "implementation_surface": "v_proj content before grouped-query attention transport",
        "component_ids": ("attention_value_transport",),
        "evidence": _evidence(
            ("d3_attention_edges", "causal_edge_intervention"),
            ("d3_attention_heads", "descriptive"),
        ),
        "claim_boundary": "Value-edge removal tests transported content at registered rows; it does not prove that all value-path information is preference-specific.",
    },
    {
        "interface_id": "q3_v_lora_scaled_adapter_injection",
        "system_layer": "routing",
        "implementation_surface": "Q3 unmerged v_proj PEFT branch base_v(x) + (alpha/r) B_v(A_v(identity(x))) before GQA value transport at evaluation",
        "component_ids": ("lora_parameterization", "attention_value_transport"),
        "implementation_model_scope": (MODEL_IDS[3],),
        "evidence": _evidence(
            ("d7_q3_lora_path", "training_dynamics"),
            ("d7_optimizer_replay", "training_dynamics"),
            ("d7_q3_lora_head_geometry", "descriptive"),
        ),
        "claim_boundary": "The active v-adapter branch is implementation- and geometry-backed, but no branch-specific bypass or matched control attributes transfer failure or ranking utility to its A/B/scale/add composition.",
    },
    {
        "interface_id": "attention_scaled_qk_logits",
        "system_layer": "routing",
        "implementation_surface": "scaled QK dot-product logits before additive causal masking and softmax",
        "component_ids": ("attention_query_key_routing", "history_routing"),
        "evidence": _evidence(
            ("d3_qk_stage_geometry_v3", "descriptive"),
        ),
        "claim_boundary": "Q/K geometry brackets logit formation, but an edge-mask intervention does not isolate the dot product or its scaling as the operator cause.",
    },
    {
        "interface_id": "attention_softmax_edge_weights",
        "system_layer": "routing",
        "implementation_surface": "post-mask softmax probabilities over query/history/candidate span edges",
        "component_ids": ("attention_query_key_routing", "history_routing"),
        "evidence": _evidence(
            ("d3_attention_edges", "causal_edge_intervention"),
            ("d3_attention_pattern_synthesis", "descriptive"),
            ("d3_attention_heads", "descriptive"),
        ),
        "claim_boundary": "Attention mass is descriptive; registered edge interventions can identify route dependence but do not isolate the softmax nonlinearity itself.",
    },
    {
        "interface_id": "attention_head_output_pre_o",
        "system_layer": "routing",
        "implementation_surface": "16 query-head outputs before o_proj regrouping",
        "component_ids": ("attention_value_transport", "attention_output"),
        "evidence": _evidence(
            ("d3_attention_heads", "descriptive"),
            ("d3_attention_groups", "interventional_localization"),
            ("d3_attention_edges", "causal_edge_intervention"),
        ),
        "claim_boundary": "Head/group localization is exploratory and cannot promote a selected head index into an architecture target.",
    },
    {
        "interface_id": "gqa_query_to_kv_grouping",
        "system_layer": "routing",
        "implementation_surface": "16 query heads grouped over 8 shared K/V heads before o_proj",
        "component_ids": (
            "attention_query_key_routing",
            "attention_value_transport",
            "lora_parameterization",
        ),
        "evidence": _evidence(
            ("d3_attention_heads", "descriptive"),
            ("d3_attention_groups", "interventional_localization"),
            ("d7_q3_lora_head_geometry", "descriptive"),
        ),
        "claim_boundary": "GQA group localization is exploratory; shared K/V topology does not identify a causal group or justify changing head counts without a new preregistered intervention.",
    },
    {
        "interface_id": "attention_o_projection",
        "system_layer": "routing",
        "implementation_surface": "o_proj output before residual addition",
        "component_ids": ("attention_output",),
        "evidence": _evidence(
            ("d2_selected_branches", "causal_sufficiency"),
            ("component_state_reverse_necessity_v2", "causal_necessity"),
            ("component_functional_design_gate_synthesis", "causal_design_gate"),
        ),
        "claim_boundary": "Only the joint S/N/specificity/control gate can qualify this functional state; it does not establish a unique causal head.",
    },
    {
        "interface_id": "post_attention_residual",
        "system_layer": "representation",
        "implementation_surface": "block input plus attention increment",
        "component_ids": ("residual_composition", "layerwise_representation"),
        "evidence": _evidence(("d2_selected_branches", "causal_sufficiency")),
        "claim_boundary": "A composed state boundary does not separately identify residual addition, attention, or nonlinear interaction as the operator cause.",
    },
    {
        "interface_id": "attention_residual_addition",
        "system_layer": "representation",
        "implementation_surface": "elementwise composition of the incoming residual with the attention o_proj increment",
        "component_ids": ("residual_composition",),
        "evidence": _evidence(
            ("d2_rmsnorm_flow", "descriptive"),
            ("d4_mlp_groups", "descriptive"),
        ),
        "claim_boundary": "Exact recomposition proves the frozen addition algebra, while state patches and branch geometry do not isolate the addition rule as the transfer-failure operator.",
    },
    {
        "interface_id": "post_attention_rmsnorm_output",
        "system_layer": "representation",
        "implementation_surface": "post_attention_layernorm output before MLP",
        "component_ids": ("normalization",),
        "evidence": _evidence(
            ("d2_selected_branches", "causal_sufficiency"),
            ("d2_rmsnorm_flow", "descriptive"),
        ),
        "claim_boundary": "A post-normalization carrier is not proof that RMSNorm selectively creates or erases transferable information.",
    },
    {
        "interface_id": "post_attention_rmsnorm_variance_rescale_and_gain",
        "system_layer": "representation",
        "implementation_surface": "post-attention RMSNorm variance rescaling and learned gain before the MLP",
        "component_ids": ("normalization",),
        "evidence": _evidence(("d2_rmsnorm_flow", "descriptive")),
        "claim_boundary": "A post-norm state boundary can localize a carrier but does not isolate the normalization denominator or gain from the MLP response it induces.",
    },
    {
        "interface_id": "mlp_gate_projection",
        "system_layer": "representation",
        "implementation_surface": "SwiGLU gate_proj output",
        "component_ids": ("mlp_feature_formation",),
        "evidence": _evidence(
            ("d4_mlp_groups", "descriptive"),
            ("d4_mlp_feature_formation_extension", "descriptive"),
        ),
        "claim_boundary": "Feature concentration or signed projection is descriptive and does not establish gate-neuron causality.",
    },
    {
        "interface_id": "mlp_up_projection",
        "system_layer": "representation",
        "implementation_surface": "SwiGLU up_proj output",
        "component_ids": ("mlp_feature_formation",),
        "evidence": _evidence(
            ("d4_mlp_groups", "descriptive"),
            ("d4_mlp_feature_formation_extension", "descriptive"),
        ),
        "claim_boundary": "Feature concentration or signed projection is descriptive and does not establish up-neuron causality.",
    },
    {
        "interface_id": "mlp_silu_gate",
        "system_layer": "representation",
        "implementation_surface": "SiLU(gate_proj) nonlinear activation",
        "component_ids": ("mlp_feature_formation",),
        "evidence": _evidence(("d4_mlp_feature_formation_extension", "descriptive")),
        "claim_boundary": "Activation formation is observed without outcome-selecting neurons; it cannot establish nonlinear operator necessity.",
    },
    {
        "interface_id": "mlp_swiglu_product",
        "system_layer": "representation",
        "implementation_surface": "SiLU(gate_proj) multiplied by up_proj",
        "component_ids": ("mlp_feature_formation",),
        "evidence": _evidence(
            ("d4_mlp_groups", "descriptive"),
            ("d4_mlp_feature_formation_extension", "descriptive"),
        ),
        "claim_boundary": "Grouped SwiGLU observations localize formation geometry; they do not qualify exact dimensions as a method target.",
    },
    {
        "interface_id": "mlp_down_projection",
        "system_layer": "representation",
        "implementation_surface": "down_proj output before block residual addition",
        "component_ids": ("mlp_output",),
        "evidence": _evidence(
            ("d2_selected_branches", "causal_sufficiency"),
            ("d4_mlp_groups", "descriptive"),
            ("component_state_reverse_necessity_v2", "causal_necessity"),
            ("component_functional_design_gate_synthesis", "causal_design_gate"),
        ),
        "claim_boundary": "Only the joint S/N/specificity/control gate can qualify the MLP output state; grouped dimensions remain descriptive.",
    },
    {
        "interface_id": "block_output_residual",
        "system_layer": "representation",
        "implementation_surface": "post-attention residual plus MLP increment",
        "component_ids": ("residual_composition", "layerwise_representation"),
        "evidence": _evidence(
            ("d2_postblock", "causal_sufficiency"),
            ("d2_selected_branches", "causal_sufficiency"),
            ("component_state_reverse_necessity_v2", "causal_necessity"),
        ),
        "claim_boundary": "The full block state is a sufficiency ceiling and does not isolate residual addition as a design component.",
    },
    {
        "interface_id": "mlp_residual_addition",
        "system_layer": "representation",
        "implementation_surface": "elementwise composition of the post-attention residual with the MLP down_proj increment",
        "component_ids": ("residual_composition",),
        "evidence": _evidence(
            ("d2_rmsnorm_flow", "descriptive"),
            ("d4_mlp_groups", "descriptive"),
        ),
        "claim_boundary": "The complete block state and branch angles do not isolate the second residual-addition rule from the MLP increment or incoming state.",
    },
    {
        "interface_id": "final_rmsnorm_input",
        "system_layer": "readout",
        "implementation_surface": "last block output before final model RMSNorm",
        "component_ids": ("normalization", "native_readout"),
        "evidence": _evidence(
            ("d6_q0_q1_readouts", "causal_readout_intervention"),
            ("d6_q2_native_readout", "causal_readout_intervention"),
            ("d6_q3_native_readout", "causal_readout_intervention"),
            ("d6_native_readout_diagnostics", "descriptive"),
            ("d2_rmsnorm_flow", "descriptive"),
        ),
        "claim_boundary": "Pre-final-norm state support must be separated from final-norm and score-path support before naming a readout bottleneck.",
    },
    {
        "interface_id": "final_rmsnorm_output",
        "system_layer": "readout",
        "implementation_surface": "final model RMSNorm output consumed by native scoring",
        "component_ids": ("normalization", "native_readout"),
        "evidence": _evidence(
            ("d6_q0_q1_readouts", "causal_readout_intervention"),
            ("d6_q2_native_readout", "causal_readout_intervention"),
            ("d6_q3_native_readout", "causal_readout_intervention"),
            ("d6_native_readout_diagnostics", "descriptive"),
            ("d2_rmsnorm_flow", "descriptive"),
        ),
        "claim_boundary": "Final-norm mediation is scoped to the frozen native score and is not evidence for an alternative learned readout.",
    },
    {
        "interface_id": "final_rmsnorm_variance_rescale_and_gain",
        "system_layer": "readout",
        "implementation_surface": "final RMSNorm variance rescaling and learned gain before tied-row native scoring",
        "component_ids": ("normalization", "native_readout"),
        "evidence": _evidence(
            ("d2_rmsnorm_flow", "descriptive"),
            ("d6_native_readout_diagnostics", "descriptive"),
        ),
        "claim_boundary": "Formula recomposition and pre/post geometry are mechanical or descriptive; they do not identify the final normalization operator as the cause of transfer failure.",
    },
    {
        "interface_id": "candidate_readout_positions",
        "system_layer": "readout",
        "implementation_surface": "all registered candidate/answer prediction positions",
        "component_ids": (
            "candidate_conditioned_interaction",
            "history_routing",
            "native_readout",
        ),
        "evidence": _evidence(
            ("d5_context", "causal_context_intervention"),
            ("d5_rope", "causal_position_intervention"),
            ("d6_q0_q1_readouts", "causal_readout_intervention"),
            ("d6_q2_native_readout", "causal_readout_intervention"),
            ("d6_q3_native_readout", "causal_readout_intervention"),
        ),
        "claim_boundary": "Candidate-relative positions must be complete for each model path; a single prompt token cannot stand in for a multi-position readout.",
    },
    {
        "interface_id": "q0_next_token_yes_no_logit_difference",
        "system_layer": "readout",
        "implementation_surface": "Q0 final-prompt-token raw yes-token minus no-token logit",
        "component_ids": (
            "native_readout",
            "score_calibration_nullspace",
            "candidate_conditioned_interaction",
        ),
        "implementation_model_scope": (MODEL_IDS[0],),
        "evidence": _evidence(
            ("d6_q0_q1_readouts", "causal_readout_intervention"),
        ),
        "claim_boundary": "Q0 readout mediation remains specialized-reranker scoped and does not establish positive ranking utility or validate a replacement head.",
    },
    {
        "interface_id": "q1_candidate_response_mean_log_likelihood",
        "system_layer": "readout",
        "implementation_surface": "Q1 mean token log-likelihood of each marked candidate response using a shared prompt KV cache",
        "component_ids": (
            "native_readout",
            "score_calibration_nullspace",
            "candidate_conditioned_interaction",
        ),
        "implementation_model_scope": (MODEL_IDS[1],),
        "evidence": _evidence(
            ("d6_q0_q1_readouts", "causal_readout_intervention"),
        ),
        "claim_boundary": "Q1 sequence-likelihood mediation is not interchangeable with a single-token logit difference and remains scoped to its frozen response serialization and cache phases.",
    },
    {
        "interface_id": "q2_next_token_yes_no_logit_difference",
        "system_layer": "readout",
        "implementation_surface": "Q2 final-prompt-token raw yes-token minus no-token logit",
        "component_ids": (
            "native_readout",
            "score_calibration_nullspace",
            "candidate_conditioned_interaction",
        ),
        "implementation_model_scope": (MODEL_IDS[2],),
        "evidence": _evidence(
            ("d6_q2_native_readout", "causal_readout_intervention"),
            ("d6_frozen_logit_lens", "descriptive"),
            ("d0_embedding_readout_geometry", "descriptive"),
            ("d6_native_readout_diagnostics", "descriptive"),
        ),
        "claim_boundary": "Q2 logit-difference mediation does not establish positive ranking utility or transfer to sequence-likelihood readouts.",
    },
    {
        "interface_id": "q3_two_path_mean_log_likelihood_difference",
        "system_layer": "readout",
        "implementation_surface": "Q3 mean Yes-path log-likelihood minus mean No-path log-likelihood across four native terms at three causal states",
        "component_ids": (
            "native_readout",
            "score_calibration_nullspace",
            "candidate_conditioned_interaction",
        ),
        "implementation_model_scope": (MODEL_IDS[3],),
        "evidence": _evidence(
            ("d6_q3_native_readout", "causal_readout_intervention"),
            ("d6_frozen_logit_lens", "descriptive"),
            ("d0_embedding_readout_geometry", "descriptive"),
            ("d6_native_readout_diagnostics", "descriptive"),
        ),
        "claim_boundary": "Q3 four-term path mediation does not establish positive ranking utility and cannot be reduced to its shared prompt state or a single-token score.",
    },
    {
        "interface_id": "q0_pointwise_bce_loss",
        "system_layer": "training",
        "implementation_surface": "Q0 specialized-reranker pointwise binary cross-entropy",
        "component_ids": ("loss_gradient",),
        "implementation_model_scope": (MODEL_IDS[0],),
        "evidence": _evidence(),
        "claim_boundary": "The Q0 loss exists in the frozen training path but has no registered loss-level mechanism artifact in this stage.",
    },
    {
        "interface_id": "q1_normalized_response_nll",
        "system_layer": "training",
        "implementation_surface": "Q1 normalized candidate-response sequence negative log-likelihood",
        "component_ids": ("loss_gradient",),
        "implementation_model_scope": (MODEL_IDS[1],),
        "evidence": _evidence(),
        "claim_boundary": "The Q1 sequence loss exists in the frozen training path but has no registered loss-level mechanism artifact in this stage.",
    },
    {
        "interface_id": "q2_pairwise_ranknet_loss",
        "system_layer": "training",
        "implementation_surface": "Q2 pairwise RankNet term with frozen weight 0.5",
        "component_ids": ("loss_gradient",),
        "implementation_model_scope": (MODEL_IDS[2],),
        "evidence": _evidence(
            ("d7_q2_objective", "training_dynamics"),
            ("d7_q2_objective_family_shares", "descriptive"),
        ),
        "claim_boundary": "RankNet gradient geometry diagnoses training pressure; it is not a randomized loss ablation or a utility result.",
    },
    {
        "interface_id": "q2_listwise_listnet_loss",
        "system_layer": "training",
        "implementation_surface": "Q2 tie-aware ListNet term with frozen weight 0.5",
        "component_ids": ("loss_gradient",),
        "implementation_model_scope": (MODEL_IDS[2],),
        "evidence": _evidence(
            ("d7_q2_objective", "training_dynamics"),
            ("d7_q2_objective_family_shares", "descriptive"),
        ),
        "claim_boundary": "ListNet gradient geometry diagnoses training pressure; it is not a randomized loss ablation or a utility result.",
    },
    {
        "interface_id": "q3_alignment_nll_loss",
        "system_layer": "training",
        "implementation_surface": "Q3 output-only Yes/No recommendation-alignment negative log-likelihood",
        "component_ids": ("loss_gradient",),
        "implementation_model_scope": (MODEL_IDS[3],),
        "evidence": _evidence(("d7_q3_lora_path", "training_dynamics")),
        "claim_boundary": "Q3 loss/LoRA training dynamics are model-specific and do not isolate the alignment objective from parameterization.",
    },
    {
        "interface_id": "bfloat16_autocast_training_forward",
        "system_layer": "training",
        "implementation_surface": "Q0--Q3 frozen BF16 autocast forward/loss path before backward, with FP16 GradScaler disabled",
        "component_ids": ("loss_gradient", "optimizer_effective_update"),
        "evidence": _evidence(("d7_optimizer_replay", "training_dynamics")),
        "claim_boundary": "The replay observes gradients produced by the frozen mixed-precision path, but no precision-matched control isolates BF16 rounding as a transfer mechanism or utility determinant.",
    },
    {
        "interface_id": "nonreentrant_gradient_checkpoint_recomputation",
        "system_layer": "training",
        "implementation_surface": "Q0--Q3 decoder activation checkpointing with use_reentrant=False during training forward/backward recomputation",
        "component_ids": ("loss_gradient", "layerwise_representation"),
        "evidence": _evidence(),
        "claim_boundary": "Gradient checkpointing is active in the frozen recipes, but no exact gradient/RNG equivalence artifact currently proves that recomputation is neutral, especially along Q3's stochastic adapter path.",
    },
    {
        "interface_id": "q3_input_activation_requires_grad_bridge",
        "system_layer": "training",
        "implementation_surface": "Q3 enable_input_require_grads bridge on frozen embedding outputs before non-reentrant checkpointed LoRA training",
        "component_ids": ("loss_gradient", "lora_parameterization", "token_embedding"),
        "implementation_model_scope": (MODEL_IDS[3],),
        "evidence": _evidence(
            ("d7_q3_lora_path", "training_dynamics"),
            ("d7_optimizer_replay", "training_dynamics"),
        ),
        "claim_boundary": "The bridge is active in the frozen Q3 loader and its downstream gradients are observed, but no bridge-only equivalence control attributes transfer failure or utility to activation requires-grad handling.",
    },
    {
        "interface_id": "q3_fp32_lora_bf16_base_cast_boundary",
        "system_layer": "training",
        "implementation_surface": "Q3 FP32 LoRA A/B tensors over a BF16 frozen base: PEFT casts projection input to adapter dtype and the scaled adapter result back to base-result dtype",
        "component_ids": ("loss_gradient", "lora_parameterization", "optimizer_effective_update"),
        "implementation_model_scope": (MODEL_IDS[3],),
        "evidence": _evidence(
            ("d7_q3_lora_path", "training_dynamics"),
            ("d7_optimizer_replay", "training_dynamics"),
        ),
        "claim_boundary": "Checkpoint tensors and PEFT source prove the mixed-dtype boundary, but no dtype-matched adapter/base control isolates its causal contribution or ranking utility.",
    },
    {
        "interface_id": "gradient_accumulation_and_global_clip",
        "system_layer": "training",
        "implementation_surface": "microbatch gradient accumulation, unscale, and frozen global-norm clipping before AdamW",
        "component_ids": ("loss_gradient", "optimizer_effective_update"),
        "evidence": _evidence(
            ("d7_optimizer_replay", "training_dynamics"),
        ),
        "claim_boundary": "Accumulated and clipped gradient diagnostics do not establish that clipping caused transfer failure or that a different threshold improves utility.",
    },
    {
        "interface_id": "adam_moment_preconditioned_direction",
        "system_layer": "training",
        "implementation_surface": "Adam first/second moments, bias correction, epsilon, and preconditioned update direction",
        "component_ids": ("optimizer_effective_update",),
        "evidence": _evidence(
            ("d7_optimizer_replay", "training_dynamics"),
        ),
        "claim_boundary": "A replayed preconditioned direction separates optimizer geometry from raw gradients but is not a randomized optimizer or utility control.",
    },
    {
        "interface_id": "decoupled_weight_decay_term",
        "system_layer": "training",
        "implementation_surface": "AdamW decoupled parameter-proportional weight-decay contribution",
        "component_ids": ("optimizer_effective_update",),
        "evidence": _evidence(
            ("d7_optimizer_replay", "training_dynamics"),
        ),
        "claim_boundary": "An exactly replayed decay term quantifies update composition but does not show that weight decay suppresses transferable preference information.",
    },
    {
        "interface_id": "learning_rate_scaled_effective_parameter_delta",
        "system_layer": "training",
        "implementation_surface": "scheduler-scaled joint moment-plus-decay parameter delta actually applied at each optimizer step",
        "component_ids": ("optimizer_effective_update",),
        "evidence": _evidence(
            ("d7_optimizer_replay", "training_dynamics"),
            ("d7_q2_parameter_update_geometry", "descriptive"),
            ("d7_q2_update_anisotropy", "descriptive"),
        ),
        "claim_boundary": "Effective-delta geometry can distinguish applied updates from gradients; magnitude or anisotropy alone does not establish positive transfer utility.",
    },
    {
        "interface_id": "lora_training_input_dropout",
        "system_layer": "training",
        "implementation_surface": "Q3 q_proj/v_proj LoRA input dropout with frozen p=0.05 before each rank-8 A down-projection during training and identity behavior at evaluation",
        "component_ids": ("lora_parameterization", "loss_gradient"),
        "implementation_model_scope": (MODEL_IDS[3],),
        "evidence": _evidence(
            ("d7_q3_lora_path", "training_dynamics"),
            ("d7_optimizer_replay", "training_dynamics"),
        ),
        "claim_boundary": "The frozen stochastic adapter path is implementation- and replay-backed, but no dropout-only control isolates its causal contribution or ranking utility.",
    },
    {
        "interface_id": "lora_q_low_rank_a_factor",
        "system_layer": "training",
        "implementation_surface": "Q3 q_proj rank-8 LoRA A down-projection after adapter dropout",
        "component_ids": ("lora_parameterization", "attention_query_key_routing"),
        "implementation_model_scope": (MODEL_IDS[3],),
        "evidence": _evidence(
            ("d7_q3_lora_path", "training_dynamics"),
            ("d7_optimizer_replay", "training_dynamics"),
        ),
        "claim_boundary": "An individual A-factor basis is gauge-dependent; its coordinates or norms cannot be promoted as a query-routing design direction.",
    },
    {
        "interface_id": "lora_q_low_rank_b_factor",
        "system_layer": "training",
        "implementation_surface": "Q3 q_proj LoRA B up-projection from rank 8 to query-head output space",
        "component_ids": ("lora_parameterization", "attention_query_key_routing"),
        "implementation_model_scope": (MODEL_IDS[3],),
        "evidence": _evidence(
            ("d7_q3_lora_path", "training_dynamics"),
            ("d7_optimizer_replay", "training_dynamics"),
        ),
        "claim_boundary": "An individual B-factor basis is gauge-dependent; B-only replay is diagnostic training geometry, not a causal parameterization result.",
    },
    {
        "interface_id": "lora_q_effective_delta_weight",
        "system_layer": "training",
        "implementation_surface": "Q3 q_proj gauge-invariant effective adapter delta (alpha/r) times B@A composed with the frozen base projection",
        "component_ids": ("lora_parameterization", "attention_query_key_routing"),
        "implementation_model_scope": (MODEL_IDS[3],),
        "evidence": _evidence(
            ("d7_q3_lora_path", "training_dynamics"),
            ("d7_optimizer_replay", "training_dynamics"),
            ("d7_q3_lora_head_geometry", "descriptive"),
        ),
        "claim_boundary": "B@A is gauge-invariant but its geometry alone cannot attribute transfer failure to q_proj LoRA or establish ranking utility.",
    },
    {
        "interface_id": "lora_v_low_rank_a_factor",
        "system_layer": "training",
        "implementation_surface": "Q3 v_proj rank-8 LoRA A down-projection after adapter dropout",
        "component_ids": ("lora_parameterization", "attention_value_transport"),
        "implementation_model_scope": (MODEL_IDS[3],),
        "evidence": _evidence(
            ("d7_q3_lora_path", "training_dynamics"),
            ("d7_optimizer_replay", "training_dynamics"),
        ),
        "claim_boundary": "An individual A-factor basis is gauge-dependent; its coordinates or norms cannot be promoted as a value-transport design direction.",
    },
    {
        "interface_id": "lora_v_low_rank_b_factor",
        "system_layer": "training",
        "implementation_surface": "Q3 v_proj LoRA B up-projection from rank 8 to value-head output space",
        "component_ids": ("lora_parameterization", "attention_value_transport"),
        "implementation_model_scope": (MODEL_IDS[3],),
        "evidence": _evidence(
            ("d7_q3_lora_path", "training_dynamics"),
            ("d7_optimizer_replay", "training_dynamics"),
        ),
        "claim_boundary": "An individual B-factor basis is gauge-dependent; B-only replay is diagnostic training geometry, not a causal parameterization result.",
    },
    {
        "interface_id": "lora_v_effective_delta_weight",
        "system_layer": "training",
        "implementation_surface": "Q3 v_proj gauge-invariant effective adapter delta (alpha/r) times B@A composed with the frozen base projection",
        "component_ids": ("lora_parameterization", "attention_value_transport"),
        "implementation_model_scope": (MODEL_IDS[3],),
        "evidence": _evidence(
            ("d7_q3_lora_path", "training_dynamics"),
            ("d7_optimizer_replay", "training_dynamics"),
            ("d7_q3_lora_head_geometry", "descriptive"),
        ),
        "claim_boundary": "B@A is gauge-invariant but its geometry alone cannot attribute transfer failure to v_proj LoRA or establish ranking utility.",
    },
)


CROSS_INTERFACE_EVIDENCE = {
    "d1_activation_anisotropy": (
        "Activation anisotropy compares representation geometry across multiple "
        "functional interfaces and cannot be assigned to one operator."
    ),
    "d1_candidate_block_flow": (
        "Candidate-common and candidate-relative flow is a multi-block trajectory, "
        "not a single implementation interface."
    ),
    "d1_candidate_residual_geometry": (
        "Candidate residual geometry spans representation and native-score relations."
    ),
    "d1_preference_subspace_geometry": (
        "Preference-subspace geometry is a derived cross-interface relation."
    ),
    "d1_query_causal_floor": (
        "The query causal floor is a multi-interface boundary control."
    ),
    "d2_q3_native_gate": (
        "The Q3 native gate validates model-wide readout scope before downstream "
        "families; it is not a component-local effect."
    ),
    "d6_q0_q1_branches": (
        "Q0/Q1 branch evidence jointly spans attention, MLP, residual, and model-"
        "specific readout semantics."
    ),
    "d6_q0_trajectory": (
        "The Q0 trajectory covers every block boundary and the specialized native "
        "readout rather than one exact interface."
    ),
    "d7_objective_common_nullspace": (
        "The common objective-nullspace analysis compares Q2 and Q3 loss/gradient "
        "pathways and cannot be assigned to one model-specific loss operator."
    ),
}


# These are report-stage debts, not newly authorized experiment families.  The
# gates make the smallest missing test explicit so that descriptive coverage
# cannot silently become an operator or architecture claim at closeout.
OPERATOR_CAUSAL_DEBT_CONTRACT = {
    "token_embedding_lookup": {
        "debt_class": "inference_operator",
        "minimum_future_evidence": "preregistered_embedding_interface_intervention",
        "smallest_falsification_gate": (
            "Patch query, history, and candidate embedding states separately under "
            "same-token identity, same/wrong-history specificity, scale, sign, and "
            "random-direction controls while preserving token IDs, positions, and mask; "
            "replicate the functional result in Q2/Q3 without borrowing tied lm-head evidence."
        ),
    },
    "input_rmsnorm_variance_rescale_and_gain": {
        "debt_class": "inference_operator",
        "minimum_future_evidence": "preregistered_rmsnorm_operator_intervention",
        "smallest_falsification_gate": (
            "At a functionally predeclared Q2/Q3 node, hold the incoming residual and "
            "all downstream weights fixed while intervening separately on RMS variance "
            "rescaling and learned gain; include frozen-operator identity, output-norm-"
            "matched direction controls, same/wrong-history specificity, reverse removal, "
            "and cross-model replication. Pre/post state patches do not satisfy this gate."
        ),
    },
    "q_head_rmsnorm_variance_rescale_and_gain": {
        "debt_class": "inference_operator",
        "minimum_future_evidence": "preregistered_q_head_rmsnorm_operator_intervention",
        "smallest_falsification_gate": (
            "Hold q_proj output, K/V, RoPE phase, mask, softmax, and o_proj fixed while "
            "intervening separately on query-head RMS variance rescaling and learned gain; "
            "require frozen identity, post-norm magnitude/direction controls, same/wrong-"
            "history specificity, reverse removal, and Q2/Q3 replication. A downstream "
            "RoPE intervention or pre/post geometry is not a q_norm operator test."
        ),
    },
    "k_head_rmsnorm_variance_rescale_and_gain": {
        "debt_class": "inference_operator",
        "minimum_future_evidence": "preregistered_k_head_rmsnorm_operator_intervention",
        "smallest_falsification_gate": (
            "Hold k_proj output, Q/V, RoPE phase, mask, softmax, and o_proj fixed while "
            "intervening separately on key-head RMS variance rescaling and learned gain; "
            "require frozen identity, post-norm magnitude/direction controls, same/wrong-"
            "history specificity, reverse removal, and Q2/Q3 replication. A downstream "
            "RoPE intervention or pre/post geometry is not a k_norm operator test."
        ),
    },
    "q3_q_lora_scaled_adapter_injection": {
        "debt_class": "inference_operator",
        "minimum_future_evidence": "preregistered_q3_q_adapter_branch_intervention",
        "smallest_falsification_gate": (
            "Within frozen Q3, hold the base q_proj output, adapter input, v-adapter, "
            "q_norm, RoPE, K/V, mask, and downstream score fixed while intervening only "
            "on the complete scaled q-adapter contribution; require exact re-add identity, "
            "magnitude/direction/random controls, full/null/wrong-user specificity, reverse "
            "removal, and frozen utility. A/B norms or B@A update geometry do not satisfy "
            "this operator gate."
        ),
    },
    "q3_v_lora_scaled_adapter_injection": {
        "debt_class": "inference_operator",
        "minimum_future_evidence": "preregistered_q3_v_adapter_branch_intervention",
        "smallest_falsification_gate": (
            "Within frozen Q3, hold the base v_proj output, adapter input, q-adapter, "
            "Q/K routing, mask, o_proj, and downstream score fixed while intervening only "
            "on the complete scaled v-adapter contribution; require exact re-add identity, "
            "magnitude/direction/random controls, full/null/wrong-user specificity, reverse "
            "removal, and frozen utility. A/B norms or B@A update geometry do not satisfy "
            "this operator gate."
        ),
    },
    "attention_residual_addition": {
        "debt_class": "inference_operator",
        "minimum_future_evidence": "preregistered_attention_residual_composition_intervention",
        "smallest_falsification_gate": (
            "Hold the incoming residual and attention o_proj increment fixed at a "
            "functionally predeclared Q2/Q3 node while changing only their composition "
            "rule or coefficient; require alpha=1 identity, magnitude/direction/random "
            "controls, same/wrong-history specificity, reverse removal, and frozen utility. "
            "Recomposing r+a or patching its output state is not an operator intervention."
        ),
    },
    "post_attention_rmsnorm_variance_rescale_and_gain": {
        "debt_class": "inference_operator",
        "minimum_future_evidence": "preregistered_rmsnorm_operator_intervention",
        "smallest_falsification_gate": (
            "Hold the post-attention residual and MLP weights fixed while separately "
            "intervening on the second RMS variance rescaling and learned gain; require "
            "frozen identity, output-norm-matched controls, same/wrong-history specificity, "
            "reverse removal, and Q2/Q3 replication so induced MLP changes are not borrowed "
            "as normalization-operator evidence."
        ),
    },
    "mlp_residual_addition": {
        "debt_class": "inference_operator",
        "minimum_future_evidence": "preregistered_mlp_residual_composition_intervention",
        "smallest_falsification_gate": (
            "Hold the post-attention residual and MLP down_proj increment fixed at a "
            "functionally predeclared Q2/Q3 node while changing only their composition "
            "rule or coefficient; require alpha=1 identity, magnitude/direction/random "
            "controls, same/wrong-history specificity, reverse removal, and frozen utility. "
            "A full block-state ceiling cannot establish this operator."
        ),
    },
    "final_rmsnorm_variance_rescale_and_gain": {
        "debt_class": "inference_operator",
        "minimum_future_evidence": "preregistered_final_rmsnorm_operator_intervention",
        "smallest_falsification_gate": (
            "Hold the final residual, readout positions, tied rows, and native score algebra "
            "fixed while intervening separately on final RMS variance rescaling and gain; "
            "require frozen identity, output-norm-matched direction controls, same/wrong-"
            "history specificity, reverse removal, Q2/Q3 replication, and frozen ranking "
            "utility. Exact formula recomposition is only a mechanical gate."
        ),
    },
    "gqa_query_to_kv_grouping": {
        "debt_class": "inference_operator",
        "minimum_future_evidence": "preregistered_identity_checked_cross_model_intervention",
        "smallest_falsification_gate": (
            "At a functionally aligned, predeclared depth region, intervene on the "
            "Q-to-shared-KV grouping while preserving Q/K/V values, mask, head count, "
            "and o_proj; require identity, random/permutation, wrong-history, and "
            "Q2/Q3 replication gates before attributing the failure to GQA topology."
        ),
    },
    "attention_scaled_qk_logits": {
        "debt_class": "inference_operator",
        "minimum_future_evidence": "preregistered_qk_logit_formation_intervention",
        "smallest_falsification_gate": (
            "At a functionally predeclared Q2/Q3 node, intervene on the complete "
            "pre-mask scaled QK-logit tensor while holding Q, K, V, mask, softmax, "
            "dropout, and o_proj contracts fixed; require recomposed identity, same/"
            "wrong-history specificity, scale, sign, random-direction, reverse-removal, "
            "and cross-model gates. Existing edge masking does not satisfy this operator test."
        ),
    },
    "autoregressive_causal_attention_mask": {
        "debt_class": "inference_operator",
        "minimum_future_evidence": "preregistered_prefix_visibility_topology_control",
        "smallest_falsification_gate": (
            "Compare the frozen autoregressive mask with a position- and token-matched "
            "alternative that changes only query/history/candidate prefix visibility while "
            "preserving every answer/continuation causal boundary and label isolation; "
            "require identity for the frozen topology, leakage audits, same/wrong-history "
            "specificity, Q2/Q3 replication, multiple seeds, and frozen ranking utility."
        ),
    },
    "kv_cache_phase_boundary": {
        "debt_class": "inference_operator",
        "minimum_future_evidence": "preregistered_phase_matched_q1_intervention",
        "smallest_falsification_gate": (
            "Within Q1, compare cache-preserving same-request, wrong-history, and "
            "phase-matched cache replacement at fixed prefix/continuation boundaries; "
            "require identical token, position, and attention-mask contracts. Keep the "
            "claim Q1-scoped until another cache-based model pathway is registered."
        ),
    },
    "mlp_gate_projection": {
        "debt_class": "inference_operator",
        "minimum_future_evidence": "preregistered_swiglu_stage_intervention",
        "smallest_falsification_gate": (
            "Patch the full gate_proj state at a functionally predeclared node while "
            "holding up_proj fixed; require identity, same/wrong-history specificity, "
            "random-direction controls, reverse removal, and Q2/Q3 replication without "
            "selecting neurons from the outcome."
        ),
    },
    "mlp_up_projection": {
        "debt_class": "inference_operator",
        "minimum_future_evidence": "preregistered_swiglu_stage_intervention",
        "smallest_falsification_gate": (
            "Patch the full up_proj state at the same predeclared functional node while "
            "holding the gate path fixed; require the paired identity, specificity, "
            "random-direction, reverse-removal, and Q2/Q3 gates without neuron selection."
        ),
    },
    "mlp_silu_gate": {
        "debt_class": "inference_operator",
        "minimum_future_evidence": "preregistered_nonlinearity_intervention",
        "smallest_falsification_gate": (
            "Use a magnitude-matched pre/post-SiLU intervention that leaves up_proj and "
            "down_proj unchanged; separate nonlinear gating from scale, sign, and random "
            "direction controls and replicate the signed effect in Q2/Q3."
        ),
    },
    "mlp_swiglu_product": {
        "debt_class": "inference_operator",
        "minimum_future_evidence": "preregistered_nonlinear_composition_intervention",
        "smallest_falsification_gate": (
            "Intervene on the complete SwiGLU product with gate-only, up-only, recomposed "
            "identity, scale, sign, and random controls; require reverse necessity and "
            "Q2/Q3 replication before claiming nonlinear composition as the bottleneck."
        ),
    },
    "q0_pointwise_bce_loss": {
        "debt_class": "training_mechanism",
        "minimum_future_evidence": "preregistered_multiseed_training_control",
        "smallest_falsification_gate": (
            "Within Q0, hold train records, initialization, optimizer, update budget, "
            "and evaluator fixed while replacing or reweighting pointwise BCE; require "
            "multiple seeds, strict-transfer utility, recurrence/overlap surfaces, and "
            "exact-null recovery before attributing failure to this loss."
        ),
    },
    "q1_normalized_response_nll": {
        "debt_class": "training_mechanism",
        "minimum_future_evidence": "preregistered_multiseed_training_control",
        "smallest_falsification_gate": (
            "Within Q1, hold templates, candidate responses, train records, optimizer, "
            "and update budget fixed while changing only normalized response NLL; require "
            "multiple seeds and the frozen utility/surface/null-recovery gates."
        ),
    },
    "q2_pairwise_ranknet_loss": {
        "debt_class": "training_mechanism",
        "minimum_future_evidence": "preregistered_multiseed_loss_term_ablation",
        "smallest_falsification_gate": (
            "Hold Q2 data, initialization, optimizer, total update norm, and the ListNet "
            "term fixed while ablating or reweighting RankNet; require multiple seeds and "
            "frozen utility, surface, and exact-null-recovery gates."
        ),
    },
    "q2_listwise_listnet_loss": {
        "debt_class": "training_mechanism",
        "minimum_future_evidence": "preregistered_multiseed_loss_term_ablation",
        "smallest_falsification_gate": (
            "Hold Q2 data, initialization, optimizer, total update norm, and the RankNet "
            "term fixed while ablating or reweighting tie-aware ListNet; require multiple "
            "seeds and frozen utility, surface, and exact-null-recovery gates."
        ),
    },
    "q3_alignment_nll_loss": {
        "debt_class": "training_mechanism",
        "minimum_future_evidence": "preregistered_multiseed_loss_parameterization_control",
        "smallest_falsification_gate": (
            "Within Q3, hold LoRA targets/rank, train records, initialization, optimizer, "
            "and update budget fixed while changing only alignment NLL; require multiple "
            "seeds and frozen utility/surface/null-recovery gates so loss is separated "
            "from low-rank parameterization."
        ),
    },
    "bfloat16_autocast_training_forward": {
        "debt_class": "training_mechanism",
        "minimum_future_evidence": "preregistered_precision_equivalence_control",
        "smallest_falsification_gate": (
            "Hold data order, initialization, dropout masks, accumulation, loss, optimizer, "
            "schedule, and update budget fixed while comparing the registered BF16 path "
            "with an FP32 reference and a numerically matched control; audit per-family raw "
            "gradients and applied deltas before any multiseed frozen-surface and utility "
            "claim. A single BF16 optimizer replay does not satisfy this gate."
        ),
    },
    "nonreentrant_gradient_checkpoint_recomputation": {
        "debt_class": "training_mechanism",
        "minimum_future_evidence": "preregistered_checkpoint_rng_equivalence_control",
        "smallest_falsification_gate": (
            "On identical microbatches, parameters, dropout masks, and RNG states, compare "
            "non-reentrant checkpointing on/off while holding autocast and loss fixed; require "
            "per-parameter gradient and effective-update equivalence within predeclared bounds, "
            "explicit Q3 adapter-dropout mask preservation, and multiple seeds plus frozen "
            "utility only if a reproducible difference survives the mechanical gate."
        ),
    },
    "q3_input_activation_requires_grad_bridge": {
        "debt_class": "training_mechanism",
        "minimum_future_evidence": "preregistered_q3_gradient_bridge_equivalence_control",
        "smallest_falsification_gate": (
            "Within Q3, hold checkpointing, data, parameters, dropout masks, autocast, loss, "
            "and optimizer fixed while comparing the registered embedding-output requires-grad "
            "bridge with a checkpoint-safe reference; require complete q/v LoRA gradient and "
            "effective-update coverage, exact bridge identity where expected, and frozen utility "
            "only after any reproducible gradient difference survives multiple seeds."
        ),
    },
    "q3_fp32_lora_bf16_base_cast_boundary": {
        "debt_class": "training_mechanism",
        "minimum_future_evidence": "preregistered_q3_adapter_dtype_boundary_control",
        "smallest_falsification_gate": (
            "Within Q3, hold data, initialization in function space, dropout masks, rank, "
            "targets, loss, optimizer, schedule, and update budget fixed while comparing the "
            "registered BF16-base/FP32-adapter cast boundary with dtype-aligned controls; "
            "match initial and integrated B@A function-update norms, audit cast-local error, "
            "raw gradients and applied deltas, then require multiple seeds and frozen utility."
        ),
    },
    "gradient_accumulation_and_global_clip": {
        "debt_class": "training_mechanism",
        "minimum_future_evidence": "preregistered_multiseed_clipping_control",
        "smallest_falsification_gate": (
            "Hold data, initialization, accumulation boundary, raw gradient stream, AdamW, "
            "schedule, and update budget fixed while changing only the preregistered clip "
            "rule or threshold; match effective update norm where feasible and require "
            "multiple seeds plus frozen surface and utility gates."
        ),
    },
    "adam_moment_preconditioned_direction": {
        "debt_class": "training_mechanism",
        "minimum_future_evidence": "preregistered_multiseed_preconditioner_control",
        "smallest_falsification_gate": (
            "Hold the raw gradient stream, clipping, decay, schedule, data, initialization, "
            "and budget fixed while replacing or randomizing only the Adam moment "
            "preconditioner; match update norm and require multiple seeds and frozen utility."
        ),
    },
    "decoupled_weight_decay_term": {
        "debt_class": "training_mechanism",
        "minimum_future_evidence": "preregistered_multiseed_weight_decay_control",
        "smallest_falsification_gate": (
            "Hold raw gradients, clipping, Adam moments, schedule, data, initialization, "
            "and optimizer-step budget fixed while changing only decoupled weight decay; "
            "require multiple seeds, parameter-norm accounting, and frozen utility gates."
        ),
    },
    "learning_rate_scaled_effective_parameter_delta": {
        "debt_class": "training_mechanism",
        "minimum_future_evidence": "preregistered_multiseed_schedule_control",
        "smallest_falsification_gate": (
            "Hold raw gradients, clipping, Adam moments, decay, data, initialization, and "
            "step budget fixed while changing only the preregistered learning-rate schedule; "
            "match integrated effective-update norm and require multiple seeds and frozen "
            "surface and ranking-utility gates."
        ),
    },
    "lora_training_input_dropout": {
        "debt_class": "training_mechanism",
        "minimum_future_evidence": "preregistered_multiseed_adapter_dropout_control",
        "smallest_falsification_gate": (
            "Within Q3, hold data, initialization, q/v targets, rank, loss, AdamW, "
            "schedule, update budget, and RNG protocol fixed while changing only LoRA "
            "input dropout; match integrated gauge-invariant B@A function-update norm "
            "where feasible and require multiple seeds plus frozen surface, null-recovery, "
            "and ranking-utility gates."
        ),
    },
    "lora_q_low_rank_a_factor": {
        "debt_class": "training_mechanism",
        "minimum_future_evidence": "preregistered_gauge_aware_factor_control",
        "smallest_falsification_gate": (
            "Within Q3, perturb the q_proj A subspace while holding B and v_proj fixed, "
            "compare orthogonal-gauge-equivalent parameterizations, and match effective "
            "B@A update norm, data, optimizer, budget, and seeds before applying frozen "
            "utility gates; coordinate-wise A effects are ineligible."
        ),
    },
    "lora_q_low_rank_b_factor": {
        "debt_class": "training_mechanism",
        "minimum_future_evidence": "preregistered_gauge_aware_factor_control",
        "smallest_falsification_gate": (
            "Within Q3, perturb the q_proj B output subspace while holding A and v_proj "
            "fixed, require orthogonal-gauge invariance and matched effective B@A update "
            "norm, data, optimizer, budget, and seeds, then apply frozen utility gates."
        ),
    },
    "lora_q_effective_delta_weight": {
        "debt_class": "training_mechanism",
        "minimum_future_evidence": "preregistered_function_space_parameterization_control",
        "smallest_falsification_gate": (
            "Compare the complete q_proj effective delta B@A with a parameter-budget and "
            "function-update-norm matched full-rank or alternative low-rank control while "
            "holding v_proj, data, initialization, optimizer, budget, and multiple seeds "
            "fixed; require frozen surface and ranking-utility gates."
        ),
    },
    "lora_v_low_rank_a_factor": {
        "debt_class": "training_mechanism",
        "minimum_future_evidence": "preregistered_gauge_aware_factor_control",
        "smallest_falsification_gate": (
            "Within Q3, perturb the v_proj A subspace while holding B and q_proj fixed, "
            "compare orthogonal-gauge-equivalent parameterizations, and match effective "
            "B@A update norm, data, optimizer, budget, and seeds before applying frozen "
            "utility gates; coordinate-wise A effects are ineligible."
        ),
    },
    "lora_v_low_rank_b_factor": {
        "debt_class": "training_mechanism",
        "minimum_future_evidence": "preregistered_gauge_aware_factor_control",
        "smallest_falsification_gate": (
            "Within Q3, perturb the v_proj B output subspace while holding A and q_proj "
            "fixed, require orthogonal-gauge invariance and matched effective B@A update "
            "norm, data, optimizer, budget, and seeds, then apply frozen utility gates."
        ),
    },
    "lora_v_effective_delta_weight": {
        "debt_class": "training_mechanism",
        "minimum_future_evidence": "preregistered_function_space_parameterization_control",
        "smallest_falsification_gate": (
            "Compare the complete v_proj effective delta B@A with a parameter-budget and "
            "function-update-norm matched full-rank or alternative low-rank control while "
            "holding q_proj, data, initialization, optimizer, budget, and multiple seeds "
            "fixed; require frozen surface and ranking-utility gates."
        ),
    },
}


def build_transformer_interface_coverage(
    *,
    completed_formal: set[str],
    completed_supplements: set[str],
    supplement_model_scopes: Mapping[str, set[str]],
    supplement_component_scopes: Mapping[str, set[str]],
) -> dict[str, Any]:
    """Return exact-interface completion without inspecting scientific effects."""

    if not completed_formal.issubset(EXPECTED_DELIVERABLES):
        raise ValueError("unknown completed formal evidence in interface inventory")
    if not completed_supplements.issubset(EXPECTED_SUPPLEMENT_IDS):
        raise ValueError("unknown completed supplement in interface inventory")
    if set(supplement_model_scopes) != set(EXPECTED_SUPPLEMENT_IDS) or any(
        not scope or not set(scope).issubset(MODEL_IDS)
        for scope in supplement_model_scopes.values()
    ):
        raise ValueError("supplement model-scope coverage drift in interface inventory")
    if set(supplement_component_scopes) != set(EXPECTED_SUPPLEMENT_IDS) or any(
        not scope or not set(scope).issubset(COMPONENT_IDS)
        for scope in supplement_component_scopes.values()
    ):
        raise ValueError("supplement component-scope coverage drift in interface inventory")

    rows = []
    seen = set()
    observed_roles = {
        evidence["role"]
        for raw in TRANSFORMER_INTERFACE_INVENTORY
        for evidence in raw["evidence"]
    }
    if set(ROLE_CLAIM_CEILINGS) != observed_roles:
        raise ValueError("interface evidence-role claim-ceiling coverage drift")
    for raw in TRANSFORMER_INTERFACE_INVENTORY:
        interface_id = str(raw["interface_id"])
        if interface_id in seen:
            raise ValueError(f"duplicate Transformer interface: {interface_id}")
        seen.add(interface_id)
        if raw["system_layer"] not in SYSTEM_LAYERS:
            raise ValueError(f"invalid system layer for Transformer interface: {interface_id}")
        components = set(raw["component_ids"])
        if not components or not components.issubset(COMPONENT_IDS):
            raise ValueError(f"invalid components for Transformer interface: {interface_id}")
        implementation_model_scope = set(
            raw.get("implementation_model_scope", MODEL_IDS)
        )
        if not implementation_model_scope or not implementation_model_scope.issubset(
            MODEL_IDS
        ):
            raise ValueError(
                f"invalid implementation model scope for Transformer interface: {interface_id}"
            )
        evidence_rows = list(raw["evidence"])
        evidence_ids = {row["evidence_id"] for row in evidence_rows}
        if len(evidence_ids) != len(evidence_rows) or not evidence_ids.issubset(EVIDENCE_IDS):
            raise ValueError(f"invalid evidence for Transformer interface: {interface_id}")
        if any(not row["role"] for row in evidence_rows):
            raise ValueError(f"empty evidence role for Transformer interface: {interface_id}")

        completed = evidence_ids & (completed_formal | completed_supplements)
        registered_models = set()
        completed_models = set()
        causal_registered = set()
        causal_completed = set()
        normalized_evidence = []
        for evidence in evidence_rows:
            evidence_id = evidence["evidence_id"]
            if evidence_id in EXPECTED_DELIVERABLES:
                if not any(
                    evidence_id in COMPONENT_ALLOWED_DELIVERABLES[component_id]
                    for component_id in components
                ):
                    raise ValueError(
                        "formal evidence is outside interface component scope: "
                        f"{interface_id}/{evidence_id}"
                    )
                model_scope = set(DELIVERABLE_MODEL_COVERAGE[evidence_id])
                evidence_kind = "formal_deliverable"
                is_completed = evidence_id in completed_formal
            else:
                if not (components & set(supplement_component_scopes[evidence_id])):
                    raise ValueError(
                        "supplement evidence is outside interface component scope: "
                        f"{interface_id}/{evidence_id}"
                    )
                model_scope = set(supplement_model_scopes[evidence_id])
                evidence_kind = "supplement"
                is_completed = evidence_id in completed_supplements
            source_model_scope = set(model_scope)
            model_scope &= implementation_model_scope
            if not model_scope:
                raise ValueError(
                    "interface evidence has no implementation-scope model: "
                    f"{interface_id}/{evidence_id}"
                )
            registered_models |= model_scope
            if is_completed:
                completed_models |= model_scope
            if evidence["role"] in CAUSAL_ROLES:
                causal_registered.add(evidence_id)
                if is_completed:
                    causal_completed.add(evidence_id)
            normalized_evidence.append(
                {
                    "evidence_id": evidence_id,
                    "evidence_kind": evidence_kind,
                    "role": evidence["role"],
                    "model_scope": sorted(model_scope),
                    "source_evidence_model_scope": sorted(source_model_scope),
                    "completed": is_completed,
                }
            )
        rows.append(
            {
                "interface_id": interface_id,
                "system_layer": raw["system_layer"],
                "implementation_surface": raw["implementation_surface"],
                "component_ids": sorted(components),
                "registered_evidence": normalized_evidence,
                "registered_evidence_count": len(evidence_rows),
                "completed_evidence_count": len(completed),
                "any_evidence_completed": bool(completed),
                "all_registered_evidence_completed": bool(evidence_ids)
                and completed == evidence_ids,
                "causal_role_registered": bool(causal_registered),
                "causal_role_completed": bool(causal_completed),
                "causal_role_evidence_registered": sorted(causal_registered),
                "causal_role_evidence_completed": sorted(causal_completed),
                "registered_claim_ceiling": _maximum_claim_ceiling(
                    evidence["role"] for evidence in normalized_evidence
                ),
                "completed_artifact_claim_ceiling": _maximum_claim_ceiling(
                    evidence["role"]
                    for evidence in normalized_evidence
                    if evidence["completed"]
                ),
                "claim_ceiling_is_artifact_availability_only": True,
                "actual_scientific_evidence_level_inferred": False,
                "model_scope_registered": sorted(registered_models),
                "model_scope_completed": sorted(completed_models),
                "implementation_model_scope": sorted(implementation_model_scope),
                "claim_boundary": raw["claim_boundary"],
                "operator_attribution_status_from_artifact_availability": (
                    "not_inferred_functional_causal_role_available"
                    if causal_registered
                    else "not_inferred_no_functional_causal_role_artifact"
                ),
                "operator_attribution_inferred_from_artifact_availability": False,
                "scientific_support_inferred_from_completion": False,
            }
        )

    if not (set(BLOCK_NODE_IDS) | set(FINAL_NODE_IDS)).issubset(seen):
        raise ValueError("hookable Transformer node is absent from interface inventory")

    mapped_evidence = {
        evidence["evidence_id"]
        for row in rows
        for evidence in row["registered_evidence"]
    }
    cross_ids = set(CROSS_INTERFACE_EVIDENCE)
    if mapped_evidence & cross_ids or mapped_evidence | cross_ids != EVIDENCE_IDS:
        raise ValueError("exact and cross-interface evidence disposition is not exhaustive")
    cross_interface_rows = []
    for evidence_id, boundary in sorted(CROSS_INTERFACE_EVIDENCE.items()):
        if evidence_id in EXPECTED_DELIVERABLES:
            evidence_kind = "formal_deliverable"
            completed = evidence_id in completed_formal
            model_scope = set(DELIVERABLE_MODEL_COVERAGE[evidence_id])
            component_scope = {
                component_id
                for component_id in COMPONENT_IDS
                if evidence_id in COMPONENT_ALLOWED_DELIVERABLES[component_id]
            }
        else:
            evidence_kind = "supplement"
            completed = evidence_id in completed_supplements
            model_scope = set(supplement_model_scopes[evidence_id])
            component_scope = set(supplement_component_scopes[evidence_id])
        if not component_scope:
            raise ValueError(
                f"cross-interface evidence has no component scope: {evidence_id}"
            )
        cross_interface_rows.append(
            {
                "evidence_id": evidence_id,
                "evidence_kind": evidence_kind,
                "completed": completed,
                "model_scope": sorted(model_scope),
                "component_scope": sorted(component_scope),
                "disposition": "cross_interface_or_scope_gate",
                "claim_boundary": boundary,
                "scientific_support_inferred_from_completion": False,
            }
        )

    causal_debt_ids = {
        row["interface_id"] for row in rows if not row["causal_role_registered"]
    }
    if set(OPERATOR_CAUSAL_DEBT_CONTRACT) != causal_debt_ids:
        raise ValueError("operator causal-debt contract differs from interface coverage")
    rows_by_id = {row["interface_id"]: row for row in rows}
    operator_causal_debt = []
    for interface_id, contract in sorted(OPERATOR_CAUSAL_DEBT_CONTRACT.items()):
        interface = rows_by_id[interface_id]
        operator_causal_debt.append(
            {
                "interface_id": interface_id,
                "system_layer": interface["system_layer"],
                "component_ids": list(interface["component_ids"]),
                "debt_class": contract["debt_class"],
                "registered_evidence_roles": sorted(
                    {row["role"] for row in interface["registered_evidence"]}
                ),
                "registered_model_scope": list(interface["model_scope_registered"]),
                "minimum_future_evidence": contract["minimum_future_evidence"],
                "smallest_falsification_gate": contract["smallest_falsification_gate"],
                "current_stage_disposition": "unresolved_no_operator_causal_claim",
                "active_experiment_authorized": False,
                "can_rank_architecture_from_current_evidence": False,
                "scientific_support_inferred_from_completion": False,
            }
        )
    system_layer_coverage = {}
    for system_layer in sorted(SYSTEM_LAYERS):
        layer_rows = [row for row in rows if row["system_layer"] == system_layer]
        if not layer_rows:
            raise ValueError(f"system layer has no exact interfaces: {system_layer}")
        system_layer_coverage[system_layer] = {
            "interface_count": len(layer_rows),
            "interfaces_with_any_completed_evidence": sum(
                row["any_evidence_completed"] for row in layer_rows
            ),
            "interfaces_with_registered_causal_role_evidence": sum(
                row["causal_role_registered"] for row in layer_rows
            ),
            "interfaces_with_completed_causal_role_evidence": sum(
                row["causal_role_completed"] for row in layer_rows
            ),
            "operator_causal_debt_count": sum(
                not row["causal_role_registered"] for row in layer_rows
            ),
            "registered_claim_ceiling_counts": _claim_ceiling_counts(
                row["registered_claim_ceiling"] for row in layer_rows
            ),
            "completed_artifact_claim_ceiling_counts": _claim_ceiling_counts(
                row["completed_artifact_claim_ceiling"] for row in layer_rows
            ),
            "interface_ids": sorted(row["interface_id"] for row in layer_rows),
            "scientific_support_inferred_from_completion": False,
        }

    return {
        "interface_count": len(rows),
        "interfaces": rows,
        "interfaces_with_any_completed_evidence": sum(
            row["any_evidence_completed"] for row in rows
        ),
        "interfaces_with_all_registered_evidence_completed": sum(
            row["all_registered_evidence_completed"] for row in rows
        ),
        "interfaces_with_registered_causal_role_evidence": sum(
            row["causal_role_registered"] for row in rows
        ),
        "interfaces_with_completed_causal_role_evidence": sum(
            row["causal_role_completed"] for row in rows
        ),
        "interfaces_without_any_completed_evidence": sorted(
            row["interface_id"] for row in rows if not row["any_evidence_completed"]
        ),
        "interfaces_without_registered_causal_role_evidence": sorted(
            row["interface_id"] for row in rows if not row["causal_role_registered"]
        ),
        "interfaces_with_registered_but_pending_causal_role_evidence": sorted(
            row["interface_id"]
            for row in rows
            if row["causal_role_registered"] and not row["causal_role_completed"]
        ),
        "interfaces_with_functional_causal_role_but_no_operator_attribution_inferred": sorted(
            row["interface_id"] for row in rows if row["causal_role_registered"]
        ),
        "operator_attribution_inferred_from_artifact_availability_count": sum(
            row["operator_attribution_inferred_from_artifact_availability"]
            for row in rows
        ),
        "operator_attribution_unresolved_from_artifact_availability_count": sum(
            not row["operator_attribution_inferred_from_artifact_availability"]
            for row in rows
        ),
        "functional_causal_role_is_operator_attribution": False,
        "operator_causal_debt_count": len(operator_causal_debt),
        "operator_causal_debt_class_counts": {
            debt_class: sum(
                row["debt_class"] == debt_class for row in operator_causal_debt
            )
            for debt_class in ("inference_operator", "training_mechanism")
        },
        "operator_causal_debt": operator_causal_debt,
        "operator_causal_debt_is_lower_bound": True,
        "operator_attribution_inferred_for_other_interfaces": False,
        "new_experiment_family_authorized_by_debt_ledger": False,
        "system_layer_coverage": system_layer_coverage,
        "registered_claim_ceiling_counts": _claim_ceiling_counts(
            row["registered_claim_ceiling"] for row in rows
        ),
        "completed_artifact_claim_ceiling_counts": _claim_ceiling_counts(
            row["completed_artifact_claim_ceiling"] for row in rows
        ),
        "claim_ceilings_are_artifact_availability_only": True,
        "actual_scientific_evidence_levels_inferred": False,
        "direct_interface_evidence_count": len(mapped_evidence),
        "cross_interface_evidence_count": len(cross_interface_rows),
        "registered_evidence_count": len(EVIDENCE_IDS),
        "cross_interface_evidence": cross_interface_rows,
        "all_registered_evidence_has_exact_or_cross_interface_disposition": True,
        "scientific_support_inferred_from_completion": False,
        "interpretation": (
            "Exact-interface completion exposes implementation coverage only. "
            "Descriptive, interventional-localization, sufficiency, necessity, and "
            "design-gate roles remain distinct; no completion flag is scientific support. "
            "A functional causal-role artifact at an interface does not by itself "
            "identify the implementation operator that produced that state. "
            "The causal-debt ledger is a lower bound containing interfaces without even "
            "a registered causal-role artifact; it does not imply operator attribution "
            "for every interface outside that ledger."
        ),
    }


def _maximum_claim_ceiling(roles: Any) -> str:
    ceilings = [ROLE_CLAIM_CEILINGS[str(role)] for role in roles]
    return max(ceilings, key=CLAIM_CEILING_ORDER.__getitem__) if ceilings else "none"


def _claim_ceiling_counts(levels: Any) -> dict[str, int]:
    values = list(levels)
    return {
        level: values.count(level)
        for level in ("none", "M", "D", "S", "N", "G")
    }
