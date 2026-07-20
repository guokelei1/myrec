"""Fail-closed schema for the final Transformer component closeout report."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from myrec.mechanism.deep_dive_closeout_audit import (
    EXPECTED_DELIVERABLES,
    audit_deep_dive_closeout,
)
from myrec.mechanism.deep_dive_evidence_topology import (
    DELIVERABLE_MODEL_COVERAGE,
    MODEL_IDS,
)
from myrec.mechanism.deep_dive_opportunity_catalog import (
    OPPORTUNITY_DESIGN_CATALOG,
    OPPORTUNITY_IDS,
    OPPORTUNITY_STAGE_BOUNDARY,
    PRIOR_WORK_COMPARATORS,
)
from myrec.mechanism.selected_branch_evaluator import (
    SELECTED_BRANCH_FOLD_SCOPE,
)


COMPONENT_IDS = (
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
)
COMPONENT_PROBE_CLAIM_BOUNDARIES = {
    "serialization_tokenization": (
        "Fixed-prompt serialization, token-length, and content controls are covered; "
        "alternative tokenizers or learned serialization policies are not compared."
    ),
    "token_embedding": (
        "Embedding and update geometry are descriptive in this stage; no embedding-layer "
        "causal intervention can establish it as the transfer-failure component."
    ),
    "positional_encoding_rope": (
        "Only layer-local post-RoPE Q/K phase-distance interventions at fixed blocks are "
        "causal; natural position IDs, token identity, and alternative position encoders "
        "are not changed. The +17 common-offset control is an FP32 geometry audit with a "
        "native-Q/K scoring no-op."
    ),
    "attention_query_key_routing": (
        "Causal scope is registered native readout-query to history-key routing; it does "
        "not trace every history-token path or make descriptive head groups primary causes."
    ),
    "attention_value_transport": (
        "Causal scope is the registered history-value contribution into native readout "
        "queries at fixed blocks, not all-token value transport or exclusive head origin."
    ),
    "attention_output": (
        "Selected-position o_proj increment patches test null-context sufficiency, not "
        "necessity, exclusive origin, or a universal attention bottleneck. Architecture "
        "priority additionally requires the separately registered position-preserving "
        "reverse-removal and structural-control synthesis."
    ),
    "mlp_feature_formation": (
        "D4 localizes fixed SwiGLU-product groups at the down_proj input descriptively; "
        "gate_proj, up_proj, and the SiLU/product operation are not separately intervened."
    ),
    "mlp_output": (
        "Selected-position down_proj increment patches test output-state sufficiency; D4 "
        "groups remain descriptive and do not identify a unique forming neuron or branch. "
        "Architecture priority additionally requires the separately registered "
        "position-preserving reverse-removal and structural-control synthesis."
    ),
    "residual_composition": (
        "Seven-node patches are null-context sufficiency tests, not additive or Shapley "
        "decomposition; incoming block-state support blocks current-block attribution, "
        "and residual design priority requires the separate reverse-removal overlay."
    ),
    "normalization": (
        "RMS/direction controls and selected-position norm-boundary state patches are "
        "covered. A primary normalization label additionally requires support to appear "
        "at a post-norm state when its paired pre-norm state does not pass the same "
        "registered gate; this is boundary-localized sufficiency, not evidence of "
        "RMSNorm operator necessity or all-token normalization."
    ),
    "layerwise_representation": (
        "The scan observes accumulated native candidate-scoring states; an exact block "
        "index is localization metadata, not architecture evidence or history-token flow."
    ),
    "history_routing": (
        "Coverage is limited to frozen history spans, registered readout edges, and fixed "
        "context controls; it does not enumerate every token-to-token history route."
    ),
    "candidate_conditioned_interaction": (
        "Evidence is scoped to frozen candidates and native candidate scoring positions; "
        "it does not establish a generic user representation or open-world interaction."
    ),
    "native_readout": (
        "Coverage is the frozen Q2 yes/no margin and Q3 teacher-forced native paths; "
        "Q3 term substitution is exact only for that teacher-forced score and does not "
        "test autoregressive generated-token feedback. Alternative learned readout "
        "heads are outside this diagnosis."
    ),
    "score_calibration_nullspace": (
        "Exact common-plus-relative score algebra and registered readout controls are "
        "covered; a rank-null common shift is not itself a utility mechanism."
    ),
    "loss_gradient": (
        "Evidence is confined to frozen train-visible objectives, surfaces, and replay "
        "states; it does not establish a dataset-independent optimization law."
    ),
    "optimizer_effective_update": (
        "Effective-update and optimizer-replay evidence is diagnostic/descriptive only; "
        "no diagnostic training control is promoted as a causal paper method."
    ),
    "lora_parameterization": (
        "Q3 LoRA path, gauge, merge, and geometry audits are descriptive; no causal LoRA "
        "variant establishes parameterization as the transfer-failure source."
    ),
}
if set(COMPONENT_PROBE_CLAIM_BOUNDARIES) != set(COMPONENT_IDS):
    raise AssertionError("component probe claim boundaries must cover all 18 components")
COMPONENT_STATUSES = {
    "supported",
    "weakened",
    "unresolved",
    "untested",
    "mechanical_failure",
}
HYPOTHESIS_IDS = tuple(f"H{index}" for index in range(6))
HYPOTHESIS_STATUSES = {"supported", "weakened", "rejected", "unresolved"}
NEGATIVE_EVIDENCE_BASES = {
    "not_applicable",
    "registered_practical_equivalence",
    "registered_significant_opposite_direction",
    "registered_independent_counterexample",
    "mixed_registered_evidence",
}
REJECTION_EVIDENCE_BASES = {
    "registered_practical_equivalence",
    "registered_significant_opposite_direction",
    "registered_independent_counterexample",
}
PRIMARY_ATTRIBUTION_MODELS = (
    "q2_recranker_generalqwen",
    "q3_tallrec_generalqwen",
)
PRIMARY_LOSS_COMPONENTS = {
    "attention_output",
    "mlp",
    "mixed_attention_mlp",
    "residual_composition",
    "residual_norm_interaction",
    "unresolved",
}
PRIMARY_ATTRIBUTION_DERIVATION_PRECEDENCE = (
    "fold1_transition_required",
    "mixed_attention_mlp",
    "attention_output",
    "mlp",
    "residual_norm_interaction",
    "residual_composition",
    "unresolved",
)
PRIMARY_ATTRIBUTION_CRITERION_DESCRIPTIONS = {
    "mixed_attention_mlp": (
        "fold1 transition reproduced and both attention_o_projection and "
        "mlp_down_projection pass all six fold1-only confirmatory target-margin "
        "gates"
    ),
    "attention_output": (
        "fold1 transition reproduced; attention_o_projection passes all six "
        "fold1-only confirmatory gates and mlp_down_projection does not"
    ),
    "mlp": (
        "fold1 transition reproduced; mlp_down_projection passes all six "
        "fold1-only confirmatory gates and attention_o_projection does not"
    ),
    "residual_norm_interaction": (
        "fold1 transition reproduced; attention and MLP are insufficient; at "
        "least one registered post-norm node passes all six gates while its paired "
        "pre-norm state does not; and the incoming block-input state does not itself "
        "pass all six gates. This localizes a norm boundary but does not establish "
        "RMSNorm operator necessity"
    ),
    "residual_composition": (
        "fold1 transition reproduced; attention, MLP, and normalization are "
        "insufficient; a post-attention or block-output residual node passes "
        "all six gates; and the incoming block-input state does not itself pass "
        "all six gates"
    ),
    "unresolved": (
        "fold1 transition is not reproduced or no registered branch/composition "
        "criterion establishes a unique component; incoming-state sufficiency "
        "keeps residual/norm attribution unresolved rather than assigning an "
        "upstream-carried state to the selected block"
    ),
}
PRIMARY_ATTRIBUTION_STRENGTH_RULES = {
    "registered_confirmatory": (
        "the fold0-selected transition reproduces on fold1 and a unique primary "
        "component is mechanically derived from fold1-only seven-node gates; "
        "this is split-sample localization, not two-fold node-effect replication"
    ),
    "exploratory_only": (
        "a fold0 transition was selected but the fixed fold1 transition did not "
        "reproduce; selected-branch evidence remains exploratory"
    ),
    "gate_stopped": (
        "the model was not run after its scientific/mechanical gate or fold0 had "
        "no negative adjacent transition"
    ),
    "unresolved": (
        "the fold1 transition reproduced but no unique registered component "
        "criterion was established, or localization metadata is insufficient"
    ),
}
ATTRIBUTION_EVIDENCE_STRENGTHS = {
    "registered_confirmatory",
    "exploratory_only",
    "gate_stopped",
    "unresolved",
}
PRIMARY_ATTRIBUTION_FOLD_SCOPE = dict(SELECTED_BRANCH_FOLD_SCOPE)
PRIMARY_ATTRIBUTION_LAYER_SELECTION_SCOPE = {
    "selector": "argmin_k_in_14_to_27(E_k-E_k_minus_1); tie_lower_k",
    "selected_transition_interpretation": (
        "largest_fold0_mean_negative_adjacent_postblock_step"
    ),
    "fold1_role": "fixed_selected_transition_reproduction",
    "earliest_loss_layer_established": False,
    "global_unique_loss_layer_established": False,
    "layer_scan_role": "unbiased_localization_for_component_decomposition",
    "layer_scan_observed_state_scope": "native_candidate_scoring_positions_only",
    "history_effect_interpretation": "accumulated_state_sufficiency_not_token_path",
    "history_token_flow_directly_observed_by_layer_scan": False,
    "exact_layer_index_is_architecture_evidence": False,
    "cross_model_exact_layer_generalization_authorized": False,
    "design_implication_requires_component_or_distributed_pattern_evidence": True,
    "claim_boundary": (
        "The selected block is the largest registered fold-0 mean negative "
        "adjacent post-block step, confirmed at that fixed transition on fold 1; "
        "it is not an estimate of the earliest loss onset or a globally unique "
        "loss layer. The scan intervenes only at native candidate-scoring states, "
        "so it tests accumulated-state sufficiency rather than directly observing "
        "history-token flow. Its exact index is localization metadata, not "
        "architecture evidence and not a cross-model invariant; design implications "
        "require component-level evidence or a registered distributed attenuation "
        "pattern."
    ),
}
PRIMARY_ATTRIBUTION_ENDPOINT_SCOPE = {
    "primary_endpoint": "target_margin",
    "secondary_utility_endpoint": "ndcg@10",
    "ndcg_practical_equivalence_band": [-0.005, 0.005],
    "utility_relevant_negative_ndcg_requires_ci95_upper_below": -0.005,
    "target_margin_component_is_not_automatically_ndcg_cause": True,
}
PRIMARY_ATTRIBUTION_INTERVENTION_SCOPE = {
    "causal_intervention_role": "null_context_sufficiency",
    "primary_component_interpretation": (
        "registered_candidate_bottleneck_not_unique_origin"
    ),
    "within_block_adjacent_change_role": (
        "descriptive_only_without_registered_directional_gate"
    ),
    "component_erasure_boundary_established": False,
    "necessity_tested": False,
    "reverse_necessity_extension_is_separate_overlay": True,
    "primary_attribution_alone_authorizes_architecture_priority": False,
    "exclusive_component_origin_established": False,
    "additive_or_shapley_contribution_estimated": False,
    "claim_boundary": (
        "Selected-node patches test whether a full-context node state is "
        "sufficient to reproduce harm in a null-context recipient; they do not "
        "establish a directional within-block erasure boundary, necessity, exclusive "
        "causal origin, or additive/Shapley component contribution. Adjacent-node "
        "contrasts remain descriptive because no directional gate was registered. "
        "Architecture priority must be decided by the separate position-preserving "
        "reverse-removal and structural-control overlay."
    ),
}
TRANSFER_FAILURE_CAUSAL_SCOPES = {
    "target_margin_primary_with_utility_relevant_ndcg_corroboration",
    "target_margin_primary_with_statistical_ndcg_corroboration_only",
    "target_margin_only",
    "unresolved",
}
TRANSFER_EXPLANATION_LEVELS = (
    "unresolved_or_gate_stopped",
    "reproduced_layer_transition_without_unique_component",
    "target_margin_component_sufficiency",
    "target_margin_component_with_statistical_ndcg_corroboration",
    "target_margin_component_with_utility_relevant_ndcg_corroboration",
)
TRANSFER_EXPLANATION_LADDER_SCOPE = {
    "ordered_levels": list(TRANSFER_EXPLANATION_LEVELS),
    "highest_level_establishes_necessity": False,
    "highest_level_establishes_exclusive_causal_origin": False,
    "highest_level_establishes_complete_transfer_failure_cause": False,
    "claim_boundary": (
        "The ladder orders the strongest registered explanation supported for "
        "each model, from unresolved through a utility-relevant NDCG-corroborated "
        "target-margin component sufficiency result. Even its highest level is a "
        "candidate-bottleneck sufficiency diagnosis, not necessity, exclusive "
        "causal origin, or a complete explanation of transfer failure."
    ),
}
CROSS_MODEL_ATTRIBUTION_SCOPES = (
    "no_registered_component_resolution",
    "single_model_registered_component_only",
    "model_heterogeneous_registered_components",
    "shared_registered_component_sufficiency_across_q2_q3",
)
CROSS_MODEL_ATTRIBUTION_BOUNDARY = {
    "observed_models": list(PRIMARY_ATTRIBUTION_MODELS),
    "generalization_beyond_q2_q3_authorized": False,
    "universal_llm4rec_mechanism_claim_authorized": False,
    "claim_boundary": (
        "Cross-model attribution compares only the frozen Q2 and Q3 systems. "
        "A shared component means registered sufficiency in both observed "
        "systems; it does not establish a universal LLM4Rec mechanism, and a "
        "single-model result cannot be projected onto the unresolved model."
    ),
}
PRIMARY_ATTRIBUTION_ALLOWED_DELIVERABLES = {
    "d2_postblock",
    "d2_selected_branches",
    "d3_attention_edges",
    "d3_attention_heads",
    "d3_attention_groups",
    "d4_mlp_groups",
    "d6_q2_native_readout",
    "d6_q3_native_readout",
}
OPPORTUNITY_STATUSES = {"primary", "secondary", "deprioritized", "rejected"}
REQUIRED_ASSERTIONS = {
    "source_test_opened": False,
    "dataset_switched": False,
    "transfer_architecture_implemented": False,
    "frozen_first_round_overwritten": False,
    "outcome_selected_layer_head_group_seed_surface_endpoint": False,
    "exact_layer_index_used_as_architecture_design_parameter": False,
    "layer_scan_alone_used_to_rank_architecture_opportunity": False,
    "selected_branch_sufficiency_alone_used_to_rank_architecture_opportunity": False,
    "layer_shape_generalized_beyond_frozen_models_or_dataset": False,
    "p_gt_0p05_or_missing_support_used_as_weakened_or_rejected_evidence": False,
    "all_19_deliverables_admitted": True,
    "all_valid_mechanical_failures_retained": True,
}
REQUIRED_NARRATIVE_FIELDS = (
    "executive_summary",
    "primary_mechanism_diagnosis",
    "signal_attenuation_answer",
    "cross_model_boundary",
    "negative_evidence_summary",
    "remaining_uncertainty",
    "recommended_next_action",
)
COMPONENT_ALLOWED_DELIVERABLES = {
    "serialization_tokenization": {
        "d1_representation",
        "d5_context",
        "d6_q0_trajectory",
        "d6_q1_trajectory",
    },
    "token_embedding": {
        "d1_representation",
        "d5_context",
        "d6_q2_native_readout",
        "d6_q3_native_readout",
        "d6_q0_trajectory",
        "d6_q1_trajectory",
        "d7_optimizer_replay",
    },
    "positional_encoding_rope": {
        "d1_representation",
        "d3_attention_heads",
        "d5_rope",
        "d6_q0_trajectory",
        "d6_q1_trajectory",
    },
    "attention_query_key_routing": {
        "d3_attention_edges",
        "d3_attention_heads",
        "d3_attention_groups",
        "d5_rope",
        "d7_optimizer_replay",
    },
    "attention_value_transport": {
        "d3_attention_edges",
        "d3_attention_heads",
        "d3_attention_groups",
        "d7_optimizer_replay",
    },
    "attention_output": {
        "d2_selected_branches",
        "d3_attention_edges",
        "d3_attention_heads",
        "d3_attention_groups",
        "d6_q0_q1_branches",
        "d7_optimizer_replay",
    },
    "mlp_feature_formation": {
        "d4_mlp_groups",
        "d7_optimizer_replay",
    },
    "mlp_output": {
        "d2_selected_branches",
        "d4_mlp_groups",
        "d6_q0_q1_branches",
        "d7_optimizer_replay",
    },
    "residual_composition": {
        "d2_postblock",
        "d2_selected_branches",
        "d4_mlp_groups",
        "d6_q0_q1_branches",
    },
    "normalization": {
        "d2_postblock",
        "d2_selected_branches",
        "d6_q2_native_readout",
        "d6_q3_native_readout",
        "d6_q0_q1_readouts",
    },
    "layerwise_representation": {
        "d1_representation",
        "d2_postblock",
        "d6_q0_trajectory",
        "d6_q1_trajectory",
    },
    "history_routing": {
        "d2_selected_branches",
        "d3_attention_edges",
        "d3_attention_heads",
        "d3_attention_groups",
        "d5_context",
    },
    "candidate_conditioned_interaction": {
        "d2_postblock",
        "d2_selected_branches",
        "d3_attention_edges",
        "d3_attention_groups",
        "d4_mlp_groups",
        "d5_context",
        "d5_rope",
    },
    "native_readout": {
        "d2_q3_native_gate",
        "d6_q2_native_readout",
        "d6_q3_native_readout",
        "d6_q0_q1_readouts",
    },
    "score_calibration_nullspace": {
        "d6_q2_native_readout",
        "d6_q3_native_readout",
        "d6_q0_q1_readouts",
        "d7_q2_objective",
    },
    "loss_gradient": {
        "d7_q2_objective",
        "d7_q3_lora_path",
        "d7_optimizer_replay",
    },
    "optimizer_effective_update": {
        "d7_q2_objective",
        "d7_q3_lora_path",
        "d7_optimizer_replay",
    },
    "lora_parameterization": {
        "d7_q3_lora_path",
        "d7_optimizer_replay",
    },
}
COMPONENT_SUPPORT_REQUIRED_CAUSAL_EVIDENCE = {
    "serialization_tokenization": {"d5_context"},
    # Token embedding/readout-row and optimizer probes are registered but do
    # not isolate a token-embedding intervention, so this component cannot be
    # upgraded to supported by borrowing RoPE causality.
    "token_embedding": set(),
    "positional_encoding_rope": {"d5_rope"},
    "attention_query_key_routing": {"d3_attention_edges", "d5_rope"},
    "attention_value_transport": {"d3_attention_edges"},
    "attention_output": {
        "d2_selected_branches",
        "d6_q0_q1_branches",
    },
    # D4 localizes SwiGLU groups but is exploratory by registration; it cannot
    # on its own promote gate/up feature formation to a causal mechanism.
    "mlp_feature_formation": set(),
    "mlp_output": {"d2_selected_branches", "d6_q0_q1_branches"},
    "residual_composition": {"d2_selected_branches"},
    "normalization": {
        "d2_selected_branches",
    },
    "layerwise_representation": {"d2_postblock"},
    "history_routing": {
        "d2_selected_branches",
        "d3_attention_edges",
        "d5_context",
    },
    "candidate_conditioned_interaction": {
        "d2_postblock",
        "d2_selected_branches",
        "d3_attention_edges",
        "d5_context",
        "d5_rope",
    },
    "native_readout": {
        "d6_q2_native_readout",
        "d6_q3_native_readout",
    },
    "score_calibration_nullspace": {
        "d6_q2_native_readout",
        "d6_q3_native_readout",
    },
    "loss_gradient": {"d7_q2_objective"},
    # The registered effective-update evidence is exact but descriptive;
    # the frozen plan forbids upgrading it to supported without a SESOI.
    "optimizer_effective_update": set(),
    # Gauge/merge identities and one-step LoRA geometry are diagnostic, not an
    # independent causal performance test of the parameterization.
    "lora_parameterization": set(),
}
COMPONENT_SUPPORT_MECHANICAL_DEPENDENCIES = {
    "attention_query_key_routing": {
        "d3_attention_edges": {"d3_attention_heads"},
    },
    "attention_value_transport": {
        "d3_attention_edges": {"d3_attention_heads"},
    },
    "history_routing": {
        "d3_attention_edges": {"d3_attention_heads"},
    },
}

# Deliverable-level coverage is insufficient for D7: Q2 replays full parameter
# families, whereas Q3 replays only q/v LoRA factors.  Keep component-specific
# model scope outcome-independent so one model cannot borrow another model's
# parameter-family evidence from the same aggregate deliverable.
COMPONENT_DELIVERABLE_MODEL_COVERAGE = {
    component_id: {
        deliverable: set(DELIVERABLE_MODEL_COVERAGE[deliverable])
        for deliverable in deliverables
    }
    for component_id, deliverables in COMPONENT_ALLOWED_DELIVERABLES.items()
}
for _component_id in (
    "token_embedding",
    "attention_output",
    "mlp_feature_formation",
    "mlp_output",
):
    COMPONENT_DELIVERABLE_MODEL_COVERAGE[_component_id][
        "d7_optimizer_replay"
    ] = {MODEL_IDS[2]}
COMPONENT_DELIVERABLE_MODEL_COVERAGE["lora_parameterization"][
    "d7_optimizer_replay"
] = {MODEL_IDS[3]}
HYPOTHESIS_ALLOWED_DELIVERABLES = {
    "H0": {
        "d1_representation",
        "d5_context",
        "d6_q0_trajectory",
        "d6_q1_trajectory",
        "d6_q2_native_readout",
        "d6_q3_native_readout",
    },
    "H1": {
        "d2_selected_branches",
        "d3_attention_edges",
        "d3_attention_heads",
        "d3_attention_groups",
        "d5_context",
    },
    "H2": {
        "d1_representation",
        "d5_context",
        "d6_q0_trajectory",
        "d6_q1_trajectory",
        "d6_q0_q1_branches",
        "d6_q0_q1_readouts",
    },
    "H3": {
        "d2_postblock",
        "d2_q3_native_gate",
        "d2_selected_branches",
        "d3_attention_edges",
        "d3_attention_groups",
        "d4_mlp_groups",
        "d6_q2_native_readout",
        "d6_q3_native_readout",
        "d6_q0_q1_readouts",
    },
    "H4": {
        "d5_context",
        "d7_q2_objective",
        "d7_q3_lora_path",
        "d7_optimizer_replay",
    },
    "H5": set(EXPECTED_DELIVERABLES),
}
HYPOTHESIS_SUPPORTED_REQUIRED_EVIDENCE_GROUPS = {
    "H0": (
        {"d1_representation"},
        {
            "d6_q0_trajectory",
            "d6_q1_trajectory",
            "d6_q2_native_readout",
            "d6_q3_native_readout",
        },
    ),
    "H1": (
        {"d2_selected_branches", "d3_attention_edges"},
        {"d3_attention_heads", "d3_attention_groups", "d5_context"},
    ),
    "H2": (
        {"d1_representation"},
        {"d6_q0_trajectory", "d6_q1_trajectory", "d6_q0_q1_branches"},
    ),
    "H3": (
        {"d2_postblock", "d2_selected_branches"},
        {"d6_q2_native_readout", "d6_q3_native_readout"},
    ),
    "H4": (
        {"d7_q2_objective"},
        {"d5_context", "d7_q3_lora_path", "d7_optimizer_replay"},
    ),
    # No registered independent second seed exists in this stopping stage.
    "H5": (),
}
HYPOTHESIS_SUPPORTED_COMPONENT_REQUIREMENTS = {
    # A routing-failure claim needs both an isolated Q/K routing effect and a
    # history-specific causal path; either alone is only partial localization.
    "H1": ("attention_query_key_routing", "history_routing"),
    # A shortcut/objective-conflict claim must pass the registered gradient
    # conflict SESOI/FDR gate, not merely cite descriptive update geometry.
    "H4": ("loss_gradient",),
}
OPPORTUNITY_ALLOWED_DELIVERABLES = {
    "OP_H1_QUERY_CONDITIONED_SPARSE_ROUTER": HYPOTHESIS_ALLOWED_DELIVERABLES[
        "H1"
    ],
    "OP_H2_ID_FREE_FACTORIZED_PREFERENCE_BOTTLENECK": HYPOTHESIS_ALLOWED_DELIVERABLES[
        "H2"
    ],
    "OP_H3_CANDIDATE_CONDITIONED_SIGNED_PREFERENCE_RESIDUAL": HYPOTHESIS_ALLOWED_DELIVERABLES[
        "H3"
    ],
    "OP_H2_H3_FACTORIZED_SIGNED_PREFERENCE_PATH": (
        HYPOTHESIS_ALLOWED_DELIVERABLES["H2"]
        | HYPOTHESIS_ALLOWED_DELIVERABLES["H3"]
    ),
    "OP_H4_SURFACE_AWARE_GRADIENT_BUDGET": HYPOTHESIS_ALLOWED_DELIVERABLES[
        "H4"
    ],
}
OPPORTUNITY_PRIMARY_REQUIRED_EVIDENCE_GROUPS = {
    "OP_H1_QUERY_CONDITIONED_SPARSE_ROUTER": (
        {"d2_selected_branches", "d3_attention_edges"},
        {"d3_attention_heads", "d3_attention_groups", "d5_context"},
    ),
    "OP_H2_ID_FREE_FACTORIZED_PREFERENCE_BOTTLENECK": (
        {"d1_representation"},
        {"d6_q0_trajectory", "d6_q1_trajectory", "d6_q0_q1_branches"},
    ),
    "OP_H3_CANDIDATE_CONDITIONED_SIGNED_PREFERENCE_RESIDUAL": (
        {"d2_postblock", "d2_selected_branches"},
        {"d6_q2_native_readout", "d6_q3_native_readout"},
    ),
    "OP_H2_H3_FACTORIZED_SIGNED_PREFERENCE_PATH": (
        {"d1_representation"},
        {"d6_q0_trajectory", "d6_q1_trajectory", "d6_q0_q1_branches"},
        {"d2_postblock", "d2_selected_branches"},
        {"d6_q2_native_readout", "d6_q3_native_readout"},
    ),
    "OP_H4_SURFACE_AWARE_GRADIENT_BUDGET": (
        {"d7_q2_objective"},
        {"d5_context", "d7_q3_lora_path", "d7_optimizer_replay"},
    ),
}
# Some opportunities have model-local evidence groups.  Require every scoped
# model to be covered by every listed group, rather than accepting the union of
# Q2 and Q3 evidence.  H2 remains intentionally cross-model: its second group
# is the fixed Q0/Q1 breadth boundary.
OPPORTUNITY_PRIMARY_PER_MODEL_EVIDENCE_GROUPS = {
    "OP_H1_QUERY_CONDITIONED_SPARSE_ROUTER": (
        {"d2_selected_branches", "d3_attention_edges"},
        {"d3_attention_heads", "d3_attention_groups", "d5_context"},
    ),
    "OP_H3_CANDIDATE_CONDITIONED_SIGNED_PREFERENCE_RESIDUAL": (
        {"d2_postblock", "d2_selected_branches"},
        {"d6_q2_native_readout", "d6_q3_native_readout"},
    ),
    "OP_H2_H3_FACTORIZED_SIGNED_PREFERENCE_PATH": (
        {"d2_postblock", "d2_selected_branches"},
        {"d6_q2_native_readout", "d6_q3_native_readout"},
    ),
    "OP_H4_SURFACE_AWARE_GRADIENT_BUDGET": (
        {"d7_q2_objective"},
        {"d5_context", "d7_q3_lora_path", "d7_optimizer_replay"},
    ),
}
OPPORTUNITY_HYPOTHESES = {
    "OP_H1_QUERY_CONDITIONED_SPARSE_ROUTER": ("H1",),
    "OP_H2_ID_FREE_FACTORIZED_PREFERENCE_BOTTLENECK": ("H2",),
    "OP_H3_CANDIDATE_CONDITIONED_SIGNED_PREFERENCE_RESIDUAL": ("H3",),
    "OP_H2_H3_FACTORIZED_SIGNED_PREFERENCE_PATH": ("H2", "H3"),
    "OP_H4_SURFACE_AWARE_GRADIENT_BUDGET": ("H4",),
}
OPPORTUNITY_ALLOWED_MODEL_SCOPE = {
    "OP_H1_QUERY_CONDITIONED_SPARSE_ROUTER": {MODEL_IDS[2], MODEL_IDS[3]},
    "OP_H2_ID_FREE_FACTORIZED_PREFERENCE_BOTTLENECK": set(MODEL_IDS),
    "OP_H3_CANDIDATE_CONDITIONED_SIGNED_PREFERENCE_RESIDUAL": {
        MODEL_IDS[2],
        MODEL_IDS[3],
    },
    # The signed residual half has confirmatory evidence only on Q2/Q3; Q0/Q1
    # breadth cannot be borrowed to expand the combined opportunity scope.
    "OP_H2_H3_FACTORIZED_SIGNED_PREFERENCE_PATH": {
        MODEL_IDS[2],
        MODEL_IDS[3],
    },
    # Only Q2 has a registered objective-conflict family.  Q3 LoRA/update
    # geometry is descriptive and cannot expand this evidence scope.
    "OP_H4_SURFACE_AWARE_GRADIENT_BUDGET": {MODEL_IDS[2]},
}
PRIMARY_COMPONENT_MATRIX_REQUIREMENTS = {
    "attention_output": ("attention_output",),
    "mlp": ("mlp_output",),
    "mixed_attention_mlp": ("attention_output", "mlp_output"),
    "residual_composition": ("residual_composition",),
    "residual_norm_interaction": ("normalization",),
    "unresolved": (),
}


class DeepDiveReportContractError(ValueError):
    """The final human decisions payload is incomplete or out of scope."""


def validate_deep_dive_report_decisions(
    decisions: Mapping[str, Any],
    *,
    admitted_deliverables: Sequence[str] = tuple(EXPECTED_DELIVERABLES),
    admitted_failure_records: Sequence[str] = (),
) -> None:
    """Require complete component, H0--H5, and opportunity judgements."""

    if not isinstance(decisions, Mapping):
        raise DeepDiveReportContractError("deep-dive decisions must be an object")
    admitted = set(map(str, admitted_deliverables))
    if admitted != set(EXPECTED_DELIVERABLES):
        raise DeepDiveReportContractError("all 19 closeout deliverables must be admitted")
    narratives = decisions.get("narratives")
    if not isinstance(narratives, Mapping):
        raise DeepDiveReportContractError("deep-dive narratives must be an object")
    if set(narratives) != set(REQUIRED_NARRATIVE_FIELDS):
        raise DeepDiveReportContractError(
            "deep-dive narratives must contain the exact required fields"
        )
    for field in REQUIRED_NARRATIVE_FIELDS:
        _require_text(narratives, field, "deep-dive narratives")
    _validate_component_matrix(
        decisions.get("component_evidence_matrix"),
        admitted,
        set(map(str, admitted_failure_records)),
    )
    _validate_primary_loss_attribution(
        decisions.get("primary_loss_attribution"), admitted
    )
    _validate_primary_component_matrix_consistency(
        decisions.get("component_evidence_matrix"),
        decisions.get("primary_loss_attribution"),
    )
    _validate_cross_model_primary_attribution(
        decisions.get("cross_model_primary_attribution"),
        decisions.get("primary_loss_attribution"),
    )
    _validate_hypothesis_matrix(decisions.get("hypothesis_status_matrix"), admitted)
    _validate_hypothesis_component_consistency(
        decisions.get("component_evidence_matrix"),
        decisions.get("hypothesis_status_matrix"),
    )
    _validate_opportunities(decisions.get("architecture_opportunity_ranking"), admitted)
    _validate_opportunity_hypothesis_consistency(
        decisions.get("hypothesis_status_matrix"),
        decisions.get("architecture_opportunity_ranking"),
    )
    assertions = decisions.get("boundary_assertions")
    if assertions != REQUIRED_ASSERTIONS:
        raise DeepDiveReportContractError("deep-dive boundary assertions differ")


def validate_deep_dive_report_against_closeout(
    root: str | Path, decisions: Mapping[str, Any]
) -> dict[str, Any]:
    """Refuse final report decisions until every registered output is admitted."""

    closeout = audit_deep_dive_closeout(root)
    if closeout.get("status") != "completed":
        raise DeepDiveReportContractError(
            "deep-dive closeout must be completed before final report generation"
        )
    deliverables = closeout.get("deliverables", {})
    admitted = [
        name
        for name, identity in deliverables.items()
        if isinstance(identity, Mapping) and identity.get("status") == "completed"
    ]
    failures = [
        str(record.get("path"))
        for record in closeout.get("mechanical_failure_records", [])
        if isinstance(record, Mapping) and record.get("status") == "mechanical_failure"
    ]
    validate_deep_dive_report_decisions(
        decisions,
        admitted_deliverables=admitted,
        admitted_failure_records=failures,
    )
    _validate_primary_attribution_against_evidence(Path(root), decisions)
    _validate_supported_components_against_evidence(Path(root), decisions)
    return closeout


def _validate_component_matrix(
    value: Any, admitted: set[str], admitted_failures: set[str]
) -> None:
    rows = _rows(value, "component evidence matrix")
    _require_exact_ids(rows, "component_id", COMPONENT_IDS, "component evidence matrix")
    cited_failures: set[str] = set()
    for row in rows:
        status = row.get("status")
        if status not in COMPONENT_STATUSES:
            raise DeepDiveReportContractError(
                f"invalid component status: {row.get('component_id')}={status}"
            )
        negative_basis = row.get("negative_evidence_basis")
        if negative_basis not in NEGATIVE_EVIDENCE_BASES:
            raise DeepDiveReportContractError(
                "invalid component negative-evidence basis: "
                f"{row.get('component_id')}={negative_basis}"
            )
        if status == "weakened" and negative_basis == "not_applicable":
            raise DeepDiveReportContractError(
                "weakened component requires explicit registered negative evidence: "
                f"{row.get('component_id')}"
            )
        _require_text(row, "finding", "component evidence matrix")
        _require_text(row, "claim_boundary", "component evidence matrix")
        if row.get("claim_boundary") != COMPONENT_PROBE_CLAIM_BOUNDARIES[
            row["component_id"]
        ]:
            raise DeepDiveReportContractError(
                "component claim boundary differs from registered probe scope: "
                f"{row['component_id']}"
            )
        _require_text(row, "optimization_implication", "component evidence matrix")
        model_scope = row.get("model_scope")
        if (
            not isinstance(model_scope, list)
            or not model_scope
            or len(model_scope) != len(set(map(str, model_scope)))
            or any(str(model_id) not in MODEL_IDS for model_id in model_scope)
        ):
            raise DeepDiveReportContractError(
                f"component model scope is invalid: {row['component_id']}"
            )
        scoped_models = set(map(str, model_scope))
        evidence = _evidence_ids(row, admitted, "component evidence matrix")
        _require_relevant_evidence(
            evidence,
            COMPONENT_ALLOWED_DELIVERABLES[row["component_id"]],
            "component evidence matrix",
        )
        for causal_deliverable, dependencies in (
            COMPONENT_SUPPORT_MECHANICAL_DEPENDENCIES.get(
                row["component_id"], {}
            ).items()
        ):
            if (
                status == "supported"
                and causal_deliverable in evidence
                and not dependencies.issubset(evidence)
            ):
                missing = ",".join(sorted(dependencies - set(evidence)))
                raise DeepDiveReportContractError(
                    "supported component lacks its mechanical dependency: "
                    f"{row['component_id']}:{causal_deliverable}:{missing}"
                )
        if evidence:
            component_coverage = COMPONENT_DELIVERABLE_MODEL_COVERAGE[
                row["component_id"]
            ]
            evidence_models = set().union(
                *(component_coverage[item] for item in evidence)
            )
            if not scoped_models.issubset(evidence_models):
                raise DeepDiveReportContractError(
                    f"component model scope lacks direct evidence: {row['component_id']}"
                )
            if any(
                not (component_coverage[item] & scoped_models)
                for item in evidence
            ):
                raise DeepDiveReportContractError(
                    f"component cites evidence outside its model scope: {row['component_id']}"
                )
        if status == "supported":
            causal = COMPONENT_SUPPORT_REQUIRED_CAUSAL_EVIDENCE[
                row["component_id"]
            ]
            unsupported_models = [
                model_id
                for model_id in scoped_models
                if not any(
                    item in causal
                    and model_id
                    in COMPONENT_DELIVERABLE_MODEL_COVERAGE[row["component_id"]][
                        item
                    ]
                    for item in evidence
                )
            ]
            if unsupported_models:
                raise DeepDiveReportContractError(
                    "supported component lacks per-model causal evidence: "
                    f"{row['component_id']}:{','.join(sorted(unsupported_models))}"
                )
        failures = row.get("mechanical_failure_records", [])
        if not isinstance(failures, list) or any(
            not str(item).strip() for item in failures
        ):
            raise DeepDiveReportContractError(
                "component mechanical failures are invalid"
            )
        if any(str(item) not in admitted_failures for item in failures):
            raise DeepDiveReportContractError(
                "component cites an unadmitted mechanical failure record"
            )
        cited_failures.update(map(str, failures))
        if status not in {"untested", "mechanical_failure"} and not evidence:
            raise DeepDiveReportContractError(
                f"tested component lacks admitted evidence: {row['component_id']}"
            )
        if status == "untested" and evidence:
            raise DeepDiveReportContractError(
                f"untested component cites evidence: {row['component_id']}"
            )
        if status == "mechanical_failure" and not failures:
            raise DeepDiveReportContractError(
                f"mechanical-failure component lacks a bound record: {row['component_id']}"
            )
    missing_failures = admitted_failures - cited_failures
    if missing_failures:
        raise DeepDiveReportContractError(
            "admitted mechanical failure records lack component assignments: "
            + ",".join(sorted(missing_failures))
        )


def _validate_hypothesis_matrix(value: Any, admitted: set[str]) -> None:
    rows = _rows(value, "hypothesis status matrix")
    _require_exact_ids(
        rows, "hypothesis_id", HYPOTHESIS_IDS, "hypothesis status matrix"
    )
    for row in rows:
        if row.get("status") not in HYPOTHESIS_STATUSES:
            raise DeepDiveReportContractError(
                f"invalid hypothesis status: {row.get('hypothesis_id')}"
            )
        negative_basis = row.get("negative_evidence_basis")
        if negative_basis not in NEGATIVE_EVIDENCE_BASES:
            raise DeepDiveReportContractError(
                "invalid hypothesis negative-evidence basis: "
                f"{row.get('hypothesis_id')}={negative_basis}"
            )
        if row.get("status") == "weakened" and negative_basis == "not_applicable":
            raise DeepDiveReportContractError(
                "weakened hypothesis requires explicit registered negative evidence: "
                f"{row.get('hypothesis_id')}"
            )
        if row.get("status") == "rejected" and negative_basis not in (
            REJECTION_EVIDENCE_BASES
        ):
            raise DeepDiveReportContractError(
                "rejected hypothesis requires equivalence, opposite direction, or an "
                f"independent counterexample: {row.get('hypothesis_id')}"
            )
        if row.get("hypothesis_id") == "H5" and row.get("status") in {
            "supported",
            "rejected",
        }:
            raise DeepDiveReportContractError(
                "H5 cannot be supported or rejected without an independent second seed"
            )
        _require_text(row, "rationale", "hypothesis status matrix")
        _require_text(row, "remaining_uncertainty", "hypothesis status matrix")
        evidence = _evidence_ids(row, admitted, "hypothesis status matrix")
        if not evidence:
            raise DeepDiveReportContractError(
                f"hypothesis lacks admitted evidence: {row['hypothesis_id']}"
            )
        _require_relevant_evidence(
            evidence,
            HYPOTHESIS_ALLOWED_DELIVERABLES[row["hypothesis_id"]],
            "hypothesis status matrix",
        )
        if row.get("status") == "supported":
            groups = HYPOTHESIS_SUPPORTED_REQUIRED_EVIDENCE_GROUPS[
                row["hypothesis_id"]
            ]
            if not groups:
                raise DeepDiveReportContractError(
                    f"supported hypothesis lacks an authorized independent confirmation: {row['hypothesis_id']}"
                )
            if any(not (set(evidence) & group) for group in groups):
                raise DeepDiveReportContractError(
                    f"supported hypothesis lacks its required independent evidence groups: {row['hypothesis_id']}"
                )


def _validate_primary_component_matrix_consistency(
    components: Any, attributions: Any
) -> None:
    component_rows = {
        str(row["component_id"]): row
        for row in _rows(components, "component evidence matrix")
    }
    for row in _rows(attributions, "primary loss attribution"):
        required = PRIMARY_COMPONENT_MATRIX_REQUIREMENTS[str(row["primary_component"])]
        missing = [
            component_id
            for component_id in required
            if component_id not in component_rows
            or component_rows[component_id].get("status") != "supported"
            or row["method_id"]
            not in component_rows[component_id].get("model_scope", [])
        ]
        if missing:
            raise DeepDiveReportContractError(
                "primary attribution contradicts component evidence matrix: "
                f"{row['method_id']}:{','.join(missing)}"
            )


def _validate_hypothesis_component_consistency(
    components: Any, hypotheses: Any
) -> None:
    component_rows = {
        str(row["component_id"]): row
        for row in _rows(components, "component evidence matrix")
    }
    for hypothesis in _rows(hypotheses, "hypothesis status matrix"):
        hypothesis_id = str(hypothesis["hypothesis_id"])
        if hypothesis.get("status") != "supported" or hypothesis_id not in (
            HYPOTHESIS_SUPPORTED_COMPONENT_REQUIREMENTS
        ):
            continue
        required = HYPOTHESIS_SUPPORTED_COMPONENT_REQUIREMENTS[hypothesis_id]
        missing = [
            component_id
            for component_id in required
            if component_rows.get(component_id, {}).get("status") != "supported"
        ]
        if missing:
            raise DeepDiveReportContractError(
                "supported hypothesis contradicts component evidence matrix: "
                f"{hypothesis_id}:{','.join(missing)}"
            )
        shared_scope = set(MODEL_IDS)
        for component_id in required:
            shared_scope &= set(
                map(str, component_rows[component_id].get("model_scope", []))
            )
        if not shared_scope:
            raise DeepDiveReportContractError(
                "supported hypothesis components lack a shared model scope: "
                f"{hypothesis_id}"
            )


def _validate_primary_loss_attribution(value: Any, admitted: set[str]) -> None:
    """Require a causal, per-model attention/MLP/residual decision."""

    rows = _rows(value, "primary loss attribution")
    _require_exact_ids(
        rows,
        "method_id",
        PRIMARY_ATTRIBUTION_MODELS,
        "primary loss attribution",
    )
    flag_names = (
        "fold1_transition_reproduced",
        "attention_branch_registered_support",
        "mlp_branch_registered_support",
        "postblock_registered_support",
        "residual_composition_criterion_met",
        "residual_norm_interaction_criterion_met",
        "node_effect_two_fold_replication_tested",
        "split_sample_component_localization",
        "descriptive_localization_used_as_primary_cause",
    )
    for row in rows:
        method_id = str(row["method_id"])
        component = row.get("primary_component")
        strength = row.get("evidence_strength")
        if component not in PRIMARY_LOSS_COMPONENTS:
            raise DeepDiveReportContractError(
                f"invalid primary loss component: {method_id}={component}"
            )
        if strength not in ATTRIBUTION_EVIDENCE_STRENGTHS:
            raise DeepDiveReportContractError(
                f"invalid attribution evidence strength: {method_id}={strength}"
            )
        for flag in flag_names:
            if type(row.get(flag)) is not bool:
                raise DeepDiveReportContractError(
                    f"primary loss attribution flag is not boolean: {method_id}:{flag}"
                )
        if row["descriptive_localization_used_as_primary_cause"] is not False:
            raise DeepDiveReportContractError(
                "descriptive head/group localization cannot establish the primary cause"
            )
        if row.get("selected_branch_node_inference_fold") != 1:
            raise DeepDiveReportContractError(
                "primary attribution node inference must be scoped to fold 1"
            )
        if row["node_effect_two_fold_replication_tested"] is not False:
            raise DeepDiveReportContractError(
                "primary attribution cannot claim two-fold node-effect replication"
            )
        if row["split_sample_component_localization"] is not True:
            raise DeepDiveReportContractError(
                "primary attribution must retain the split-sample scope"
            )
        layer_scope = PRIMARY_ATTRIBUTION_LAYER_SELECTION_SCOPE
        for key in (
            "selected_transition_interpretation",
            "earliest_loss_layer_established",
            "global_unique_loss_layer_established",
            "layer_scan_role",
            "layer_scan_observed_state_scope",
            "history_effect_interpretation",
            "history_token_flow_directly_observed_by_layer_scan",
            "exact_layer_index_is_architecture_evidence",
            "cross_model_exact_layer_generalization_authorized",
            "design_implication_requires_component_or_distributed_pattern_evidence",
        ):
            if row.get(key) != layer_scope[key]:
                raise DeepDiveReportContractError(
                    f"primary attribution layer-selection scope differs: {method_id}:{key}"
                )
        if row.get("primary_attribution_endpoint") != "target_margin":
            raise DeepDiveReportContractError(
                "primary attribution endpoint must remain target_margin"
            )
        if type(row.get("strict_transfer_ndcg_component_corroborated")) is not bool:
            raise DeepDiveReportContractError(
                "strict-transfer NDCG corroboration flag is not boolean"
            )
        if type(
            row.get(
                "strict_transfer_ndcg_beyond_equivalence_component_corroborated"
            )
        ) is not bool:
            raise DeepDiveReportContractError(
                "strict-transfer utility-relevant NDCG corroboration flag is not boolean"
            )
        if row.get("target_margin_component_is_not_automatically_ndcg_cause") is not True:
            raise DeepDiveReportContractError(
                "target-margin attribution cannot be promoted automatically to NDCG cause"
            )
        intervention_scope = PRIMARY_ATTRIBUTION_INTERVENTION_SCOPE
        for key in (
            "causal_intervention_role",
            "primary_component_interpretation",
            "within_block_adjacent_change_role",
            "component_erasure_boundary_established",
            "necessity_tested",
            "exclusive_component_origin_established",
            "additive_or_shapley_contribution_estimated",
        ):
            if row.get(key) != intervention_scope[key]:
                raise DeepDiveReportContractError(
                    f"primary attribution intervention scope differs: {method_id}:{key}"
                )
        transfer_scope = row.get("transfer_failure_causal_scope")
        if transfer_scope not in TRANSFER_FAILURE_CAUSAL_SCOPES:
            raise DeepDiveReportContractError(
                f"invalid transfer-failure causal scope: {method_id}={transfer_scope}"
            )
        transfer_explanation_level = row.get("transfer_explanation_level")
        if transfer_explanation_level not in TRANSFER_EXPLANATION_LEVELS:
            raise DeepDiveReportContractError(
                "invalid transfer explanation level: "
                f"{method_id}={transfer_explanation_level}"
            )
        _require_text(row, "rationale", "primary loss attribution")
        _require_text(row, "claim_boundary", "primary loss attribution")
        if row.get("claim_boundary") != intervention_scope["claim_boundary"]:
            raise DeepDiveReportContractError(
                "primary attribution claim boundary differs from registered intervention scope: "
                f"{method_id}"
            )
        evidence = _evidence_ids(row, admitted, "primary loss attribution")
        _require_relevant_evidence(
            evidence,
            PRIMARY_ATTRIBUTION_ALLOWED_DELIVERABLES,
            "primary loss attribution",
        )
        decisive = {"d2_postblock", "d2_selected_branches"}
        if not decisive.issubset(evidence):
            raise DeepDiveReportContractError(
                "primary loss attribution requires both D2 post-block and selected-branch evidence"
            )

        fold1 = row["fold1_transition_reproduced"]
        attention = row["attention_branch_registered_support"]
        mlp = row["mlp_branch_registered_support"]
        postblock = row["postblock_registered_support"]
        composition = row["residual_composition_criterion_met"]
        residual_norm = row["residual_norm_interaction_criterion_met"]
        if component != "unresolved" and (
            strength != "registered_confirmatory" or not fold1
        ):
            raise DeepDiveReportContractError(
                "resolved primary attribution requires reproduced fold-1 confirmatory evidence"
            )
        expected_flags = {
            "attention_output": (
                postblock and attention and not mlp and not composition and not residual_norm
            ),
            "mlp": (
                postblock and mlp and not attention and not composition and not residual_norm
            ),
            "mixed_attention_mlp": (
                postblock and attention and mlp and not composition and not residual_norm
            ),
            "residual_composition": (
                postblock and not attention and not mlp and composition and not residual_norm
            ),
            "residual_norm_interaction": (
                postblock and not attention and not mlp and residual_norm and not composition
            ),
        }
        if component in expected_flags and not expected_flags[component]:
            raise DeepDiveReportContractError(
                f"primary loss component conflicts with registered branch flags: {method_id}"
            )
        if component == "unresolved" and strength == "registered_confirmatory":
            raise DeepDiveReportContractError(
                "unresolved attribution cannot claim registered-confirmatory resolution"
            )
        if component == "unresolved" and any(
            (attention, mlp, composition, residual_norm)
        ):
            raise DeepDiveReportContractError(
                "unresolved attribution conflicts with a satisfied component criterion"
            )
        if strength == "gate_stopped" and fold1:
            raise DeepDiveReportContractError(
                "gate-stopped attribution cannot claim a reproduced fold-1 transition"
            )
        if composition and residual_norm:
            raise DeepDiveReportContractError(
                "residual composition and residual/norm criteria are mutually exclusive"
            )
        corroborated = row["strict_transfer_ndcg_component_corroborated"]
        utility_corroborated = row[
            "strict_transfer_ndcg_beyond_equivalence_component_corroborated"
        ]
        if utility_corroborated and not corroborated:
            raise DeepDiveReportContractError(
                "utility-relevant NDCG corroboration requires statistical corroboration"
            )
        expected_transfer_scope = (
            "unresolved"
            if component == "unresolved"
            else (
                "target_margin_primary_with_utility_relevant_ndcg_corroboration"
                if utility_corroborated
                else (
                    "target_margin_primary_with_statistical_ndcg_corroboration_only"
                    if corroborated
                    else "target_margin_only"
                )
            )
        )
        if transfer_scope != expected_transfer_scope:
            raise DeepDiveReportContractError(
                f"transfer-failure causal scope conflicts with endpoint evidence: {method_id}"
            )
        expected_explanation_level = _derive_transfer_explanation_level(
            component=component,
            fold1_transition_reproduced=fold1,
            ndcg_corroborated=corroborated,
            utility_ndcg_corroborated=utility_corroborated,
        )
        if transfer_explanation_level != expected_explanation_level:
            raise DeepDiveReportContractError(
                "transfer explanation level conflicts with registered evidence: "
                f"{method_id}"
            )
        if component == "unresolved" and (corroborated or utility_corroborated):
            raise DeepDiveReportContractError(
                "unresolved primary attribution cannot claim NDCG component corroboration"
            )


def _validate_primary_attribution_against_evidence(
    root: Path, decisions: Mapping[str, Any]
) -> None:
    """Cross-check causal attribution flags against the admitted D2 metrics."""

    postblock = _read_report_evidence(
        root / EXPECTED_DELIVERABLES["d2_postblock"],
        "transformer_deep_dive_d2_postblock_synthesis",
    )
    selected = _read_report_evidence(
        root / EXPECTED_DELIVERABLES["d2_selected_branches"],
        "transformer_deep_dive_d2_selected_branch_synthesis",
    )
    evidence_rows = {
        row["method_id"]: row
        for row in derive_primary_attribution_evidence(postblock, selected)
    }
    decisions_by_model = {
        str(row["method_id"]): row for row in decisions["primary_loss_attribution"]
    }
    for method_id in PRIMARY_ATTRIBUTION_MODELS:
        row = decisions_by_model[method_id]
        evidence = evidence_rows[method_id]
        direct_flags = {
            key: evidence[key]
            for key in (
                "fold1_transition_reproduced",
                "postblock_registered_support",
                "attention_branch_registered_support",
                "mlp_branch_registered_support",
                "residual_composition_criterion_met",
                "residual_norm_interaction_criterion_met",
                "selected_branch_node_inference_fold",
                "node_effect_two_fold_replication_tested",
                "split_sample_component_localization",
                "selected_transition_interpretation",
                "earliest_loss_layer_established",
                "global_unique_loss_layer_established",
                "layer_scan_role",
                "layer_scan_observed_state_scope",
                "history_effect_interpretation",
                "history_token_flow_directly_observed_by_layer_scan",
                "exact_layer_index_is_architecture_evidence",
                "cross_model_exact_layer_generalization_authorized",
                "design_implication_requires_component_or_distributed_pattern_evidence",
                "primary_attribution_endpoint",
                "strict_transfer_ndcg_component_corroborated",
                "strict_transfer_ndcg_beyond_equivalence_component_corroborated",
                "target_margin_component_is_not_automatically_ndcg_cause",
                "transfer_failure_causal_scope",
                "transfer_explanation_level",
                "causal_intervention_role",
                "primary_component_interpretation",
                "within_block_adjacent_change_role",
                "component_erasure_boundary_established",
                "necessity_tested",
                "exclusive_component_origin_established",
                "additive_or_shapley_contribution_estimated",
            )
        }
        for key, observed in direct_flags.items():
            if row[key] != observed:
                raise DeepDiveReportContractError(
                    f"primary attribution flag differs from D2 evidence: {method_id}:{key}"
                )
        if row["evidence_strength"] != evidence["derived_evidence_strength"]:
            raise DeepDiveReportContractError(
                "primary attribution strength differs from deterministic D2 evidence: "
                f"{method_id}"
            )
        if row["primary_component"] != evidence["derived_primary_component"]:
            raise DeepDiveReportContractError(
                "primary attribution label differs from deterministic D2 evidence: "
                f"{method_id}"
            )


def _validate_cross_model_primary_attribution(
    value: Any, primary_rows: Any
) -> None:
    """Require model heterogeneity to remain explicit and evidence-derived."""

    if not isinstance(value, Mapping):
        raise DeepDiveReportContractError(
            "cross-model primary attribution must be an object"
        )
    rows = _rows(primary_rows, "primary loss attribution")
    expected = derive_cross_model_primary_attribution(rows)
    for key in (
        "scope",
        "q2_primary_component",
        "q3_primary_component",
        "shared_primary_component",
        "both_models_resolved",
        "same_component_across_models",
        "generalization_beyond_q2_q3_authorized",
        "universal_llm4rec_mechanism_claim_authorized",
    ):
        if value.get(key) != expected[key]:
            raise DeepDiveReportContractError(
                f"cross-model primary attribution differs: {key}"
            )
    if value.get("claim_boundary") != CROSS_MODEL_ATTRIBUTION_BOUNDARY[
        "claim_boundary"
    ]:
        raise DeepDiveReportContractError(
            "cross-model primary attribution claim boundary differs"
        )
    _require_text(value, "rationale", "cross-model primary attribution")


def derive_cross_model_primary_attribution(
    primary_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Derive the strongest cross-model scope without universalizing Q2/Q3."""

    by_model = {str(row.get("method_id")): row for row in primary_rows}
    if set(by_model) != set(PRIMARY_ATTRIBUTION_MODELS):
        raise DeepDiveReportContractError(
            "cross-model attribution requires exact Q2/Q3 primary rows"
        )
    q2_component = str(
        by_model["q2_recranker_generalqwen"].get("primary_component")
        or by_model["q2_recranker_generalqwen"].get(
            "derived_primary_component"
        )
        or "unresolved"
    )
    q3_component = str(
        by_model["q3_tallrec_generalqwen"].get("primary_component")
        or by_model["q3_tallrec_generalqwen"].get(
            "derived_primary_component"
        )
        or "unresolved"
    )
    resolved = [
        component
        for component in (q2_component, q3_component)
        if component != "unresolved"
    ]
    both_resolved = len(resolved) == 2
    same_component = both_resolved and q2_component == q3_component
    if not resolved:
        scope = "no_registered_component_resolution"
    elif len(resolved) == 1:
        scope = "single_model_registered_component_only"
    elif same_component:
        scope = "shared_registered_component_sufficiency_across_q2_q3"
    else:
        scope = "model_heterogeneous_registered_components"
    return {
        "scope": scope,
        "q2_primary_component": q2_component,
        "q3_primary_component": q3_component,
        "shared_primary_component": q2_component if same_component else None,
        "both_models_resolved": both_resolved,
        "same_component_across_models": same_component,
        "generalization_beyond_q2_q3_authorized": False,
        "universal_llm4rec_mechanism_claim_authorized": False,
        "claim_boundary": CROSS_MODEL_ATTRIBUTION_BOUNDARY["claim_boundary"],
    }


def derive_primary_attribution_evidence(
    postblock: Mapping[str, Any], selected: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Derive the unique attention/MLP/residual label from registered D2 gates."""

    localization = postblock.get("localization")
    selected_rows = selected.get("rows")
    if not isinstance(localization, Mapping) or not isinstance(selected_rows, list):
        raise DeepDiveReportContractError("D2 attribution evidence schema differs")
    if selected.get("fold_scope") != PRIMARY_ATTRIBUTION_FOLD_SCOPE:
        raise DeepDiveReportContractError(
            "D2 selected-branch fold-scope boundary differs"
        )
    rows = []
    for method_id in PRIMARY_ATTRIBUTION_MODELS:
        model_localization = localization.get(method_id)
        if not isinstance(model_localization, Mapping):
            raise DeepDiveReportContractError(
                f"D2 localization is missing for primary attribution: {method_id}"
            )
        transition = model_localization.get("resolved") is True
        localization_status = str(model_localization.get("status") or "")
        residual_nodes = ("post_attention_residual", "block_output_residual")
        norm_nodes = ("input_rmsnorm_output", "post_attention_rmsnorm_output")
        primary_nodes = (
            "block_input_residual",
            "attention_o_projection",
            "mlp_down_projection",
            *residual_nodes,
            *norm_nodes,
        )
        support_by_endpoint = {
            endpoint: {
                node: _registered_node_support(
                    selected_rows, method_id, node, endpoint=endpoint
                )
                for node in primary_nodes
            }
            for endpoint in ("target_margin", "ndcg@10")
        }
        margin_support = support_by_endpoint["target_margin"]
        ndcg_support = support_by_endpoint["ndcg@10"]
        ndcg_utility_support = {
            node: _registered_node_support(
                selected_rows,
                method_id,
                node,
                endpoint="ndcg@10",
                require_ndcg_beyond_equivalence=True,
            )
            for node in primary_nodes
        }
        attention_support = margin_support["attention_o_projection"]
        mlp_support = margin_support["mlp_down_projection"]
        incoming_state_support = margin_support["block_input_residual"]
        residual_margin_nodes = {
            node for node in residual_nodes if margin_support[node]
        }
        residual_ndcg_nodes = {
            node for node in residual_nodes if ndcg_support[node]
        }
        residual_ndcg_utility_nodes = {
            node for node in residual_nodes if ndcg_utility_support[node]
        }
        norm_margin_nodes = {node for node in norm_nodes if margin_support[node]}
        norm_ndcg_nodes = {node for node in norm_nodes if ndcg_support[node]}
        norm_ndcg_utility_nodes = {
            node for node in norm_nodes if ndcg_utility_support[node]
        }
        norm_predecessors = {
            "input_rmsnorm_output": "block_input_residual",
            "post_attention_rmsnorm_output": "post_attention_residual",
        }
        isolated_norm_margin_nodes = {
            node
            for node in norm_nodes
            if margin_support[node]
            and not margin_support[norm_predecessors[node]]
        }
        unisolated_norm_margin_nodes = norm_margin_nodes - isolated_norm_margin_nodes
        isolated_norm_ndcg_nodes = {
            node
            for node in norm_nodes
            if ndcg_support[node]
            and not ndcg_support[norm_predecessors[node]]
        }
        isolated_norm_ndcg_utility_nodes = {
            node
            for node in norm_nodes
            if ndcg_utility_support[node]
            and not ndcg_support[norm_predecessors[node]]
        }
        residual_node_support = bool(residual_margin_nodes)
        norm_node_support = bool(norm_margin_nodes)
        isolated_norm_support = bool(isolated_norm_margin_nodes)
        branch_components_insufficient = not attention_support and not mlp_support
        residual_composition = bool(
            transition
            and branch_components_insufficient
            and not incoming_state_support
            and residual_node_support
            and not isolated_norm_support
        )
        residual_norm = bool(
            transition
            and branch_components_insufficient
            and not incoming_state_support
            and isolated_norm_support
        )
        incoming_state_confound = bool(
            transition
            and branch_components_insufficient
            and incoming_state_support
            and (residual_node_support or norm_node_support)
        )
        if not transition:
            component = "unresolved"
        elif attention_support and mlp_support:
            component = "mixed_attention_mlp"
        elif attention_support:
            component = "attention_output"
        elif mlp_support:
            component = "mlp"
        elif residual_norm:
            # Fixed precedence prevents outcome-dependent choice when both a
            # residual node and a normalization node satisfy their gates.
            component = "residual_norm_interaction"
        elif residual_composition:
            component = "residual_composition"
        else:
            component = "unresolved"
        if component != "unresolved":
            evidence_strength = "registered_confirmatory"
        elif localization_status == "unresolved":
            evidence_strength = "exploratory_only"
        elif localization_status in {
            "fold0_no_negative_transition",
            "not_run_due_to_gate_or_mechanical_stop",
        }:
            evidence_strength = "gate_stopped"
        else:
            evidence_strength = "unresolved"
        ndcg_corroborated = {
            "attention_output": ndcg_support["attention_o_projection"],
            "mlp": ndcg_support["mlp_down_projection"],
            "mixed_attention_mlp": (
                ndcg_support["attention_o_projection"]
                and ndcg_support["mlp_down_projection"]
            ),
            "residual_composition": bool(
                residual_margin_nodes & residual_ndcg_nodes
            ),
            "residual_norm_interaction": bool(
                isolated_norm_margin_nodes & isolated_norm_ndcg_nodes
            ),
            "unresolved": False,
        }[component]
        ndcg_utility_corroborated = {
            "attention_output": ndcg_utility_support[
                "attention_o_projection"
            ],
            "mlp": ndcg_utility_support["mlp_down_projection"],
            "mixed_attention_mlp": (
                ndcg_utility_support["attention_o_projection"]
                and ndcg_utility_support["mlp_down_projection"]
            ),
            "residual_composition": bool(
                residual_margin_nodes & residual_ndcg_utility_nodes
            ),
            "residual_norm_interaction": bool(
                isolated_norm_margin_nodes & isolated_norm_ndcg_utility_nodes
            ),
            "unresolved": False,
        }[component]
        transfer_scope = (
            "unresolved"
            if component == "unresolved"
            else (
                "target_margin_primary_with_utility_relevant_ndcg_corroboration"
                if ndcg_utility_corroborated
                else (
                    "target_margin_primary_with_statistical_ndcg_corroboration_only"
                    if ndcg_corroborated
                    else "target_margin_only"
                )
            )
        )
        transfer_explanation_level = _derive_transfer_explanation_level(
            component=component,
            fold1_transition_reproduced=transition,
            ndcg_corroborated=ndcg_corroborated,
            utility_ndcg_corroborated=ndcg_utility_corroborated,
        )
        rows.append(
            {
                "method_id": method_id,
                "localization_status": localization_status or "unspecified",
                "fold1_transition_reproduced": transition,
                "postblock_registered_support": transition,
                "attention_branch_registered_support": attention_support,
                "mlp_branch_registered_support": mlp_support,
                "incoming_block_state_registered_support": incoming_state_support,
                "incoming_state_confounds_residual_or_norm_attribution": (
                    incoming_state_confound
                ),
                "residual_node_registered_support": residual_node_support,
                "normalization_node_registered_support": norm_node_support,
                "normalization_boundary_isolated_registered_support": (
                    isolated_norm_support
                ),
                "normalization_boundary_isolated_nodes": sorted(
                    isolated_norm_margin_nodes
                ),
                "normalization_state_support_without_boundary_isolation": bool(
                    unisolated_norm_margin_nodes
                ),
                "normalization_nodes_without_boundary_isolation": sorted(
                    unisolated_norm_margin_nodes
                ),
                "attention_branch_ndcg_registered_support": ndcg_support[
                    "attention_o_projection"
                ],
                "mlp_branch_ndcg_registered_support": ndcg_support[
                    "mlp_down_projection"
                ],
                "attention_branch_ndcg_beyond_equivalence_support": (
                    ndcg_utility_support["attention_o_projection"]
                ),
                "mlp_branch_ndcg_beyond_equivalence_support": (
                    ndcg_utility_support["mlp_down_projection"]
                ),
                "residual_nodes_with_both_endpoint_support": sorted(
                    residual_margin_nodes & residual_ndcg_nodes
                ),
                "normalization_nodes_with_both_endpoint_support": sorted(
                    norm_margin_nodes & norm_ndcg_nodes
                ),
                "normalization_boundaries_with_both_endpoint_isolation": sorted(
                    isolated_norm_margin_nodes & isolated_norm_ndcg_nodes
                ),
                "residual_composition_criterion_met": residual_composition,
                "residual_norm_interaction_criterion_met": residual_norm,
                "derived_primary_component": component,
                "derived_evidence_strength": evidence_strength,
                "derived_resolution": component != "unresolved",
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
                "strict_transfer_ndcg_component_corroborated": ndcg_corroborated,
                "strict_transfer_ndcg_beyond_equivalence_component_corroborated": (
                    ndcg_utility_corroborated
                ),
                "target_margin_component_is_not_automatically_ndcg_cause": True,
                "transfer_failure_causal_scope": transfer_scope,
                "transfer_explanation_level": transfer_explanation_level,
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
                "descriptive_head_or_group_used_as_primary_cause": False,
                "raw_effect_values_emitted": False,
            }
        )
    return rows


def _derive_transfer_explanation_level(
    *,
    component: str,
    fold1_transition_reproduced: bool,
    ndcg_corroborated: bool,
    utility_ndcg_corroborated: bool,
) -> str:
    """Return the strongest authorized explanation without causal overreach."""

    if component == "unresolved":
        return (
            "reproduced_layer_transition_without_unique_component"
            if fold1_transition_reproduced
            else "unresolved_or_gate_stopped"
        )
    if utility_ndcg_corroborated:
        return (
            "target_margin_component_with_utility_relevant_ndcg_corroboration"
        )
    if ndcg_corroborated:
        return "target_margin_component_with_statistical_ndcg_corroboration"
    return "target_margin_component_sufficiency"


def _registered_node_support(
    rows: Sequence[Any],
    method_id: str,
    node: str,
    *,
    endpoint: str = "target_margin",
    require_ndcg_beyond_equivalence: bool = False,
) -> bool:
    if require_ndcg_beyond_equivalence and endpoint != "ndcg@10":
        raise DeepDiveReportContractError(
            "NDCG equivalence gate requested for a non-NDCG endpoint"
        )
    required = {
        f"same__{node}",
        f"same_minus_cross__{node}",
        f"same_minus_wrong__{node}",
        f"norm__{node}",
        f"direction__{node}",
        f"random__{node}",
    }
    matched = [
        row
        for row in rows
        if isinstance(row, Mapping)
        and row.get("method_id") == method_id
        and row.get("endpoint") == endpoint
        and str(row.get("contrast_id")) in required
    ]
    contrast_ids = [str(row.get("contrast_id")) for row in matched]
    if len(contrast_ids) != len(set(contrast_ids)):
        raise DeepDiveReportContractError(
            "duplicate selected-branch contrast in primary attribution: "
            f"{method_id}:{node}:{endpoint}"
        )
    observed = {str(row["contrast_id"]): row for row in matched}
    return set(observed) == required and all(
        item.get("registered_support") is True
        and item.get("missing") is False
        and item.get("evidence_role")
        == "registered_confirmatory_branch_localization"
        and (
            not require_ndcg_beyond_equivalence
            or _negative_ndcg_ci_beyond_equivalence(item.get("ci95"))
        )
        for item in observed.values()
    )


def _negative_ndcg_ci_beyond_equivalence(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return False
    try:
        lower, upper = map(float, value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(lower) and math.isfinite(upper) and upper < -0.005


def _read_report_evidence(path: Path, analysis_type: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DeepDiveReportContractError(
            f"cannot read admitted attribution evidence: {path}"
        ) from exc
    if (
        not isinstance(value, dict)
        or value.get("status") != "completed"
        or value.get("analysis_type") != analysis_type
    ):
        raise DeepDiveReportContractError(
            f"admitted attribution evidence schema differs: {path}"
        )
    return value


RESULT_LEVEL_ANALYSIS_TYPES = {
    "d2_postblock": "transformer_deep_dive_d2_postblock_synthesis",
    "d2_selected_branches": (
        "transformer_deep_dive_d2_selected_branch_synthesis"
    ),
    "d3_attention_edges": "transformer_deep_dive_d3_attention_edges",
    "d5_context": "transformer_deep_dive_d5_contextual_controls",
    "d5_rope": "transformer_deep_dive_d5_rope",
    "d6_q2_native_readout": (
        "transformer_deep_dive_d6_q2_native_readout"
    ),
    "d6_q3_native_readout": (
        "transformer_deep_dive_d6_q3_native_readout"
    ),
    "d6_q0_q1_branches": (
        "transformer_deep_dive_d6_q0_q1_branch_extension"
    ),
    "d7_q2_objective": (
        "transformer_deep_dive_d7_q2_objective_conflict"
    ),
}
RESULT_LEVEL_SUPPORTED_COMPONENT_ROUTES = frozenset(
    {
        ("serialization_tokenization", "d5_context"),
        ("positional_encoding_rope", "d5_rope"),
        ("attention_query_key_routing", "d3_attention_edges"),
        ("attention_query_key_routing", "d5_rope"),
        ("attention_value_transport", "d3_attention_edges"),
        ("attention_output", "d2_selected_branches"),
        ("attention_output", "d6_q0_q1_branches"),
        ("mlp_output", "d2_selected_branches"),
        ("mlp_output", "d6_q0_q1_branches"),
        ("residual_composition", "d2_selected_branches"),
        ("normalization", "d2_selected_branches"),
        ("layerwise_representation", "d2_postblock"),
        ("history_routing", "d2_selected_branches"),
        ("history_routing", "d3_attention_edges"),
        ("history_routing", "d5_context"),
        ("candidate_conditioned_interaction", "d2_postblock"),
        ("candidate_conditioned_interaction", "d2_selected_branches"),
        ("candidate_conditioned_interaction", "d3_attention_edges"),
        ("candidate_conditioned_interaction", "d5_context"),
        ("candidate_conditioned_interaction", "d5_rope"),
        ("native_readout", "d6_q2_native_readout"),
        ("native_readout", "d6_q3_native_readout"),
        ("score_calibration_nullspace", "d6_q2_native_readout"),
        ("score_calibration_nullspace", "d6_q3_native_readout"),
        ("loss_gradient", "d7_q2_objective"),
    }
)
RESULT_LEVEL_EQUIVALENCE_COMPONENT_ROUTES = frozenset(
    {
        ("positional_encoding_rope", "d5_rope"),
        ("native_readout", "d6_q2_native_readout"),
        ("native_readout", "d6_q3_native_readout"),
        ("loss_gradient", "d7_q2_objective"),
    }
)
RESULT_LEVEL_EQUIVALENCE_COMPONENT_GATE_DESCRIPTIONS = {
    ("positional_encoding_rope", "d5_rope"): (
        "For every fixed block 13/20/27 and every readout_q/history_k/paired_qk "
        "cell, both registered NDCG compression-minus-expansion and "
        "compression-minus-baseline all-population CIs lie wholly inside "
        "+/-0.005; all/fold0/fold1 rows are present and finite"
    ),
    ("native_readout", "d6_q2_native_readout"): (
        "For both final_rmsnorm_input and final_rmsnorm_output, all three "
        "registered comparison structures are complete, and same-minus-null "
        "and same-minus-cross NDCG all-population CIs lie wholly inside "
        "+/-0.005; all/fold0/fold1 rows are present and finite"
    ),
    ("native_readout", "d6_q3_native_readout"): (
        "For shared_prompt, yes_context, no_context, and joint scopes, "
        "all three registered comparison structures are complete, and "
        "same-minus-null and same-minus-cross NDCG all-population CIs lie "
        "wholly inside +/-0.005; all/fold0/fold1 rows are present and finite"
    ),
    ("loss_gradient", "d7_q2_objective"): (
        "The exact 2-state x 3-surface x 2-endpoint family is complete and all "
        "six RankNet-ListNet cosine rows have their registered CI wholly inside "
        "the +/-0.1 SESOI"
    ),
}
if set(RESULT_LEVEL_EQUIVALENCE_COMPONENT_GATE_DESCRIPTIONS) != set(
    RESULT_LEVEL_EQUIVALENCE_COMPONENT_ROUTES
):
    raise AssertionError(
        "component practical-equivalence gate descriptions must cover every route"
    )
RESULT_LEVEL_SUPPORTED_COMPONENT_GATE_DESCRIPTIONS = {
    ("serialization_tokenization", "d5_context"): (
        "history_content_neutral; either registered endpoint; BH q<0.05, "
        "all/fold0/fold1 same nonzero direction, all CI excludes zero"
    ),
    ("positional_encoding_rope", "d5_rope"): (
        "NDCG compression-minus-expansion passes BH/fold/CI gates and "
        "compression-minus-baseline CI lies wholly outside +/-0.005"
    ),
    ("attention_query_key_routing", "d3_attention_edges"): (
        "history_logits_mask only; either registered endpoint; BH/fold/CI gates"
    ),
    ("attention_query_key_routing", "d5_rope"): (
        "registered RoPE dual gate on Q/K phase-distance intervention"
    ),
    ("attention_value_transport", "d3_attention_edges"): (
        "history_value_edge_zero only; either registered endpoint; BH/fold/CI gates"
    ),
    ("attention_output", "d2_selected_branches"): (
        "attention_o_projection passes all six target-margin same/stress/"
        "specificity/direction-scale confirmatory gates"
    ),
    ("attention_output", "d6_q0_q1_branches"): (
        "attention_o_projection same-minus-null; one registered endpoint; "
        "BH/fold/CI gates"
    ),
    ("mlp_output", "d2_selected_branches"): (
        "mlp_down_projection passes all six target-margin same/stress/"
        "specificity/direction-scale confirmatory gates"
    ),
    ("mlp_output", "d6_q0_q1_branches"): (
        "mlp_down_projection same-minus-null; one registered endpoint; "
        "BH/fold/CI gates"
    ),
    ("residual_composition", "d2_selected_branches"): (
        "attention/MLP are insufficient, incoming block state is insufficient, a "
        "post_attention_residual or block_output_residual passes all six target-margin "
        "confirmatory gates, and no isolated norm boundary takes precedence"
    ),
    ("normalization", "d2_selected_branches"): (
        "attention/MLP and incoming block state are insufficient, and an RMSNorm output "
        "passes all six target-margin gates while its paired pre-norm state does not"
    ),
    ("layerwise_representation", "d2_postblock"): (
        "frozen fold0-selected adjacent transition reproduces on fold1"
    ),
    ("history_routing", "d2_selected_branches"): (
        "at least one selected node passes all six history-specific "
        "target-margin confirmatory gates"
    ),
    ("history_routing", "d3_attention_edges"): (
        "one registered history logits/value/neutral-KV intervention passes "
        "BH/fold/CI gates"
    ),
    ("history_routing", "d5_context"): (
        "history_attention_null; either registered endpoint; BH/fold/CI gates"
    ),
    ("candidate_conditioned_interaction", "d2_postblock"): (
        "frozen fold0-selected candidate-margin transition reproduces on fold1"
    ),
    ("candidate_conditioned_interaction", "d2_selected_branches"): (
        "at least one selected node passes all six target-margin confirmatory gates"
    ),
    ("candidate_conditioned_interaction", "d3_attention_edges"): (
        "one registered history edge intervention changes a registered ranking "
        "endpoint with BH/fold/CI support"
    ),
    ("candidate_conditioned_interaction", "d5_context"): (
        "content-neutral or attention-null context intervention passes "
        "BH/fold/CI gates"
    ),
    ("candidate_conditioned_interaction", "d5_rope"): (
        "registered RoPE dual gate changes strict-transfer NDCG"
    ),
    ("native_readout", "d6_q2_native_readout"): (
        "one final-norm node has same-minus-null and same-minus-cross support "
        "on the same endpoint, each passing BH/fold/CI gates"
    ),
    ("native_readout", "d6_q3_native_readout"): (
        "one native readout scope has same-minus-null and same-minus-cross "
        "support on the same endpoint, each passing BH/fold/CI gates"
    ),
    ("score_calibration_nullspace", "d6_q2_native_readout"): (
        "qrels-blind exact score=common+relative recomposition and zero-sum "
        "relative identity within fixed numerical tolerances"
    ),
    ("score_calibration_nullspace", "d6_q3_native_readout"): (
        "qrels-blind exact score=common+relative recomposition and zero-sum "
        "relative identity within fixed numerical tolerances"
    ),
    ("loss_gradient", "d7_q2_objective"): (
        "RankNet-ListNet cosine CI lies below -0.1 SESOI and BH q<0.05"
    ),
}
D7_OBJECTIVE_FAMILY_STATES = (
    "base_initialization",
    "frozen_final_checkpoint",
)
D7_OBJECTIVE_FAMILY_SURFACES = (
    "recurrence",
    "strict_transfer",
    "other_overlap",
)
D7_OBJECTIVE_FAMILY_ENDPOINTS = (
    "ranknet_listnet_cosine",
    "observed_minus_label_shuffle_cosine",
)
D7_OBJECTIVE_FAMILY_KEYS = frozenset(
    (state, surface, endpoint)
    for state in D7_OBJECTIVE_FAMILY_STATES
    for surface in D7_OBJECTIVE_FAMILY_SURFACES
    for endpoint in D7_OBJECTIVE_FAMILY_ENDPOINTS
)


def _validate_supported_components_against_evidence(
    root: Path, decisions: Mapping[str, Any]
) -> None:
    """Require support and registered-equivalence claims to pass result gates."""

    cache: dict[str, dict[str, Any]] = {}
    for row in _rows(
        decisions.get("component_evidence_matrix"), "component evidence matrix"
    ):
        if (
            row.get("status") == "weakened"
            and row.get("negative_evidence_basis")
            == "registered_practical_equivalence"
        ):
            component_id = str(row["component_id"])
            cited = [
                str(item)
                for item in row.get("evidence_deliverables", [])
                if (component_id, str(item))
                in RESULT_LEVEL_EQUIVALENCE_COMPONENT_ROUTES
            ]
            if not cited:
                raise DeepDiveReportContractError(
                    "equivalence-weakened component lacks a registered equivalence route: "
                    f"{component_id}"
                )
            for method_id in map(str, row.get("model_scope", [])):
                equivalent = False
                for deliverable in cited:
                    if method_id not in COMPONENT_DELIVERABLE_MODEL_COVERAGE[
                        component_id
                    ][deliverable]:
                        continue
                    if deliverable not in cache:
                        cache[deliverable] = _read_report_evidence(
                            root / EXPECTED_DELIVERABLES[deliverable],
                            RESULT_LEVEL_ANALYSIS_TYPES[deliverable],
                        )
                    if component_result_practical_equivalence(
                        component_id,
                        method_id,
                        deliverable,
                        cache[deliverable],
                    ):
                        equivalent = True
                        break
                if not equivalent:
                    raise DeepDiveReportContractError(
                        "equivalence-weakened component differs from registered result "
                        f"evidence: {component_id}:{method_id}"
                    )
            continue
        if row.get("status") != "supported":
            continue
        component_id = str(row["component_id"])
        causal = COMPONENT_SUPPORT_REQUIRED_CAUSAL_EVIDENCE[component_id]
        cited = [
            str(item)
            for item in row.get("evidence_deliverables", [])
            if str(item) in causal
        ]
        for method_id in map(str, row.get("model_scope", [])):
            supports = False
            for deliverable in cited:
                if (
                    component_id,
                    deliverable,
                ) not in RESULT_LEVEL_SUPPORTED_COMPONENT_ROUTES:
                    raise DeepDiveReportContractError(
                        "supported component has no result-level component route: "
                        f"{component_id}:{deliverable}"
                    )
                if method_id not in COMPONENT_DELIVERABLE_MODEL_COVERAGE[
                    component_id
                ][deliverable]:
                    continue
                if deliverable not in cache:
                    analysis_type = RESULT_LEVEL_ANALYSIS_TYPES.get(deliverable)
                    if analysis_type is None:
                        raise DeepDiveReportContractError(
                            "supported component has no result-level evidence route: "
                            f"{component_id}:{deliverable}"
                        )
                    cache[deliverable] = _read_report_evidence(
                        root / EXPECTED_DELIVERABLES[deliverable], analysis_type
                    )
                if component_result_support(
                    component_id, method_id, deliverable, cache[deliverable]
                ):
                    supports = True
                    break
            if not supports:
                raise DeepDiveReportContractError(
                    "supported component differs from registered result evidence: "
                    f"{component_id}:{method_id}"
                )


def component_result_support(
    component_id: str,
    method_id: str,
    deliverable: str,
    metrics: Mapping[str, Any],
) -> bool:
    """Route a component claim to its outcome-independent registered gate."""

    if deliverable == "d2_selected_branches":
        rows = metrics.get("rows")
        if not isinstance(rows, list):
            raise DeepDiveReportContractError("D2 selected-branch schema differs")
        if component_id in {"residual_composition", "normalization"}:
            attention_support = _registered_node_support(
                rows, method_id, "attention_o_projection"
            )
            mlp_support = _registered_node_support(
                rows, method_id, "mlp_down_projection"
            )
            incoming_support = _registered_node_support(
                rows, method_id, "block_input_residual"
            )
            residual_support = any(
                _registered_node_support(rows, method_id, node)
                for node in (
                    "post_attention_residual",
                    "block_output_residual",
                )
            )
            isolated_norm_support = any(
                _registered_node_support(rows, method_id, post_norm)
                and not _registered_node_support(rows, method_id, pre_norm)
                for pre_norm, post_norm in (
                    ("block_input_residual", "input_rmsnorm_output"),
                    (
                        "post_attention_residual",
                        "post_attention_rmsnorm_output",
                    ),
                )
            )
            branch_components_insufficient = (
                not attention_support and not mlp_support
            )
            if component_id == "normalization":
                return bool(
                    branch_components_insufficient
                    and not incoming_support
                    and isolated_norm_support
                )
            return bool(
                branch_components_insufficient
                and not incoming_support
                and residual_support
                and not isolated_norm_support
            )
        node_routes = {
            "attention_output": ("attention_o_projection",),
            "mlp_output": ("mlp_down_projection",),
            "history_routing": (
                "block_input_residual",
                "input_rmsnorm_output",
                "attention_o_projection",
                "post_attention_residual",
                "post_attention_rmsnorm_output",
                "mlp_down_projection",
                "block_output_residual",
            ),
            "candidate_conditioned_interaction": (
                "block_input_residual",
                "input_rmsnorm_output",
                "attention_o_projection",
                "post_attention_residual",
                "post_attention_rmsnorm_output",
                "mlp_down_projection",
                "block_output_residual",
            ),
        }
        nodes = node_routes.get(component_id)
        return bool(
            nodes
            and any(
                _registered_node_support(rows, method_id, node) for node in nodes
            )
        )
    if deliverable == "d2_postblock":
        localization = metrics.get("localization")
        return bool(
            component_id
            in {"layerwise_representation", "candidate_conditioned_interaction"}
            and isinstance(localization, Mapping)
            and isinstance(localization.get(method_id), Mapping)
            and localization[method_id].get("resolved") is True
        )
    if deliverable == "d3_attention_edges":
        conditions = {
            "attention_query_key_routing": ("history_logits_mask",),
            "attention_value_transport": ("history_value_edge_zero",),
            "history_routing": (
                "history_logits_mask",
                "history_value_edge_zero",
                "neutral_history_kv",
            ),
            "candidate_conditioned_interaction": (
                "history_logits_mask",
                "history_value_edge_zero",
                "neutral_history_kv",
            ),
        }.get(component_id, ())
        return _nested_registered_support(
            metrics,
            method_id=method_id,
            dimension_name="condition",
            dimension_values=conditions,
            inference_key="registered",
        )
    if deliverable == "d5_context":
        conditions = {
            "serialization_tokenization": ("history_content_neutral",),
            "history_routing": ("history_attention_null",),
            "candidate_conditioned_interaction": (
                "history_content_neutral",
                "history_attention_null",
            ),
        }.get(component_id, ())
        return _nested_registered_support(
            metrics,
            method_id=method_id,
            dimension_name="condition",
            dimension_values=conditions,
            inference_key="registered",
        )
    if deliverable == "d5_rope":
        return bool(
            component_id
            in {
                "positional_encoding_rope",
                "attention_query_key_routing",
                "candidate_conditioned_interaction",
            }
            and _rope_registered_support(metrics, method_id)
        )
    if deliverable in {"d6_q2_native_readout", "d6_q3_native_readout"}:
        if component_id == "native_readout":
            return _native_readout_registered_support(metrics, method_id)
        if component_id == "score_calibration_nullspace":
            return _readout_nullspace_identity_support(metrics, method_id)
        return False
    if deliverable == "d6_q0_q1_branches":
        node = {
            "attention_output": "attention_o_projection",
            "mlp_output": "mlp_down_projection",
        }.get(component_id)
        return bool(
            node
            and _breadth_branch_registered_support(metrics, method_id, node)
        )
    if deliverable == "d7_q2_objective":
        family_rows = metrics.get("family_rows")
        if not isinstance(family_rows, list):
            return False
        family_keys = [
            (row.get("state"), row.get("surface"), row.get("endpoint"))
            for row in family_rows
            if isinstance(row, Mapping)
        ]
        if len(family_keys) != len(set(family_keys)):
            raise DeepDiveReportContractError(
                "duplicate D7 objective-conflict family key"
            )
        if set(family_keys) != D7_OBJECTIVE_FAMILY_KEYS:
            raise DeepDiveReportContractError(
                "D7 objective-conflict family key coverage differs"
            )
        return bool(
            component_id == "loss_gradient"
            and method_id == MODEL_IDS[2]
            and any(
                isinstance(row, Mapping)
                and row.get("endpoint") == "ranknet_listnet_cosine"
                and row.get("conflict_beyond_sesoi") is True
                and row.get("bh_q_below_0.05") is True
                for row in family_rows
            )
        )
    raise DeepDiveReportContractError(
        "supported component has an unimplemented result route: "
        f"{component_id}:{deliverable}"
    )


def component_result_practical_equivalence(
    component_id: str,
    method_id: str,
    deliverable: str,
    metrics: Mapping[str, Any],
) -> bool:
    """Admit weakening only through a preregistered complete SESOI gate."""

    if (component_id, deliverable) not in RESULT_LEVEL_EQUIVALENCE_COMPONENT_ROUTES:
        return False
    if deliverable == "d5_rope":
        return _rope_complete_practical_equivalence(metrics, method_id)
    if deliverable == "d6_q2_native_readout":
        if method_id != MODEL_IDS[2]:
            return False
        return _native_readout_complete_practical_equivalence(metrics, method_id)
    if deliverable == "d6_q3_native_readout":
        if method_id != MODEL_IDS[3]:
            return False
        return _native_readout_complete_practical_equivalence(metrics, method_id)
    if deliverable == "d7_q2_objective":
        rows = metrics.get("family_rows")
        if not isinstance(rows, list):
            return False
        keys = [
            (row.get("state"), row.get("surface"), row.get("endpoint"))
            for row in rows
            if isinstance(row, Mapping)
        ]
        cosine_rows = [
            row
            for row in rows
            if isinstance(row, Mapping)
            and row.get("endpoint") == "ranknet_listnet_cosine"
        ]
        return bool(
            component_id == "loss_gradient"
            and method_id == MODEL_IDS[2]
            and len(rows) == len(D7_OBJECTIVE_FAMILY_KEYS)
            and len(keys) == len(set(keys))
            and set(keys) == D7_OBJECTIVE_FAMILY_KEYS
            and len(cosine_rows)
            == len(D7_OBJECTIVE_FAMILY_STATES)
            * len(D7_OBJECTIVE_FAMILY_SURFACES)
            and all(
                row.get("practical_equivalence_within_sesoi") is True
                for row in cosine_rows
            )
        )
    return False


def _rope_complete_practical_equivalence(
    metrics: Mapping[str, Any], method_id: str
) -> bool:
    gate = metrics.get("position_support_gate")
    if not isinstance(gate, Mapping) or gate.get(
        "active_ci95_equivalence_band"
    ) != [-0.005, 0.005]:
        return False
    results = metrics.get("results")
    model_results = results.get(method_id) if isinstance(results, Mapping) else None
    if (
        not isinstance(model_results, Mapping)
        or set(map(str, model_results)) != {"13", "20", "27"}
    ):
        return False
    cells = []
    for block_results in model_results.values():
        if (
            not isinstance(block_results, Mapping)
            or set(block_results) != {"readout_q", "history_k", "paired_qk"}
        ):
            return False
        for contrast_results in block_results.values():
            if not isinstance(contrast_results, Mapping):
                return False
            ndcg = contrast_results.get("ndcg@10")
            if not isinstance(ndcg, Mapping):
                return False
            cells.append(ndcg)
    return bool(
        len(cells) == 9
        and all(
            _all_population_ci_within_equivalence(
                cell.get("registered_compression_minus_expansion"), 0.005
            )
            and _all_population_ci_within_equivalence(
                cell.get("registered_compression_minus_baseline_support_gate"),
                0.005,
            )
            for cell in cells
        )
    )


def _native_readout_complete_practical_equivalence(
    metrics: Mapping[str, Any], method_id: str
) -> bool:
    if metrics.get("method_id") != method_id:
        return False
    results = metrics.get("results")
    if not isinstance(results, Mapping):
        return False
    expected_scopes = (
        {"final_rmsnorm_input", "final_rmsnorm_output"}
        if method_id == MODEL_IDS[2]
        else {"shared_prompt", "yes_context", "no_context", "joint"}
        if method_id == MODEL_IDS[3]
        else set()
    )
    if set(results) != expected_scopes or not expected_scopes:
        return False
    return all(
        isinstance(scope, Mapping)
        and set(scope)
        == {"same_minus_null", "same_minus_full", "same_minus_cross"}
        and all(
            isinstance(scope.get(comparison), Mapping)
            and _all_population_ci_within_equivalence(
                scope[comparison].get("ndcg@10"), 0.005
            )
            for comparison in ("same_minus_null", "same_minus_cross")
        )
        and isinstance(scope.get("same_minus_full"), Mapping)
        and _all_population_ci_bounds(
            scope["same_minus_full"].get("ndcg@10")
        )
        is not None
        for scope in results.values()
    )


def _all_population_ci_within_equivalence(rows: Any, bound: float) -> bool:
    bounds = _all_population_ci_bounds(rows)
    if bounds is None:
        return False
    lower, upper = bounds
    return bool(lower >= -bound and upper <= bound)


def _all_population_ci_bounds(rows: Any) -> tuple[float, float] | None:
    if not isinstance(rows, list):
        return None
    if len(rows) != 3 or not all(isinstance(row, Mapping) for row in rows):
        return None
    by_fold = {
        str(row.get("normalized_query_fold")): row
        for row in rows
    }
    if len(by_fold) != len(rows) or set(by_fold) != {"all", "0", "1"}:
        return None
    try:
        means = [float(by_fold[fold]["mean"]) for fold in ("all", "0", "1")]
        lower, upper = map(float, by_fold["all"]["ci95"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (
        all(math.isfinite(mean) for mean in means)
        and math.isfinite(lower)
        and math.isfinite(upper)
        and lower <= upper
    ):
        return None
    return lower, upper


def _nested_registered_support(
    metrics: Mapping[str, Any],
    *,
    method_id: str,
    dimension_name: str,
    dimension_values: Sequence[str],
    inference_key: str,
) -> bool:
    """Check any registered model/block/condition/endpoint cell fail-closed."""

    results = metrics.get("results")
    if not dimension_values or not isinstance(results, Mapping):
        return False
    method_results = results.get(method_id)
    if not isinstance(method_results, Mapping):
        return False
    blocks = (
        method_results.values()
        if all(str(key).isdigit() for key in method_results)
        else (method_results,)
    )
    for block_results in blocks:
        if not isinstance(block_results, Mapping):
            continue
        for dimension in dimension_values:
            dimension_results = block_results.get(dimension)
            if not isinstance(dimension_results, Mapping):
                continue
            for endpoint, endpoint_result in dimension_results.items():
                if endpoint not in {"target_margin", "ndcg@10"} or not isinstance(
                    endpoint_result, Mapping
                ):
                    continue
                rows = endpoint_result.get(inference_key)
                q_value = _family_q(
                    metrics,
                    method_id=method_id,
                    **{dimension_name: dimension, "endpoint": endpoint},
                    block_zero_based=(
                        int(next(
                            key
                            for key, value in method_results.items()
                            if value is block_results
                        ))
                        if block_results is not method_results
                        else None
                    ),
                )
                if _stable_registered_effect(rows, q_value):
                    return True
    return False


def _family_q(metrics: Mapping[str, Any], **filters: Any) -> float | None:
    active_filters = {
        key: value for key, value in filters.items() if value is not None
    }
    rows = [
        row
        for row in metrics.get("family_rows", [])
        if isinstance(row, Mapping)
        and all(row.get(key) == value for key, value in active_filters.items())
    ]
    if len(rows) != 1:
        return None
    try:
        value = float(rows[0]["bh_q"])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _stable_registered_effect(rows: Any, bh_q: float | None) -> bool:
    if not isinstance(rows, list) or bh_q is None or not bh_q < 0.05:
        return False
    by_fold = {
        str(row.get("normalized_query_fold")): row
        for row in rows
        if isinstance(row, Mapping)
    }
    if set(by_fold) != {"all", "0", "1"}:
        return False
    try:
        means = [float(by_fold[fold]["mean"]) for fold in ("all", "0", "1")]
        lower, upper = map(float, by_fold["all"]["ci95"])
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        all(math.isfinite(value) and value != 0.0 for value in means)
        and (all(value > 0.0 for value in means) or all(value < 0.0 for value in means))
        and math.isfinite(lower)
        and math.isfinite(upper)
        and (lower > 0.0 or upper < 0.0)
    )


def _rope_registered_support(metrics: Mapping[str, Any], method_id: str) -> bool:
    if metrics.get("position_support_gate") != {
        "active_contrast": "compression_minus_baseline",
        "active_endpoint": "ndcg@10",
        "active_ci95_equivalence_band": [-0.005, 0.005],
        "requires_compression_minus_expansion_bh_q_below_alpha_0p05": True,
        "requires_all_fold0_fold1_same_nonzero_direction": True,
        "active_contrast_is_confirmatory_family_member": False,
    }:
        return False
    results = metrics.get("results")
    if not isinstance(results, Mapping) or not isinstance(
        results.get(method_id), Mapping
    ):
        return False
    for block, block_results in results[method_id].items():
        if not isinstance(block_results, Mapping):
            continue
        for contrast, contrast_results in block_results.items():
            if not isinstance(contrast_results, Mapping):
                continue
            endpoint = contrast_results.get("ndcg@10")
            if not isinstance(endpoint, Mapping):
                continue
            comparison_rows = endpoint.get(
                "registered_compression_minus_expansion"
            )
            q_value = _family_q(
                metrics,
                method_id=method_id,
                block_zero_based=int(block),
                contrast=contrast,
                endpoint="ndcg@10",
            )
            if not _stable_registered_effect(comparison_rows, q_value):
                continue
            active_rows = endpoint.get(
                "registered_compression_minus_baseline_support_gate"
            )
            if _stable_equivalence_band_exclusion(active_rows, 0.005):
                return True
    return False


def _stable_equivalence_band_exclusion(rows: Any, bound: float) -> bool:
    if not isinstance(rows, list):
        return False
    by_fold = {
        str(row.get("normalized_query_fold")): row
        for row in rows
        if isinstance(row, Mapping)
    }
    if set(by_fold) != {"all", "0", "1"}:
        return False
    try:
        means = [float(by_fold[fold]["mean"]) for fold in ("all", "0", "1")]
        lower, upper = map(float, by_fold["all"]["ci95"])
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        all(math.isfinite(value) and value != 0.0 for value in means)
        and (all(value > 0.0 for value in means) or all(value < 0.0 for value in means))
        and math.isfinite(lower)
        and math.isfinite(upper)
        and (lower > bound or upper < -bound)
    )


def _native_readout_registered_support(
    metrics: Mapping[str, Any], method_id: str
) -> bool:
    if metrics.get("method_id") != method_id:
        return False
    results = metrics.get("results")
    if not isinstance(results, Mapping):
        return False
    for node, node_results in results.items():
        if not isinstance(node_results, Mapping):
            continue
        for endpoint in ("target_margin", "ndcg@10"):
            if all(
                isinstance(node_results.get(comparison), Mapping)
                and _stable_registered_effect(
                    node_results[comparison].get(endpoint),
                    _family_q(
                        metrics,
                        **(
                            {"node": node}
                            if method_id == MODEL_IDS[2]
                            else {"readout_scope": node}
                        ),
                        comparison=comparison,
                        endpoint=endpoint,
                    ),
                )
                for comparison in ("same_minus_null", "same_minus_cross")
            ):
                return True
    return False


def _readout_nullspace_identity_support(
    metrics: Mapping[str, Any], method_id: str
) -> bool:
    """Admit only the exact rank-null common/relative score identity."""

    if metrics.get("method_id") != method_id:
        return False
    decomposition = metrics.get("readout_decomposition")
    if not isinstance(decomposition, Mapping):
        return False
    algebra = decomposition.get("algebra")
    if not isinstance(algebra, Mapping):
        return False
    try:
        recomposition = float(algebra["maximum_recomposition_abs_error"])
        relative_sum = float(algebra["maximum_relative_sum_abs_error"])
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        decomposition.get("qrels_read") is False
        and decomposition.get("confirmatory_family_membership") is False
        and algebra.get("score_identity") == "score_ij = common_i + relative_ij"
        and algebra.get("common_definition") == "mean_j(score_ij)"
        and algebra.get("relative_definition") == "score_ij - common_i"
        and math.isfinite(recomposition)
        and recomposition <= 1.0e-12
        and math.isfinite(relative_sum)
        and relative_sum <= 1.0e-9
    )


def _breadth_branch_registered_support(
    metrics: Mapping[str, Any], method_id: str, node: str
) -> bool:
    results = metrics.get("results")
    if not isinstance(results, Mapping) or not isinstance(
        results.get(method_id), Mapping
    ):
        return False
    for block, block_results in results[method_id].items():
        if not isinstance(block_results, Mapping):
            continue
        node_results = block_results.get(node)
        if not isinstance(node_results, Mapping):
            continue
        endpoint_results = node_results.get("same_minus_null")
        if not isinstance(endpoint_results, Mapping):
            continue
        for endpoint, inference in endpoint_results.items():
            if endpoint in {"target_margin", "ndcg@10"} and _stable_registered_effect(
                inference,
                _family_q(
                    metrics,
                    method_id=method_id,
                    block_zero_based=int(block),
                    node=node,
                    comparison="same_minus_null",
                    endpoint=endpoint,
                ),
            ):
                return True
    return False


def _validate_opportunities(value: Any, admitted: set[str]) -> None:
    rows = _rows(value, "architecture opportunity ranking")
    _require_exact_ids(
        rows, "opportunity_id", OPPORTUNITY_IDS, "architecture opportunity ranking"
    )
    ranks = [row.get("rank") for row in rows]
    if sorted(ranks) != list(range(1, len(OPPORTUNITY_IDS) + 1)):
        raise DeepDiveReportContractError(
            "architecture opportunity ranks must be 1..5"
        )
    primary_rows = [row for row in rows if row.get("status") == "primary"]
    if len(primary_rows) > 1 or (
        primary_rows and primary_rows[0].get("rank") != 1
    ):
        raise DeepDiveReportContractError(
            "architecture opportunity ranking allows at most one rank-1 primary"
        )
    for row in rows:
        if row.get("status") not in OPPORTUNITY_STATUSES:
            raise DeepDiveReportContractError(
                f"invalid opportunity status: {row.get('opportunity_id')}"
            )
        _require_text(row, "rationale", "architecture opportunity ranking")
        _require_text(row, "falsification_gate", "architecture opportunity ranking")
        _require_text(row, "innovation_claim", "architecture opportunity ranking")
        _require_text(row, "training_signal", "architecture opportunity ranking")
        _require_text(
            row, "training_data_requirements", "architecture opportunity ranking"
        )
        _require_text(
            row, "exact_null_recovery_invariant", "architecture opportunity ranking"
        )
        _require_string_list(
            row, "required_modules", "architecture opportunity ranking", minimum=1
        )
        _require_string_list(
            row, "critical_ablations", "architecture opportunity ranking", minimum=3
        )
        differences = row.get("prior_work_differences")
        if not isinstance(differences, Mapping) or set(differences) != set(
            PRIOR_WORK_COMPARATORS
        ):
            raise DeepDiveReportContractError(
                "architecture opportunity prior-work comparators differ"
            )
        for comparator in PRIOR_WORK_COMPARATORS:
            _require_text(
                differences,
                comparator,
                "architecture opportunity prior-work differences",
            )
        if row.get("stage_boundary") != OPPORTUNITY_STAGE_BOUNDARY:
            raise DeepDiveReportContractError(
                "architecture opportunity stage boundary differs"
            )
        catalog = OPPORTUNITY_DESIGN_CATALOG[row["opportunity_id"]]
        for field, expected in catalog.items():
            if row.get(field) != expected:
                raise DeepDiveReportContractError(
                    "architecture opportunity design catalog drift: "
                    f"{row['opportunity_id']}:{field}"
                )
        evidence = _evidence_ids(row, admitted, "architecture opportunity ranking")
        if not evidence:
            raise DeepDiveReportContractError(
                f"opportunity lacks admitted evidence: {row['opportunity_id']}"
            )
        _require_relevant_evidence(
            evidence,
            OPPORTUNITY_ALLOWED_DELIVERABLES[row["opportunity_id"]],
            "architecture opportunity ranking",
        )
        model_scope = row.get("model_scope")
        if (
            not isinstance(model_scope, list)
            or not model_scope
            or len(model_scope) != len(set(map(str, model_scope)))
            or any(str(model_id) not in MODEL_IDS for model_id in model_scope)
        ):
            raise DeepDiveReportContractError(
                "architecture opportunity model scope is invalid"
            )
        if not set(map(str, model_scope)).issubset(
            OPPORTUNITY_ALLOWED_MODEL_SCOPE[row["opportunity_id"]]
        ):
            raise DeepDiveReportContractError(
                "architecture opportunity exceeds its preregistered model scope"
            )
        evidence_models = set().union(
            *(DELIVERABLE_MODEL_COVERAGE[item] for item in evidence)
        )
        if not set(map(str, model_scope)).issubset(evidence_models):
            raise DeepDiveReportContractError(
                "architecture opportunity model scope lacks direct evidence"
            )
        if row.get("status") == "primary":
            groups = OPPORTUNITY_PRIMARY_REQUIRED_EVIDENCE_GROUPS[
                row["opportunity_id"]
            ]
            missing_groups = [
                sorted(group) for group in groups if not (set(evidence) & group)
            ]
            if missing_groups:
                raise DeepDiveReportContractError(
                    "primary opportunity lacks its required confirmatory evidence groups"
                )
            per_model_groups = OPPORTUNITY_PRIMARY_PER_MODEL_EVIDENCE_GROUPS.get(
                row["opportunity_id"], ()
            )
            uncovered = [
                (model_id, index)
                for model_id in map(str, model_scope)
                for index, group in enumerate(per_model_groups, start=1)
                if not any(
                    deliverable in evidence
                    and model_id in DELIVERABLE_MODEL_COVERAGE[deliverable]
                    for deliverable in group
                )
            ]
            if uncovered:
                formatted = ",".join(
                    f"{model_id}:group{index}"
                    for model_id, index in uncovered
                )
                raise DeepDiveReportContractError(
                    "primary opportunity has cross-model borrowed evidence: "
                    f"{row['opportunity_id']}:{formatted}"
                )


def _validate_opportunity_hypothesis_consistency(
    hypotheses: Any, opportunities: Any
) -> None:
    hypothesis_status = {
        str(row["hypothesis_id"]): str(row["status"])
        for row in _rows(hypotheses, "hypothesis status matrix")
    }
    opportunity_rows = _rows(opportunities, "architecture opportunity ranking")
    primary_eligible = []
    primary_rows = []
    for row in opportunity_rows:
        linked = OPPORTUNITY_HYPOTHESES[str(row["opportunity_id"])]
        statuses = [hypothesis_status[hypothesis_id] for hypothesis_id in linked]
        if any(status == "rejected" for status in statuses) and row.get(
            "status"
        ) != "rejected":
            raise DeepDiveReportContractError(
                "architecture opportunity contradicts a rejected linked hypothesis"
            )
        if row.get("status") == "primary" and any(
            status not in {"supported", "weakened"} for status in statuses
        ):
            raise DeepDiveReportContractError(
                "primary opportunity requires supported or weakened linked hypotheses"
            )
        evidence = set(map(str, row.get("evidence_deliverables", [])))
        evidence_ready = all(
            evidence & group
            for group in OPPORTUNITY_PRIMARY_REQUIRED_EVIDENCE_GROUPS[
                str(row["opportunity_id"])
            ]
        )
        if all(status in {"supported", "weakened"} for status in statuses) and evidence_ready:
            primary_eligible.append(row)
        if row.get("status") == "primary":
            primary_rows.append(row)
    if not primary_rows:
        if primary_eligible:
            raise DeepDiveReportContractError(
                "architecture opportunity ranking omits an evidence-eligible rank-1 primary"
            )
        rank1 = next(row for row in opportunity_rows if row.get("rank") == 1)
        if rank1.get("status") not in {"deprioritized", "rejected"}:
            raise DeepDiveReportContractError(
                "no-primary architecture ranking requires a deprioritized or rejected rank-1"
            )


def _rows(value: Any, label: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or any(
        not isinstance(row, Mapping) for row in value
    ):
        raise DeepDiveReportContractError(f"{label} must be an object list")
    return list(value)


def _require_exact_ids(
    rows: Sequence[Mapping[str, Any]],
    key: str,
    expected: Sequence[str],
    label: str,
) -> None:
    observed = [str(row.get(key) or "") for row in rows]
    counts = Counter(observed)
    if set(observed) != set(expected) or any(
        count != 1 for count in counts.values()
    ):
        raise DeepDiveReportContractError(f"{label} IDs must appear exactly once")


def _require_text(row: Mapping[str, Any], key: str, label: str) -> None:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DeepDiveReportContractError(f"{label} has empty {key}")


def _require_string_list(
    row: Mapping[str, Any], key: str, label: str, *, minimum: int
) -> None:
    value = row.get(key)
    if (
        not isinstance(value, list)
        or len(value) < minimum
        or len(set(map(str, value))) != len(value)
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise DeepDiveReportContractError(f"{label} has invalid {key}")


def _evidence_ids(
    row: Mapping[str, Any], admitted: set[str], label: str
) -> list[str]:
    evidence = row.get("evidence_deliverables")
    if not isinstance(evidence, list) or len(set(map(str, evidence))) != len(
        evidence
    ):
        raise DeepDiveReportContractError(f"{label} evidence list is invalid")
    result = [str(item) for item in evidence]
    if any(item not in admitted for item in result):
        raise DeepDiveReportContractError(
            f"{label} cites an unadmitted deliverable"
        )
    return result


def _require_relevant_evidence(
    evidence: Sequence[str], allowed: set[str], label: str
) -> None:
    irrelevant = sorted(set(evidence) - allowed)
    if irrelevant:
        raise DeepDiveReportContractError(
            f"{label} cites semantically irrelevant evidence: {irrelevant}"
        )
