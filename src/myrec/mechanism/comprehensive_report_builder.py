"""Fail-closed builder for the final comprehensive Transformer report.

The formal deep-dive report and the supplemental registry have deliberately
different scientific roles.  This builder admits both only after their own
auditors are terminal, then validates a human interpretation worksheet.  It
does not reopen qrels or score bundles and it never derives an architecture
choice from an absolute layer, head, or neuron index.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from myrec.mechanism.comprehensive_readiness import build_comprehensive_readiness
from myrec.mechanism.deep_dive_closeout_audit import EXPECTED_DELIVERABLES
from myrec.mechanism.deep_dive_evidence_topology import MODEL_IDS
from myrec.mechanism.deep_dive_report_builder import REPORT_ANALYSIS_TYPE
from myrec.mechanism.deep_dive_progress import SELECTED_NODES
from myrec.mechanism.deep_dive_report_contract import (
    COMPONENT_ALLOWED_DELIVERABLES,
    COMPONENT_DELIVERABLE_MODEL_COVERAGE,
    COMPONENT_IDS,
    COMPONENT_PROBE_CLAIM_BOUNDARIES,
    COMPONENT_SUPPORT_REQUIRED_CAUSAL_EVIDENCE,
    HYPOTHESIS_ALLOWED_DELIVERABLES,
    HYPOTHESIS_SUPPORTED_COMPONENT_REQUIREMENTS,
    HYPOTHESIS_SUPPORTED_REQUIRED_EVIDENCE_GROUPS,
    OPPORTUNITY_ALLOWED_DELIVERABLES,
    OPPORTUNITY_IDS,
)
from myrec.mechanism.supplemental_evidence_registry import (
    EXPECTED_SUPPLEMENT_IDS,
    audit_supplemental_evidence_registry,
)
from myrec.mechanism.postblock_sweep_evaluator import POSTBLOCK_BLOCKS
from myrec.mechanism.transformer_interface_inventory import (
    build_transformer_interface_coverage,
)
from myrec.utils.hashing import sha256_file


ANALYSIS_TYPE = "transformer_comprehensive_mechanism_report"
COMPREHENSIVE_REPORT_PLAN_IDENTITY = {
    "path": "experiments/motivation/transformer_comprehensive_report_plan.md",
    "sha256": "edbf2c94474e194f91f7856e2a4a57ac33250f671319d4e6d5c0ad880c1e32ab",
}
FROZEN_OBSERVATION_EVIDENCE_IDENTITIES = (
    {
        "evidence_id": "first_round_protocol",
        "evidence_kind": "frozen_observation_source",
        "path": "experiments/motivation/protocol.yaml",
        "sha256": "6788d27cce8186be02dae4595129157fcca5032b49c1107ec83fdd2f9ecf8e43",
    },
    {
        "evidence_id": "first_round_machine_summary",
        "evidence_kind": "frozen_observation_source",
        "path": "reports/motivation_current_summary.json",
        "sha256": "d77e9b2251e75f2d7937dd15f0b685cf68bc50056b985652a0dbedfb43413443",
    },
    {
        "evidence_id": "first_round_results_register",
        "evidence_kind": "frozen_observation_source",
        "path": "experiments/pps_results.md",
        "sha256": "b6dc3773d5f5c8781429533724b724a8930094c737e5b056eba63e182ab23aa5",
    },
    {
        "evidence_id": "first_mechanism_plan",
        "evidence_kind": "frozen_observation_source",
        "path": "experiments/motivation/mechanism_analysis_plan.md",
        "sha256": "60dd7a5e6a5083b7827fdfe7ea1df98b6dc811f159b6f52289d784ff5cb10123",
    },
    {
        "evidence_id": "first_mechanism_probe_manifest",
        "evidence_kind": "frozen_observation_source",
        "path": "experiments/motivation/probe_manifest.yaml",
        "sha256": "adedf0e662b9d8529162b8abffedcf6b10962913f28580af6119d807cc5d929c",
    },
    {
        "evidence_id": "first_mechanism_diagnosis_json",
        "evidence_kind": "frozen_observation_source",
        "path": "reports/motivation_mechanism_first_diagnosis.json",
        "sha256": "f4e225256c461fed012c5953733cd8c0652c59cd7fceeef89658fcbf6ee38383",
    },
    {
        "evidence_id": "first_mechanism_diagnosis_markdown",
        "evidence_kind": "frozen_observation_source",
        "path": "reports/motivation_mechanism_first_diagnosis.md",
        "sha256": "8d7060cf9e480b62c470b34fac80e70a344f0e7a42d41e42ea7302ac85692d0f",
    },
)
EVIDENCE_LEVELS = ("M", "D", "S", "N", "G", "U")
HYPOTHESIS_IDS = tuple(f"H{index}" for index in range(6))
PRIMARY_DESIGN_MODELS = frozenset(MODEL_IDS[2:])
SYSTEM_LAYER_IDS = ("input", "representation", "routing", "readout", "training")
SYSTEM_LAYER_STATUSES = {
    "supported",
    "weakened",
    "unresolved",
    "mechanical_failure",
}
SYSTEM_LAYER_COMPONENTS = {
    "input": {
        "serialization_tokenization",
        "token_embedding",
        "positional_encoding_rope",
    },
    "representation": {
        "mlp_feature_formation",
        "mlp_output",
        "residual_composition",
        "normalization",
        "layerwise_representation",
    },
    "routing": {
        "attention_query_key_routing",
        "attention_value_transport",
        "attention_output",
        "history_routing",
    },
    "readout": {
        "candidate_conditioned_interaction",
        "native_readout",
        "score_calibration_nullspace",
    },
    "training": {
        "loss_gradient",
        "optimizer_effective_update",
        "lora_parameterization",
    },
}
if set(SYSTEM_LAYER_COMPONENTS) != set(SYSTEM_LAYER_IDS) or set().union(
    *SYSTEM_LAYER_COMPONENTS.values()
) != set(COMPONENT_IDS):
    raise RuntimeError("five-system-layer component coverage drift")
CAUSAL_CHAIN_NODES = (
    "incoming_state",
    "attention",
    "mlp",
    "block_output",
    "final_norm",
    "native_score",
)
CAUSAL_CHAIN_STATUSES = {
    "supported",
    "weakened",
    "unresolved",
    "mechanical_failure",
}
CAUSAL_CHAIN_COMPONENTS = {
    "incoming_state": {"layerwise_representation", "history_routing"},
    "attention": {
        "attention_query_key_routing",
        "attention_value_transport",
        "attention_output",
    },
    "mlp": {"mlp_feature_formation", "mlp_output"},
    "block_output": {"residual_composition"},
    "final_norm": {"normalization"},
    "native_score": {"native_readout", "score_calibration_nullspace"},
}
if set(CAUSAL_CHAIN_COMPONENTS) != set(CAUSAL_CHAIN_NODES):
    raise RuntimeError("functional causal-chain component coverage drift")
CAUSAL_CHAIN_CLAIM_BOUNDARIES = {
    "incoming_state": (
        "Support means a history-conditioned state is already present at the incoming "
        "interface; it does not identify its upstream origin or implicate composition "
        "inside the current block."
    ),
    "attention": (
        "Support is limited to the registered routing, value-edge, or output-state "
        "intervention; it does not identify a unique head or establish attention as "
        "the exclusive origin."
    ),
    "mlp": (
        "Support distinguishes a registered MLP output-state mediator from incoming "
        "state; descriptive SwiGLU groups do not establish feature-forming neuron or "
        "operator causality."
    ),
    "block_output": (
        "A complete block-output state can be a behavioral ceiling. Absolute-state "
        "support alone does not establish residual addition, nonlinear interaction, "
        "or an operator-level design target."
    ),
    "final_norm": (
        "Support localizes a pre/post normalization state boundary only when the paired "
        "pre-norm state fails; it does not establish RMSNorm operator necessity."
    ),
    "native_score": (
        "Support is scoped to the frozen native scoring path and candidate-relative "
        "behavior; it does not validate an alternative learned readout or utility gain."
    ),
}
if set(CAUSAL_CHAIN_CLAIM_BOUNDARIES) != set(CAUSAL_CHAIN_NODES):
    raise RuntimeError("functional causal-chain claim-boundary coverage drift")
FAILURE_MODES = (
    "signal_absent_before_candidate_path",
    "localized_state_attenuation",
    "distributed_state_attenuation",
    "candidate_transport_failure",
    "state_present_but_readout_misaligned",
    "objective_update_mismatch",
    "multiple_bottlenecks",
    "unresolved",
)
FAILURE_MODE_REQUIRED_COMPONENTS = {
    "signal_absent_before_candidate_path": {
        "serialization_tokenization",
        "token_embedding",
        "positional_encoding_rope",
        "layerwise_representation",
        "history_routing",
    },
    "localized_state_attenuation": (
        SYSTEM_LAYER_COMPONENTS["representation"]
        | SYSTEM_LAYER_COMPONENTS["routing"]
    ),
    "distributed_state_attenuation": (
        SYSTEM_LAYER_COMPONENTS["representation"]
        | SYSTEM_LAYER_COMPONENTS["routing"]
    ),
    "candidate_transport_failure": (
        SYSTEM_LAYER_COMPONENTS["routing"]
        | {"candidate_conditioned_interaction"}
    ),
    "state_present_but_readout_misaligned": {
        "native_readout",
        "score_calibration_nullspace",
    },
    "objective_update_mismatch": set(SYSTEM_LAYER_COMPONENTS["training"]),
}
if set(FAILURE_MODE_REQUIRED_COMPONENTS) != set(FAILURE_MODES) - {
    "multiple_bottlenecks",
    "unresolved",
}:
    raise RuntimeError("failure-mode component coverage drift")
FAILURE_DIAGNOSTIC_RESOLUTION_BY_LEVEL = {
    "D": "descriptive_candidate",
    "S": "sufficiency_candidate",
    "N": "necessity_mediator_candidate",
    "G": "bidirectionally_supported_failure_path",
}
FAILURE_MODE_CLAIM_BOUNDARY = {
    "registered_signal_erasure_experiment_exists": False,
    "postblock_sufficiency_attenuation_is_signal_erasure": False,
    "component_mediation_is_signal_erasure": False,
    "state_present_but_readout_misaligned_requires_native_score_chain": True,
    "interpretation": (
        "The registered post-block scan measures attenuation of behavioral "
        "sufficiency, and the bidirectional component overlay measures state "
        "mediation. Neither experiment directly measures destruction of semantic "
        "information by an operator, so a causal signal-erasure claim is not "
        "authorized. A causal readout-misalignment claim additionally requires "
        "model-scoped native-readout evidence, a supported native-readout component, "
        "and a supported native-score chain."
    ),
}
NECESSITY_DIRECTION_CLAIM_BOUNDARY = {
    "registered_behavior": "harmful_full_history_target_margin_response",
    "positive_neutral_removal_means_harm_reduction": True,
    "component_is_beneficial_for_transfer_authorized": False,
    "strengthen_or_preserve_component_authorized": False,
    "interpretation": (
        "N/G identifies a state interface that mediates the registered harmful "
        "full-history response. It does not show that the component benefits strict "
        "transfer, nor that a method should strengthen or preserve that state."
    ),
}
CLAIM_INVARIANTS = {
    "layer_scan_is_localization_only": True,
    "exact_layer_head_or_neuron_is_architecture_evidence": False,
    "absolute_layer_index_portability_claimed": False,
    "descriptive_supplement_may_upgrade_confirmatory_claim": False,
    "selected_branch_sufficiency_alone_may_rank_architecture": False,
    "diagnostic_patch_promoted_as_method": False,
    "architecture_implemented": False,
    "opportunity_utility_gain_established": False,
    "cross_dataset_or_model_scale_generalization_claimed": False,
    "behavioral_sufficiency_attenuation_is_signal_erasure": False,
    "component_mediation_is_signal_erasure": False,
    "necessity_support_means_component_is_beneficial": False,
    "design_gate_authorizes_strengthening_the_harmful_state": False,
    "source_test_opened": False,
}
COMPONENT_STATUSES = {
    "supported",
    "weakened",
    "unresolved",
    "untested",
    "mechanical_failure",
}
HYPOTHESIS_STATUSES = {"supported", "weakened", "rejected", "unresolved"}
HYPOTHESIS_NEGATIVE_EVIDENCE_BASES = {
    "registered_refutation",
    "registered_weakening",
    "cross_model_or_endpoint_conflict",
    "measurement_population_instability",
    "insufficient_causal_evidence",
}
NEGATIVE_ENDPOINT_SCOPES = {
    "target_margin",
    "ndcg@10",
    "hidden_state_geometry",
    "attention_structure",
    "mlp_structure",
    "native_score",
    "training_dynamics",
    "mechanical_contract",
}
NEGATIVE_SURFACE_SCOPES = {
    "overall",
    "recurrence",
    "strict_transfer",
    "other_overlap",
    "all_registered_surfaces",
    "not_applicable_qrels_blind",
}
NEGATIVE_CONTRAST_SCOPES = {
    "full_minus_null",
    "full_minus_wrong_user",
    "same_minus_wrong_user",
    "cross_request_stress",
    "structural_controls",
    "not_applicable_descriptive",
}
NEGATIVE_FOLD_SCOPES = {"fold0", "fold1", "full_dev", "not_applicable"}
DESIGN_PRIORITIES = {
    "design_qualified",
    "candidate_to_test",
    "deprioritized",
    "not_recommended",
}
DESIGN_PRIORITY_ORDER = {
    "design_qualified": 0,
    "candidate_to_test": 1,
    "deprioritized": 2,
    "not_recommended": 3,
}
OPPORTUNITY_EVIDENCE_STRENGTH = {
    "G": 4,
    "S": 3,
    "N": 3,
    "D": 2,
    "U": 1,
    "M": 0,
}
INTERVENTION_POLARITIES = {
    "suppress_harmful_state",
    "reroute_history_state",
    "recalibrate_candidate_readout",
    "preserve_or_strengthen_beneficial_state",
    "diagnostic_only",
}
NOT_RECOMMENDED_BASES = {
    "registered_refutation",
    "insufficient_causal_evidence",
    "position_or_measurement_confound",
    "descriptive_only",
    "mechanical_non_result",
}
NOT_RECOMMENDED_BASIS_LEVELS = {
    "registered_refutation": {"S"},
    "insufficient_causal_evidence": {"D", "S", "N", "G", "U"},
    "position_or_measurement_confound": {"M", "D"},
    "descriptive_only": {"D"},
    "mechanical_non_result": {"M"},
}
HARM_MEDIATOR_INTERVENTION_POLARITIES = {
    "suppress_harmful_state",
    "reroute_history_state",
    "recalibrate_candidate_readout",
}
DESIGN_GATE_SUPPLEMENT = "component_functional_design_gate_synthesis"
NECESSITY_SUPPLEMENT = "component_state_reverse_necessity_v2"
DESCRIPTIVE_SUPPLEMENTS = set(EXPECTED_SUPPLEMENT_IDS) - {
    NECESSITY_SUPPLEMENT,
    DESIGN_GATE_SUPPLEMENT,
}
EVIDENCE_DISPOSITIONS = {
    "interpreted_in_findings",
    "negative_or_conflicting",
    "bounded_no_scientific_claim",
}
DESIGN_NODE_COMPONENTS = {
    # A passing incoming-state gate says the harmful state already arrived from
    # upstream.  It can qualify the history-state path, but it cannot be used to
    # attribute the state to composition inside the selected block.
    "block_input_residual": {"history_routing"},
    "attention_o_projection": {"attention_output"},
    "mlp_down_projection": {"mlp_output"},
    # The V2 plan registers block output as a complete-state ceiling.  An
    # absolute-state ceiling does not isolate residual addition or a nonlinear
    # interaction, so it has no direct design-component mapping.
    "block_output_residual": set(),
}
DESIGN_NODE_CLAIM_ROLES = {
    "block_input_residual": "upstream_incoming_state_control",
    "attention_o_projection": "attention_branch_state_mediator",
    "mlp_down_projection": "mlp_branch_state_mediator",
    "block_output_residual": "complete_block_state_ceiling",
}
if set(DESIGN_NODE_CLAIM_ROLES) != set(DESIGN_NODE_COMPONENTS):
    raise RuntimeError("component design-node claim-role coverage drift")
NECESSITY_COMPONENTS = set().union(*DESIGN_NODE_COMPONENTS.values())
FUNCTIONAL_LOCALIZATION_CONTRACT = {
    "observational_question": (
        "At which functional interface does a history-conditioned state, "
        "candidate-relative geometry, or native-score relation change?"
    ),
    "authorized_role": (
        "Bracket pre/post interfaces for registered attention, MLP, residual, "
        "normalization, and readout sufficiency/necessity tests."
    ),
    "cross_model_alignment_unit": (
        "Functional node and signed causal behavior, never an absolute block index."
    ),
    "not_authorized": [
        "operator causality from an observational trajectory",
        "an architecture target from an exact layer number",
        "cross-dataset or model-scale portability from one transition index",
    ],
}
LOCALIZATION_TO_DESIGN_BRIDGE = (
    {
        "stage": "state_localization",
        "question": (
            "Does history-conditioned candidate-relative state or its relation to "
            "the native score change across a bounded functional-depth region?"
        ),
        "required_evidence": (
            "Complete trajectories with absolute and relative state measures, fixed "
            "depth regions, and no best-index outcome selection."
        ),
        "authorized_consequence": (
            "Choose pre/post interfaces for already registered component tests only."
        ),
        "design_authority": False,
    },
    {
        "stage": "component_disambiguation",
        "question": (
            "Is the localized behavior carried by incoming residual state, attention "
            "output, MLP output, or only their composed block state?"
        ),
        "required_evidence": (
            "Same-parent branch interventions with recomposition and identity gates."
        ),
        "authorized_consequence": (
            "Attribute sufficiency to a functional state interface without claiming "
            "operator necessity or a portable internal index."
        ),
        "design_authority": False,
    },
    {
        "stage": "bidirectional_causal_mediation",
        "question": (
            "Is the same functional state both a sufficient carrier and a necessary, "
            "history-specific mediator under position-preserving removal?"
        ),
        "required_evidence": (
            "Forward same-request sufficiency, wrong-user specificity, cross-request "
            "and structural controls, plus reverse neutral-state removal."
        ),
        "authorized_consequence": (
            "Qualify the functional node as a model-local design candidate."
        ),
        "design_authority": False,
    },
    {
        "stage": "cross_model_functional_replication",
        "question": (
            "Does the same signed causal behavior recur at the same functional node "
            "in both frozen model pathways?"
        ),
        "required_evidence": (
            "All bidirectional gates pass independently in Q2 and Q3 while exact "
            "internal indices remain lineage metadata."
        ),
        "authorized_consequence": (
            "Permit the node, but never its absolute index, to change architecture "
            "opportunity ranking within the frozen dataset and model scope."
        ),
        "design_authority": True,
    },
)
COMPONENT_FUNCTIONAL_QUESTIONS = {
    "serialization_tokenization": (
        "Do serialization, truncation, history length, or shifted semantic positions "
        "create an input-side transfer confound before model computation?"
    ),
    "token_embedding": (
        "Are history, query, candidate, or native readout token directions absent, "
        "frozen, or update-starved at the embedding interface?"
    ),
    "positional_encoding_rope": (
        "Does relative phase geometry, rather than history content, impair query-to-history "
        "or history-to-readout interaction?"
    ),
    "attention_query_key_routing": (
        "Does attention route the relevant query-conditioned history to native candidate "
        "readout positions, under position-preserving controls?"
    ),
    "attention_value_transport": (
        "Does the history value path causally transport content after routing, rather than "
        "merely exhibit attention mass?"
    ),
    "attention_output": (
        "Is the attention output state a history-specific sufficient and necessary carrier "
        "at a functionally localized transition?"
    ),
    "mlp_feature_formation": (
        "Do SwiGLU gate/up interactions form history-sensitive or candidate-relative "
        "features beyond generic activation concentration?"
    ),
    "mlp_output": (
        "Is the MLP down-projection state a history-specific sufficient and necessary "
        "carrier, independently of incoming residual state?"
    ),
    "residual_composition": (
        "Does the behavioral change require the composed residual state when attention and "
        "MLP branches alone are insufficient, and is it already present on input?"
    ),
    "normalization": (
        "Does RMS normalization selectively redirect candidate-relative history state, or "
        "only rescale common and relative components together?"
    ),
    "layerwise_representation": (
        "Where are history-conditioned and candidate-relative states formed, rewritten, "
        "or distributed across functional depth, without treating an index as a method?"
    ),
    "history_routing": (
        "Is the transported state tied to the same user's history rather than a generic, "
        "wrong-user, or cross-request history response?"
    ),
    "candidate_conditioned_interaction": (
        "Does history create signed differences among candidates, rather than primarily a "
        "candidate-common hidden displacement?"
    ),
    "native_readout": (
        "Does the model's actual scoring path use an available history-conditioned state "
        "with the correct candidate-relative sign and calibration?"
    ),
    "score_calibration_nullspace": (
        "How much history response lies in exact scalar-score rank-null directions versus "
        "candidate-relative score changes?"
    ),
    "loss_gradient": (
        "Do recurrence, strict-transfer, and overlap objectives allocate conflicting or "
        "misdirected gradients after label-shuffle controls?"
    ),
    "optimizer_effective_update": (
        "Do raw gradients survive optimizer state, scheduling, and parameter scaling as the "
        "effective update actually applied to the model?"
    ),
    "lora_parameterization": (
        "Does the Q/V-only low-rank adaptation path restrict routing, value transport, or "
        "readout rotation relative to the fully updated model?"
    ),
}
if set(COMPONENT_FUNCTIONAL_QUESTIONS) != set(COMPONENT_IDS):
    raise RuntimeError("component functional-question coverage drift")
REQUIRED_NARRATIVES = (
    "executive_summary",
    "frozen_observation_and_scope",
    "layer_trajectory_interpretation",
    "cross_model_boundary",
    "paper_claim_boundary",
)
EXECUTION_AXIS_CENSUS = {
    "models": list(MODEL_IDS),
    "registered_endpoints": [
        {
            "endpoint": "target_margin",
            "role": "primary_candidate_relative_mechanism_endpoint",
        },
        {
            "endpoint": "ndcg@10",
            "role": "secondary_shared_utility_endpoint",
        },
    ],
    "normalized_query_folds": [
        {
            "fold": 0,
            "role": "registered_localization_or_discovery_only",
        },
        {
            "fold": 1,
            "role": "fixed_transition_confirmation",
        },
    ],
    "boundary": (
        "Descriptive supplements may be qrels-blind and have no registered endpoint; "
        "fold 0 selects only where explicitly preregistered, while fold 1 confirms "
        "the fixed transition and cannot reselect it."
    ),
}
HISTORY_SIGNAL_OBSERVATION_SCOPE_CONTRACT = (
    {
        "scope_id": "serialized_history_content_span",
        "evidence_ids": ["d3_attention_heads", "d5_context"],
        "model_scope": [MODEL_IDS[2], MODEL_IDS[3]],
        "sequence_or_token_scope": (
            "The exact contiguous token span covering all retained serialized history "
            "content, with query and candidate spans resolved separately."
        ),
        "granularity": (
            "Span-level attention summaries plus position-preserving content or key-span "
            "controls; no outcome-selected event or token."
        ),
        "question_answered": (
            "Whether the retained history span is attended to or causally changes scoring "
            "when content or visibility is changed without changing the candidate slate."
        ),
        "not_observed": (
            "No individual history event is assigned a causal contribution, and a whole-"
            "span effect does not identify which event supplied transferable preference."
        ),
    },
    {
        "scope_id": "attention_history_edges",
        "evidence_ids": ["d3_attention_edges", "d3_attention_heads"],
        "model_scope": [MODEL_IDS[2], MODEL_IDS[3]],
        "sequence_or_token_scope": (
            "History-summary and native-readout query rows crossed with exact query, "
            "history, and candidate key/value spans."
        ),
        "granularity": (
            "All registered heads and GQA groups on the frozen 512-row sample, with "
            "span-edge mass, contribution, geometry, and fixed edge interventions."
        ),
        "question_answered": (
            "Whether routing mass and transported value contribution use the history span "
            "at summary or native-readout queries."
        ),
        "not_observed": (
            "Attention mass is not preference content, span aggregation is not per-event "
            "attribution, and edge removal does not isolate the softmax operator itself."
        ),
    },
    {
        "scope_id": "layer_state_trajectory_endpoints",
        "evidence_ids": ["d1_representation", "d2_postblock"],
        "model_scope": [MODEL_IDS[2], MODEL_IDS[3]],
        "sequence_or_token_scope": (
            "query_end, history_summary_end, and every model-native candidate readout "
            "position across all 29 residual states."
        ),
        "granularity": (
            "Full internal-dev trajectories and fixed depth-region summaries; absolute "
            "block indices remain lineage only."
        ),
        "question_answered": (
            "Where a history-conditioned carrier or candidate-relative state changes "
            "between registered functional interfaces."
        ),
        "not_observed": (
            "history_summary_end is an endpoint carrier, not a tokenwise trace of every "
            "history event; localization alone does not identify a causal component."
        ),
    },
    {
        "scope_id": "component_state_mediation_at_native_readout",
        "evidence_ids": [
            "d2_selected_branches",
            "component_state_reverse_necessity_v2",
        ],
        "model_scope": [MODEL_IDS[2], MODEL_IDS[3]],
        "sequence_or_token_scope": (
            "All native scoring positions at the fixed selected transition, separately "
            "for incoming residual, attention o-proj, MLP down-proj, and block output."
        ),
        "granularity": (
            "Same-request sufficiency, wrong-history specificity, structural controls, "
            "and position-preserving neutral-state removal."
        ),
        "question_answered": (
            "Which functional state is sufficient or necessary for the registered signed "
            "score response after a transition has been independently localized."
        ),
        "not_observed": (
            "A native-readout state intervention does not show which history event formed "
            "that state or establish the internal operator that produced it."
        ),
    },
    {
        "scope_id": "swiglu_feature_formation_endpoints",
        "evidence_ids": [
            "d4_mlp_groups",
            "d4_mlp_feature_formation_extension",
        ],
        "model_scope": [MODEL_IDS[2], MODEL_IDS[3]],
        "sequence_or_token_scope": (
            "query_end, history_summary_end, and every native readout position at the "
            "fixed blocks 13, 20, and 27."
        ),
        "granularity": (
            "Frozen 512-row sample, sixteen fixed dimension groups, and exact gate-pre, "
            "SiLU-gate, up-proj, and product delta algebra."
        ),
        "question_answered": (
            "Whether history-conditioned changes are formed mainly through gate, up, or "
            "their interaction at registered semantic endpoints."
        ),
        "not_observed": (
            "This is not a per-history-event MLP trajectory, neuron selection, additive "
            "causal attribution, or proof that the SwiGLU operator is necessary."
        ),
    },
    {
        "scope_id": "position_and_rope_span_controls",
        "evidence_ids": ["d5_context", "d5_rope"],
        "model_scope": [MODEL_IDS[2], MODEL_IDS[3]],
        "sequence_or_token_scope": (
            "The complete retained history-content span and registered native-readout Q "
            "positions, with token length, masks, and non-target positions held fixed."
        ),
        "granularity": (
            "Full-span content neutralization or attention-null plus fixed compression/"
            "expansion phase interventions at blocks 13, 20, and 27."
        ),
        "question_answered": (
            "Whether history content visibility or relative rotary phase, rather than a "
            "generic length/position shift, changes the registered endpoints."
        ),
        "not_observed": (
            "A span or phase intervention does not identify a semantic event selector or "
            "authorize a natural-sequence position rewrite."
        ),
    },
    {
        "scope_id": "native_score_readout",
        "evidence_ids": [
            "d6_q0_q1_readouts",
            "d6_q2_native_readout",
            "d6_q3_native_readout",
        ],
        "model_scope": list(MODEL_IDS),
        "sequence_or_token_scope": (
            "Each frozen model's actual scoring positions and score algebra, including "
            "Q1 continuation phases and every Q3 Yes/No target position."
        ),
        "granularity": (
            "Model-native final-state, final-RMSNorm, and score-path interventions rather "
            "than a shared surrogate logit lens."
        ),
        "question_answered": (
            "Whether an available history-conditioned state is used with a candidate-"
            "relative sign by the scoring path that produces the frozen ranking."
        ),
        "not_observed": (
            "Readout mediation does not localize the upstream history token, attention "
            "edge, or MLP feature that created the state."
        ),
    },
    {
        "scope_id": "q0_q1_model_specific_sequence_breadth",
        "evidence_ids": ["d6_q0_trajectory", "d6_q1_trajectory"],
        "model_scope": [MODEL_IDS[0], MODEL_IDS[1]],
        "sequence_or_token_scope": (
            "Q0 specialized pointwise trajectory and Q1 prefix-cache/continuation phases "
            "under their own frozen serialization and native scoring semantics."
        ),
        "granularity": (
            "Model-specific breadth controls used to test portability of a functional "
            "claim without forcing Q2/Q3 token-position semantics onto Q0/Q1."
        ),
        "question_answered": (
            "Whether a functional state/readout pattern recurs outside the two primary "
            "mechanism anchors under the correct model pathway."
        ),
        "not_observed": (
            "These pathways are not positionwise interchangeable with Q2/Q3 and cannot "
            "upgrade a Q2/Q3 component claim by analogy alone."
        ),
    },
)
_history_scope_ids = [row["scope_id"] for row in HISTORY_SIGNAL_OBSERVATION_SCOPE_CONTRACT]
_history_scope_evidence = set().union(
    *(set(row["evidence_ids"]) for row in HISTORY_SIGNAL_OBSERVATION_SCOPE_CONTRACT)
)
if (
    len(_history_scope_ids) != len(set(_history_scope_ids))
    or not _history_scope_evidence.issubset(
        set(EXPECTED_DELIVERABLES) | set(EXPECTED_SUPPLEMENT_IDS)
    )
    or any(
        not row["evidence_ids"]
        or not row["model_scope"]
        or not set(row["model_scope"]).issubset(MODEL_IDS)
        or any(
            not isinstance(row[field], str) or not row[field].strip()
            for field in (
                "scope_id",
                "sequence_or_token_scope",
                "granularity",
                "question_answered",
                "not_observed",
            )
        )
        for row in HISTORY_SIGNAL_OBSERVATION_SCOPE_CONTRACT
    )
):
    raise RuntimeError("history-signal observation-scope contract drift")
FROZEN_OBSERVATION_SCOPE_CONTRACT = (
    {
        "scope_id": "frozen_observation",
        "definition": (
            "The frozen single-seed retrospective KuaiSearch confirmation shows "
            "recurrence-dominant history use across Q0--Q3; reliable strict-transfer "
            "gain was not established."
        ),
        "boundary": (
            "This is the phenomenon to explain, not evidence that transferable "
            "preference information is absent in principle."
        ),
    },
    {
        "scope_id": "recurrence_surface",
        "definition": (
            "target-repeat: the current positive item occurs in user history."
        ),
        "boundary": (
            "A positive response can use exact-item identity or near-duplicate text and "
            "does not establish cross-item preference transfer."
        ),
    },
    {
        "scope_id": "strict_transfer_surface",
        "definition": (
            "target-nonrepeat/no-candidate-overlap: neither the positive target nor any "
            "candidate item ID overlaps user history."
        ),
        "boundary": (
            "A confidence interval crossing zero means gain was not established; it is "
            "not proof that a transferable signal does not exist."
        ),
    },
    {
        "scope_id": "other_overlap_surface",
        "definition": (
            "target-nonrepeat/other-candidate-overlap: the positive target is new but "
            "another candidate overlaps user history."
        ),
        "boundary": (
            "This surface is reported separately because recurrence can favor a "
            "competitor and cannot be pooled into strict transfer."
        ),
    },
    {
        "scope_id": "full_null_wrong_user",
        "definition": (
            "Full history is compared with null history and with a frozen wrong-user "
            "assignment on the same request and candidate set."
        ),
        "boundary": (
            "Null isolates the history response. Wrong-user is a specificity diagnostic, "
            "not standalone proof of user-specific causality because fallback assignments "
            "remain in the frozen control."
        ),
    },
    {
        "scope_id": "population_and_model_scope",
        "definition": (
            "Claims are limited to the frozen Q0--Q3 variants, one training seed, and "
            "the retrospective KuaiSearch development/confirmation boundary."
        ),
        "boundary": (
            "No official-method, forward-temporal, cross-dataset, model-scale, or "
            "universal LLM4Rec generalization is authorized."
        ),
    },
)
PAPER_METHOD_STAGE_REQUIREMENTS = (
    {
        "requirement_id": "functional_causal_target",
        "evidence_needed": (
            "A Q2/Q3-replicated G-level functional node or an explicitly bounded "
            "model-specific causal target, with S/N/specificity and structural controls."
        ),
        "current_stage_boundary": (
            "The mechanism report may rank a target; it may not substitute a layer "
            "index or a one-way diagnostic patch for a method."
        ),
    },
    {
        "requirement_id": "independent_method_instantiation",
        "evidence_needed": (
            "A project-owned trainable architecture with an exact no-op path, explicit "
            "mechanism interface, and diagnostic components independently ablatable."
        ),
        "current_stage_boundary": "No transfer architecture is implemented in this stage.",
    },
    {
        "requirement_id": "preregistered_utility",
        "evidence_needed": (
            "Frozen primary ranking utility, strict-transfer, recurrence, overlap, and "
            "adverse-surface endpoints with seed and selection rules fixed in advance."
        ),
        "current_stage_boundary": (
            "Target-margin mediation and practical equivalence are not positive NDCG gain."
        ),
    },
    {
        "requirement_id": "replication_and_generalization",
        "evidence_needed": (
            "Independent training seeds and, after a new authorization/protocol, a "
            "forward or source-test evaluation and any claimed cross-model/data scope."
        ),
        "current_stage_boundary": (
            "The present single-seed retrospective KuaiSearch scope cannot establish "
            "generalization. Source test remains closed."
        ),
    },
    {
        "requirement_id": "baselines_ablations_efficiency",
        "evidence_needed": (
            "Strong matched baselines, component and training-signal ablations, negative "
            "controls, uncertainty, parameter/latency/memory cost, and failure analysis."
        ),
        "current_stage_boundary": (
            "The current report supplies mechanism falsification targets, not a paper "
            "method comparison table."
        ),
    },
)
REPORT_SECTION_CONTRACT = (
    {
        "section_id": "execution_and_evidence",
        "required_payload_paths": (
            "formal_execution_census",
            "evidence_admission.readiness",
            "execution_axis_census",
            "evidence_disposition",
            "frozen_model_architecture_audit",
        ),
    },
    {
        "section_id": "frozen_observation_and_scope",
        "required_payload_paths": (
            "narratives.frozen_observation_and_scope",
            "frozen_observation_scope_contract",
            "frozen_observation_evidence",
            "frozen_observation_machine_snapshot",
        ),
    },
    {
        "section_id": "layer_trajectory_without_index_design",
        "required_payload_paths": (
            "formal_layerwise_attenuation_profile",
            "formal_attenuation_transition_profile",
        ),
    },
    {
        "section_id": "component_18_by_4_matrix",
        "required_payload_paths": (
            "component_matrix",
            "component_model_matrix",
            "component_evidence_role_coverage",
            "transformer_internal_interface_coverage",
            "history_signal_observation_scope_contract",
        ),
    },
    {
        "section_id": "functional_causal_chain",
        "required_payload_paths": (
            "functional_causal_chain",
            "component_bidirectional_gate_matrix",
            "failure_mode_diagnosis",
        ),
    },
    {
        "section_id": "five_system_layers",
        "required_payload_paths": ("system_layers",),
    },
    {
        "section_id": "q0_q3_cross_model_boundaries",
        "required_payload_paths": ("model_boundaries", "cross_model_synthesis"),
    },
    {
        "section_id": "h0_h5_matrix",
        "required_payload_paths": (
            "prior_mechanism_diagnosis_snapshot",
            "hypothesis_matrix",
        ),
    },
    {
        "section_id": "negative_conflicting_and_mechanical_results",
        "required_payload_paths": (
            "negative_and_conflicting_results",
            "evidence_admission.readiness.mechanical_nonresults",
        ),
    },
    {
        "section_id": "optimization_opportunity_ranking",
        "required_payload_paths": (
            "prior_mechanism_diagnosis_snapshot.architecture_opportunity_matrix",
            "formal_architecture_opportunity_ranking",
            "formal_opportunity_disposition",
            "opportunity_lineage_matrix",
            "optimization_opportunities",
        ),
    },
    {
        "section_id": "not_recommended_directions",
        "required_payload_paths": ("not_recommended",),
    },
    {
        "section_id": "paper_claim_boundary",
        "required_payload_paths": (
            "claim_invariants",
            "paper_method_stage_requirements",
        ),
    },
    {
        "section_id": "reproducibility_appendix",
        "required_payload_paths": ("reproducibility_ledger",),
    },
)
if len(REPORT_SECTION_CONTRACT) != 13 or len(
    {row["section_id"] for row in REPORT_SECTION_CONTRACT}
) != 13:
    raise RuntimeError("comprehensive report section coverage drift")

# Exact indices are valid lineage in the layer-trajectory section, but cannot
# become a design-facing opportunity identifier, target, or rationale.
_ABSOLUTE_INTERNAL_INDEX = re.compile(
    r"(?i)(?:layer|block|head|neuron)\s*[-_:#]?\s*\d+"
)
_FREE_TEXT_METRIC_LITERAL = re.compile(
    r"(?i)\b(?:ndcg(?:@10)?|target[-_ ]?margin|ci(?:95)?|p|q|effect(?: size)?)"
    r"\s*(?:=|:)\s*[-+]?(?:\d+(?:\.\d+)?|\.\d+)"
)
HUMAN_INTERPRETATION_TEXT_FIELDS = {
    "title",
    "claim",
    "text",
    "summary",
    "remaining_uncertainty",
    "diagnosis",
    "reason_remaining",
    "do_not_infer",
    "contradictory_evidence",
    "do_not_generalize",
    "interpretation_boundary",
    "mechanism_target",
    "reason",
    "expected_benefit",
    "hypothesized_innovation",
    "training_signal_requirements",
    "key_ablations",
    "closest_baseline_families",
    "baseline_differentiation",
    "key_risks",
    "falsification_gate",
    "direction",
}


def build_frozen_observation_snapshot(root: str | Path) -> dict[str, Any]:
    """Load the frozen machine summary without reopening qrels or score bundles."""

    root_path = Path(root).resolve()
    summary_identity = next(
        identity
        for identity in FROZEN_OBSERVATION_EVIDENCE_IDENTITIES
        if identity["evidence_id"] == "first_round_machine_summary"
    )
    _validate_repository_evidence_identity(root_path, summary_identity)
    summary_path = root_path / str(summary_identity["path"])
    summary = _load_json(summary_path)
    if (
        summary.get("schema_version") != 1
        or summary.get("report_id") != "pps_motivation_v12_first_round_single_seed"
        or summary.get("status")
        != "first_round_single_seed_complete_preliminary"
    ):
        raise ValueError("frozen first-round summary header differs")

    boundary = _require_mapping(
        summary.get("evidence_boundary"), "frozen first-round evidence boundary"
    )
    if (
        boundary.get("dataset_id") != "kuaisearch"
        or boundary.get("pilot_seed") != 20260714
        or boundary.get("second_seed_run") is not False
        or boundary.get("metric") != "graded_ndcg_at_10"
        or boundary.get("primary_counterfactual") != "full_minus_null"
        or boundary.get("shared_evaluator_only") is not True
        or boundary.get("source_test_opened") is not False
    ):
        raise ValueError("frozen first-round evidence boundary differs")
    bootstrap = _require_mapping(
        boundary.get("bootstrap"), "frozen first-round bootstrap"
    )
    if bootstrap != {
        "cluster": "normalized_query",
        "samples": 5000,
        "seed": 20260715,
    }:
        raise ValueError("frozen first-round bootstrap contract differs")

    counts = _exact_mapping(
        summary,
        "new_holdout_surface_counts",
        (
            "all",
            "recurrence_target_repeat",
            "strict_transfer_target_nonrepeat_no_candidate_overlap",
            "other_overlap_target_nonrepeat_other_candidate_overlap",
            "target_nonrepeat_no_history",
            "no_observed_positive",
        ),
    )
    if any(type(value) is not int or value < 0 for value in counts.values()):
        raise ValueError("frozen first-round surface count is invalid")
    if counts["all"] != sum(
        value for key, value in counts.items() if key != "all"
    ):
        raise ValueError("frozen first-round surface counts do not reconstruct all")

    raw_results = _exact_mapping(
        summary, "new_holdout_primary_results", MODEL_IDS
    )
    surfaces = ("overall", "recurrence", "strict_transfer", "other_overlap")
    contrasts = ("full_minus_null", "full_minus_wrong_user")
    methods = []
    for method_id in MODEL_IDS:
        result = _require_mapping(
            raw_results[method_id], f"frozen first-round result {method_id}"
        )
        method = {
            "method_id": method_id,
            "analysis_run_id": _nonempty_string(
                result.get("analysis_run_id"), f"{method_id} analysis_run_id"
            ),
            "full_ndcg_at_10": _finite_float(
                result.get("full_ndcg_at_10"), f"{method_id} full_ndcg_at_10"
            ),
        }
        for contrast in contrasts:
            raw_contrast = _exact_mapping(result, contrast, surfaces)
            normalized_surfaces = {}
            for surface in surfaces:
                raw_cell = _require_mapping(
                    raw_contrast[surface], f"{method_id}.{contrast}.{surface}"
                )
                interval = raw_cell.get("query_cluster_ci95")
                if not isinstance(interval, list) or len(interval) != 2:
                    raise ValueError(
                        f"frozen first-round interval differs: "
                        f"{method_id}.{contrast}.{surface}"
                    )
                lower = _finite_float(
                    interval[0], f"{method_id}.{contrast}.{surface}.ci95[0]"
                )
                upper = _finite_float(
                    interval[1], f"{method_id}.{contrast}.{surface}.ci95[1]"
                )
                if lower > upper:
                    raise ValueError("frozen first-round interval is reversed")
                normalized_surfaces[surface] = {
                    "mean": _finite_float(
                        raw_cell.get("mean"),
                        f"{method_id}.{contrast}.{surface}.mean",
                    ),
                    "query_cluster_ci95": [lower, upper],
                }
            method[contrast] = normalized_surfaces

        raw_contribution = _exact_mapping(
            result,
            "full_minus_null_population_weighted_contribution",
            ("recurrence", "strict_transfer", "other_overlap"),
        )
        method["full_minus_null_population_weighted_contribution"] = {
            key: _finite_float(value, f"{method_id}.contribution.{key}")
            for key, value in raw_contribution.items()
        }
        evidence_identity = {
            "evidence_id": method_id,
            "evidence_kind": "frozen_first_round_evaluator_output",
            "path": _nonempty_string(
                result.get("evidence_path"), f"{method_id} evidence_path"
            ),
            "sha256": _nonempty_string(
                result.get("evidence_sha256"), f"{method_id} evidence_sha256"
            ),
        }
        _validate_repository_evidence_identity(root_path, evidence_identity)
        evidence_payload = _load_json(
            root_path / str(evidence_identity["path"])
        )
        if (
            evidence_payload.get("analysis_run_id") != method["analysis_run_id"]
            or evidence_payload.get("analysis_type")
            != "motivation_v12_shared_evaluator_evidence"
            or evidence_payload.get("metric_source")
            != "myrec.eval.history_response_evaluator"
            or evidence_payload.get("split") != "confirmation"
            or evidence_payload.get("label_mode") != "graded"
        ):
            raise ValueError(
                f"frozen shared-evaluator evidence header differs: {method_id}"
            )
        raw_artifacts = _exact_mapping(
            evidence_payload,
            "shared_evaluator_artifacts",
            ("metadata", "metrics", "per_request", "target_aware_surfaces"),
        )
        evaluator_artifacts = []
        for artifact_id, raw_identity in raw_artifacts.items():
            identity = _require_mapping(
                raw_identity, f"{method_id} evaluator artifact {artifact_id}"
            )
            normalized_identity = _normalized_evidence_identity(
                evidence_id=f"{method_id}:{artifact_id}",
                evidence_kind="frozen_first_round_evaluator_artifact",
                identity=identity,
            )
            _validate_repository_evidence_identity(root_path, normalized_identity)
            evaluator_artifacts.append(normalized_identity)
        metrics_sha256 = _nonempty_string(
            result.get("metrics_sha256"), f"{method_id} metrics_sha256"
        )
        metrics_identity = next(
            identity
            for identity in evaluator_artifacts
            if identity["evidence_id"] == f"{method_id}:metrics"
        )
        if metrics_identity["sha256"] != metrics_sha256:
            raise ValueError(
                f"frozen machine summary metrics identity differs: {method_id}"
            )

        integrity = _require_mapping(
            evidence_payload.get("integrity"), f"{method_id} evaluator integrity"
        )
        if (
            integrity.get("all_request_partition_reconstructs_overall") is not True
            or integrity.get("method_owned_metrics") is not False
            or integrity.get("metric_formulas_changed") is not False
            or integrity.get("shared_evaluator_qrels_read") is not True
        ):
            raise ValueError(f"frozen shared-evaluator integrity differs: {method_id}")
        pre_qrels = _require_mapping(
            integrity.get("pre_qrels_score_bundle_audit"),
            f"{method_id} pre-qrels audit",
        )
        if pre_qrels.get("passed") is not True or pre_qrels.get("qrels_read") is not False:
            raise ValueError(f"frozen first-round pre-qrels audit failed: {method_id}")
        pre_qrels_identity = _normalized_evidence_identity(
            evidence_id=f"{method_id}:pre_qrels_score_bundle_audit",
            evidence_kind="frozen_first_round_integrity_artifact",
            identity=pre_qrels,
        )
        _validate_repository_evidence_identity(root_path, pre_qrels_identity)

        qrels_lock = _require_mapping(
            integrity.get("qrels_hash_lock"), f"{method_id} qrels hash lock"
        )
        if qrels_lock.get("verified_before_shared_evaluator") is not True:
            raise ValueError(f"frozen first-round qrels lock differs: {method_id}")
        qrels_lock_identity = _normalized_evidence_identity(
            evidence_id=f"{method_id}:qrels_hash_lock",
            evidence_kind="frozen_first_round_integrity_artifact",
            identity=qrels_lock,
        )
        _validate_repository_evidence_identity(root_path, qrels_lock_identity)

        records = _require_mapping(
            evidence_payload.get("records"), f"{method_id} confirmation records"
        )
        if (
            records.get("request_count") != counts["all"]
            or records.get("split") != "confirmation"
        ):
            raise ValueError(f"frozen first-round record boundary differs: {method_id}")
        records_identity = _normalized_evidence_identity(
            evidence_id=f"{method_id}:confirmation_records",
            evidence_kind="frozen_first_round_input_artifact",
            identity=records,
        )
        _validate_repository_evidence_identity(root_path, records_identity)

        method["evaluator_evidence"] = {
            **evidence_identity,
            "metrics_sha256": metrics_sha256,
            "pre_qrels_audit_passed": result.get("pre_qrels_audit_passed")
            is True,
            "shared_evaluator_artifacts": evaluator_artifacts,
            "pre_qrels_score_bundle_audit": pre_qrels_identity,
            "qrels_hash_lock": qrels_lock_identity,
            "confirmation_records": records_identity,
        }
        if method["evaluator_evidence"]["pre_qrels_audit_passed"] is not True:
            raise ValueError(f"frozen first-round pre-qrels audit failed: {method_id}")
        methods.append(method)

    def all_intervals(predicate: Any, *, contrast: str, surface: str) -> bool:
        return all(
            predicate(method[contrast][surface]["query_cluster_ci95"])
            for method in methods
        )

    claim_checks = {
        "all_full_minus_null_overall_ci_positive": all_intervals(
            lambda interval: interval[0] > 0,
            contrast="full_minus_null",
            surface="overall",
        ),
        "all_full_minus_wrong_user_overall_ci_positive": all_intervals(
            lambda interval: interval[0] > 0,
            contrast="full_minus_wrong_user",
            surface="overall",
        ),
        "all_full_minus_null_recurrence_ci_positive": all_intervals(
            lambda interval: interval[0] > 0,
            contrast="full_minus_null",
            surface="recurrence",
        ),
        "all_full_minus_wrong_user_recurrence_ci_positive": all_intervals(
            lambda interval: interval[0] > 0,
            contrast="full_minus_wrong_user",
            surface="recurrence",
        ),
        "all_full_minus_null_strict_transfer_ci_cross_zero": all_intervals(
            lambda interval: interval[0] <= 0 <= interval[1],
            contrast="full_minus_null",
            surface="strict_transfer",
        ),
        "all_full_minus_wrong_user_strict_transfer_ci_cross_zero": all_intervals(
            lambda interval: interval[0] <= 0 <= interval[1],
            contrast="full_minus_wrong_user",
            surface="strict_transfer",
        ),
        "all_full_minus_null_other_overlap_ci_negative": all_intervals(
            lambda interval: interval[1] < 0,
            contrast="full_minus_null",
            surface="other_overlap",
        ),
        "all_recurrence_contributions_exceed_strict_transfer": all(
            method["full_minus_null_population_weighted_contribution"]["recurrence"]
            > method["full_minus_null_population_weighted_contribution"][
                "strict_transfer"
            ]
            for method in methods
        ),
    }
    if not all(claim_checks.values()):
        raise ValueError("frozen qualitative observation differs from machine summary")

    return {
        "source": dict(summary_identity),
        "report_id": summary["report_id"],
        "status": summary["status"],
        "dataset_id": boundary["dataset_id"],
        "dataset_version": _nonempty_string(
            boundary.get("dataset_version"), "frozen first-round dataset_version"
        ),
        "pilot_seed": boundary["pilot_seed"],
        "second_seed_run": boundary["second_seed_run"],
        "bootstrap": dict(bootstrap),
        "metric": boundary["metric"],
        "primary_counterfactual": boundary["primary_counterfactual"],
        "surface_counts": dict(counts),
        "methods": methods,
        "claim_checks": claim_checks,
        "source_test_opened": False,
        "qrels_or_score_bundles_opened_by_this_snapshot": False,
        "interpretation_boundary": (
            "These values are copied from the frozen shared-evaluator summary. They "
            "establish a preliminary single-seed retrospective recurrence-dominant "
            "observation, not absence of transferable signal, official-method failure, "
            "multi-seed robustness, or forward-temporal generalization."
        ),
    }


def build_prior_mechanism_diagnosis_snapshot(root: str | Path) -> dict[str, Any]:
    """Retain the frozen M0--M3 H0--H5 state and all source byte identities."""

    root_path = Path(root).resolve()
    diagnosis_identity = next(
        identity
        for identity in FROZEN_OBSERVATION_EVIDENCE_IDENTITIES
        if identity["evidence_id"] == "first_mechanism_diagnosis_json"
    )
    _validate_repository_evidence_identity(root_path, diagnosis_identity)
    diagnosis = _load_json(root_path / str(diagnosis_identity["path"]))
    if (
        diagnosis.get("schema_version") != 1
        or diagnosis.get("report_id")
        != "pps_motivation_mechanism_first_diagnosis_v1"
        or diagnosis.get("status") != "first_mechanism_diagnosis_complete"
    ):
        raise ValueError("frozen first mechanism diagnosis header differs")
    scope = _require_mapping(
        diagnosis.get("scope_and_boundaries"), "prior mechanism scope"
    )
    if (
        scope.get("dataset_id") != "kuaisearch"
        or scope.get("evaluation_population") != "internal_dev_only"
        or scope.get("source_test_opened") is not False
        or scope.get("new_dataset_opened") is not False
        or scope.get("proposed_architecture_implementation_authorized") is not False
        or scope.get("diagnostic_training_control_role")
        != "diagnostic_control_not_paper_method"
    ):
        raise ValueError("frozen first mechanism diagnosis scope differs")

    raw_registry = _require_mapping(
        diagnosis.get("artifact_registry"), "prior mechanism artifact registry"
    )
    if len(raw_registry) != 18:
        raise ValueError("prior mechanism artifact registry coverage differs")
    artifact_registry = []
    for evidence_id, raw_identity in sorted(raw_registry.items()):
        identity = _require_mapping(
            raw_identity, f"prior mechanism artifact {evidence_id}"
        )
        if identity.get("copied_verbatim_without_statistical_recomputation") is not True:
            raise ValueError(
                f"prior mechanism artifact was not copied verbatim: {evidence_id}"
            )
        normalized_identity = _normalized_evidence_identity(
            evidence_id=str(evidence_id),
            evidence_kind="frozen_prior_mechanism_artifact",
            identity=identity,
        )
        _validate_repository_evidence_identity(root_path, normalized_identity)
        artifact_registry.append(
            {
                **normalized_identity,
                "stage": _nonempty_string(
                    identity.get("stage"), f"{evidence_id} stage"
                ),
                "kind": _nonempty_string(
                    identity.get("kind"), f"{evidence_id} kind"
                ),
                "run_id": _nonempty_string(
                    identity.get("run_id"), f"{evidence_id} run_id"
                ),
                "copied_verbatim_without_statistical_recomputation": True,
            }
        )

    artifact_ids = {row["evidence_id"] for row in artifact_registry}
    raw_evidence_index = diagnosis.get("evidence_index")
    if not isinstance(raw_evidence_index, list) or len(raw_evidence_index) != 11:
        raise ValueError("prior mechanism logical evidence coverage differs")
    evidence_index = []
    logical_evidence_ids = set()
    for raw_row in raw_evidence_index:
        row = _require_mapping(raw_row, "prior mechanism logical evidence")
        evidence_id = _nonempty_string(
            row.get("evidence_id"), "prior mechanism logical evidence_id"
        )
        if evidence_id in logical_evidence_ids:
            raise ValueError("duplicate prior mechanism logical evidence")
        logical_evidence_ids.add(evidence_id)
        source_artifact_ids = _string_list(
            row.get("artifact_ids"), f"{evidence_id} artifact_ids"
        )
        if not set(source_artifact_ids).issubset(artifact_ids):
            raise ValueError(
                f"prior logical evidence references unknown artifact: {evidence_id}"
            )
        if row.get("valid_result") is not True:
            raise ValueError(f"prior logical evidence is not valid: {evidence_id}")
        evidence_index.append(
            {
                "evidence_id": evidence_id,
                "stage": _nonempty_string(row.get("stage"), f"{evidence_id} stage"),
                "artifact_ids": source_artifact_ids,
                "summary": _nonempty_string(
                    row.get("summary"), f"{evidence_id} summary"
                ),
                "scope": _nonempty_string(row.get("scope"), f"{evidence_id} scope"),
                "valid_result": True,
            }
        )

    raw_opportunities = diagnosis.get("architecture_opportunity_matrix")
    if not isinstance(raw_opportunities, list) or len(raw_opportunities) != len(
        OPPORTUNITY_IDS
    ):
        raise ValueError("prior mechanism opportunity coverage differs")
    opportunities = []
    opportunity_ids = set()
    prior_work_comparators = {"CoPPS", "BATA", "HMPPS", "MemRerank"}
    for raw_row in raw_opportunities:
        row = _require_mapping(raw_row, "prior mechanism opportunity")
        opportunity_id = _nonempty_string(
            row.get("opportunity_id"), "prior mechanism opportunity_id"
        )
        if opportunity_id in opportunity_ids or opportunity_id not in OPPORTUNITY_IDS:
            raise ValueError("prior mechanism opportunity identity differs")
        opportunity_ids.add(opportunity_id)
        evidence_ids = _string_list(
            row.get("evidence_ids"), f"{opportunity_id} evidence_ids"
        )
        if not set(evidence_ids).issubset(logical_evidence_ids):
            raise ValueError(
                f"prior opportunity references unknown logical evidence: {opportunity_id}"
            )
        hypotheses_for_opportunity = _string_list(
            row.get("bottleneck_hypotheses"),
            f"{opportunity_id} bottleneck_hypotheses",
        )
        if not set(hypotheses_for_opportunity).issubset(HYPOTHESIS_IDS):
            raise ValueError(
                f"prior opportunity references unknown hypothesis: {opportunity_id}"
            )
        priority = _nonempty_string(
            row.get("priority"), f"{opportunity_id} priority"
        )
        if priority not in {
            "primary_candidate",
            "secondary_candidate",
            "boundary_only",
            "deprioritized",
        }:
            raise ValueError(f"prior opportunity priority differs: {opportunity_id}")
        raw_prior_work = _require_mapping(
            row.get("prior_work_differentiation"),
            f"{opportunity_id} prior work differentiation",
        )
        if set(raw_prior_work) != prior_work_comparators:
            raise ValueError(
                f"prior opportunity comparator coverage differs: {opportunity_id}"
            )
        prior_work = {}
        for comparator in sorted(raw_prior_work):
            comparison = _require_mapping(
                raw_prior_work[comparator],
                f"{opportunity_id} comparator {comparator}",
            )
            prior_work[comparator] = {
                "shared_ground": _nonempty_string(
                    comparison.get("shared_ground"),
                    f"{opportunity_id} {comparator} shared_ground",
                ),
                "substantive_difference": _nonempty_string(
                    comparison.get("substantive_difference"),
                    f"{opportunity_id} {comparator} substantive_difference",
                ),
                "source_ref": _nonempty_string(
                    comparison.get("source_ref"),
                    f"{opportunity_id} {comparator} source_ref",
                ),
            }
        if (
            row.get("implementation_status") != "not_started_not_authorized"
            or row.get("evaluation_contract_unchanged") is not True
        ):
            raise ValueError(
                f"prior opportunity implementation boundary differs: {opportunity_id}"
            )
        opportunities.append(
            {
                "opportunity_id": opportunity_id,
                "priority": priority,
                "bottleneck_hypotheses": hypotheses_for_opportunity,
                "evidence_ids": evidence_ids,
                "innovation_target": _nonempty_string(
                    row.get("innovation_target"),
                    f"{opportunity_id} innovation_target",
                ),
                "architecture_requirement": _nonempty_string(
                    row.get("architecture_requirement"),
                    f"{opportunity_id} architecture_requirement",
                ),
                "necessary_modules": _string_list(
                    row.get("necessary_modules"),
                    f"{opportunity_id} necessary_modules",
                ),
                "training_signals": _string_list(
                    row.get("training_signals"),
                    f"{opportunity_id} training_signals",
                ),
                "train_only_data_requirements": _string_list(
                    row.get("train_only_data_requirements"),
                    f"{opportunity_id} train_only_data_requirements",
                ),
                "key_ablations": _string_list(
                    row.get("key_ablations"), f"{opportunity_id} key_ablations"
                ),
                "falsifiable_predictions": _string_list(
                    row.get("falsifiable_predictions"),
                    f"{opportunity_id} falsifiable_predictions",
                ),
                "prior_work_differentiation": prior_work,
                "implementation_status": "not_started_not_authorized",
                "evaluation_contract_unchanged": True,
            }
        )
    if opportunity_ids != set(OPPORTUNITY_IDS):
        raise ValueError("prior mechanism opportunity identities differ")
    opportunities.sort(
        key=lambda row: OPPORTUNITY_IDS.index(row["opportunity_id"])
    )

    raw_hypotheses = diagnosis.get("hypothesis_status_matrix")
    if not isinstance(raw_hypotheses, list) or len(raw_hypotheses) != len(
        HYPOTHESIS_IDS
    ):
        raise ValueError("prior mechanism H0--H5 coverage differs")
    hypotheses = []
    seen_hypotheses = set()
    allowed_prior_statuses = {"supported", "weakened", "rejected", "unresolved"}
    for raw_row in raw_hypotheses:
        row = _require_mapping(raw_row, "prior mechanism hypothesis")
        hypothesis_id = _nonempty_string(
            row.get("hypothesis_id"), "prior mechanism hypothesis_id"
        )
        if hypothesis_id not in HYPOTHESIS_IDS or hypothesis_id in seen_hypotheses:
            raise ValueError("prior mechanism hypothesis identity differs")
        seen_hypotheses.add(hypothesis_id)
        status = _nonempty_string(
            row.get("status"), f"prior mechanism {hypothesis_id} status"
        )
        if status not in allowed_prior_statuses:
            raise ValueError("prior mechanism hypothesis status differs")
        component_statuses = _require_mapping(
            row.get("component_statuses"),
            f"prior mechanism {hypothesis_id} component statuses",
        )
        if not component_statuses or any(
            not isinstance(key, str)
            or not key
            or value not in allowed_prior_statuses
            for key, value in component_statuses.items()
        ):
            raise ValueError("prior mechanism component status differs")
        hypotheses.append(
            {
                "hypothesis_id": hypothesis_id,
                "status": status,
                "claim_level": _nonempty_string(
                    row.get("claim_level"),
                    f"prior mechanism {hypothesis_id} claim_level",
                ),
                "statement_verbatim": _nonempty_string(
                    row.get("statement_verbatim"),
                    f"prior mechanism {hypothesis_id} statement",
                ),
                "rationale": _nonempty_string(
                    row.get("rationale"),
                    f"prior mechanism {hypothesis_id} rationale",
                ),
                "component_statuses": dict(component_statuses),
                "supporting_evidence_ids": _string_list(
                    row.get("supporting_evidence_ids"),
                    f"prior mechanism {hypothesis_id} supporting evidence",
                    allow_empty=True,
                ),
                "opposing_evidence_ids": _string_list(
                    row.get("opposing_evidence_ids"),
                    f"prior mechanism {hypothesis_id} opposing evidence",
                    allow_empty=True,
                ),
                "contradiction_ids": _string_list(
                    row.get("contradiction_ids"),
                    f"prior mechanism {hypothesis_id} contradictions",
                    allow_empty=True,
                ),
                "remaining_uncertainty": _string_list(
                    row.get("remaining_uncertainty"),
                    f"prior mechanism {hypothesis_id} remaining uncertainty",
                ),
                "scope_limitations": _string_list(
                    row.get("scope_limitations"),
                    f"prior mechanism {hypothesis_id} scope limitations",
                ),
                "triangulation": copy.deepcopy(
                    _require_mapping(
                        row.get("triangulation"),
                        f"prior mechanism {hypothesis_id} triangulation",
                    )
                ),
            }
        )
    if seen_hypotheses != set(HYPOTHESIS_IDS):
        raise ValueError("prior mechanism H0--H5 identities differ")
    hypotheses.sort(key=lambda row: HYPOTHESIS_IDS.index(row["hypothesis_id"]))
    referenced_logical_evidence = {
        evidence_id
        for row in hypotheses
        for key in ("supporting_evidence_ids", "opposing_evidence_ids")
        for evidence_id in row[key]
    }
    if not referenced_logical_evidence.issubset(logical_evidence_ids):
        raise ValueError("prior H0--H5 references unknown logical evidence")

    raw_contradictions = diagnosis.get("contradictions")
    if not isinstance(raw_contradictions, list) or len(raw_contradictions) != 10:
        raise ValueError("prior mechanism contradiction coverage differs")
    contradictions = []
    contradiction_ids = set()
    for raw_row in raw_contradictions:
        row = _require_mapping(raw_row, "prior mechanism contradiction")
        contradiction_id = _nonempty_string(
            row.get("contradiction_id"), "prior mechanism contradiction_id"
        )
        if contradiction_id in contradiction_ids:
            raise ValueError("duplicate prior mechanism contradiction")
        contradiction_ids.add(contradiction_id)
        contradictions.append(
            {
                "contradiction_id": contradiction_id,
                "description": _nonempty_string(
                    row.get("description"), f"{contradiction_id} description"
                ),
                "interpretation": _nonempty_string(
                    row.get("interpretation"), f"{contradiction_id} interpretation"
                ),
                "evidence_ids": _string_list(
                    row.get("evidence_ids"), f"{contradiction_id} evidence_ids"
                ),
                "mechanical_failure": row.get("mechanical_failure") is True,
            }
        )
    referenced_contradictions = {
        contradiction_id
        for row in hypotheses
        for contradiction_id in row["contradiction_ids"]
    }
    if not referenced_contradictions.issubset(contradiction_ids):
        raise ValueError("prior H0--H5 references an unknown contradiction")
    if any(
        not set(row["evidence_ids"]).issubset(logical_evidence_ids)
        for row in contradictions
    ):
        raise ValueError("prior contradiction references unknown logical evidence")

    return {
        "source": dict(diagnosis_identity),
        "report_id": diagnosis["report_id"],
        "status": diagnosis["status"],
        "scope_and_boundaries": dict(scope),
        "artifact_registry": artifact_registry,
        "evidence_index": evidence_index,
        "hypothesis_status_matrix": hypotheses,
        "contradictions": contradictions,
        "architecture_opportunity_matrix": opportunities,
        "scientific_effect_values_recomputed": False,
        "source_test_opened": False,
        "new_dataset_opened": False,
        "architecture_implemented": False,
        "interpretation_boundary": (
            "This is the frozen pre-deep-dive M0--M3 diagnosis. The final H0--H5 "
            "matrix may update it only with admitted deep-dive evidence; prior "
            "contradictions, opportunity falsification contracts, and unresolved "
            "single-seed/data boundaries remain visible."
        ),
    }


def build_comprehensive_report(
    root: str | Path,
    decisions: Mapping[str, Any],
    *,
    formal_report_path: str | Path,
    json_output: str | Path,
    markdown_output: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Validate all final gates and atomically emit JSON plus Markdown."""

    root_path = Path(root).resolve()
    frozen_observation_snapshot = build_frozen_observation_snapshot(root_path)
    prior_mechanism_snapshot = build_prior_mechanism_diagnosis_snapshot(root_path)
    report_plan_path = _resolve(
        root_path, COMPREHENSIVE_REPORT_PLAN_IDENTITY["path"]
    )
    if sha256_file(report_plan_path) != COMPREHENSIVE_REPORT_PLAN_IDENTITY["sha256"]:
        raise ValueError("comprehensive report plan hash drift")
    readiness = build_comprehensive_readiness(root_path)
    if readiness.get("status") != "completed" or readiness.get(
        "final_comprehensive_report_ready"
    ) is not True:
        raise ValueError("comprehensive report prerequisites are not terminal")
    component_evidence_role_coverage = _build_component_evidence_role_coverage(
        readiness
    )

    supplements = audit_supplemental_evidence_registry(root_path)
    if supplements.get("status") != "completed":
        raise ValueError("supplemental evidence registry is not terminal")
    completed_supplements = {
        str(row["evidence_id"]): row for row in supplements["entries"]
    }
    if set(completed_supplements) != set(EXPECTED_SUPPLEMENT_IDS) or any(
        row.get("status") != "completed" for row in completed_supplements.values()
    ):
        raise ValueError("supplemental evidence coverage is incomplete")
    interface_coverage = build_transformer_interface_coverage(
        completed_formal=set(EXPECTED_DELIVERABLES),
        completed_supplements=set(completed_supplements),
        supplement_model_scopes={
            evidence_id: set(row["model_scope"])
            for evidence_id, row in completed_supplements.items()
        },
        supplement_component_scopes={
            evidence_id: set(row["components"])
            for evidence_id, row in completed_supplements.items()
        },
    )
    if readiness.get("transformer_internal_interface_coverage") != interface_coverage:
        raise ValueError("Transformer internal-interface readiness coverage differs")

    formal_path = _resolve(root_path, formal_report_path)
    formal = _load_json(formal_path)
    _audit_formal_report(formal)
    design_payload = _load_json(
        _resolve(root_path, completed_supplements[DESIGN_GATE_SUPPLEMENT]["path"])
    )
    component_gate_matrix = _build_component_bidirectional_gate_matrix(
        design_payload,
        evidence_identity=completed_supplements[DESIGN_GATE_SUPPLEMENT],
    )
    design_nodes = set(
        component_gate_matrix["cross_model"]["design_prioritized_nodes"]
    )
    necessity_component_models = _derive_necessity_component_models(design_payload)

    normalized = validate_comprehensive_decisions(
        decisions,
        completed_formal=set(EXPECTED_DELIVERABLES),
        completed_supplements=set(completed_supplements),
        supplement_metadata=completed_supplements,
        design_qualified_nodes=design_nodes,
        necessity_supported_component_models=necessity_component_models,
    )
    _audit_comprehensive_against_formal(normalized, formal)
    normalized = _bind_opportunity_evidence_identities(
        normalized,
        formal=formal,
        completed_supplements=completed_supplements,
    )
    opportunity_lineage_matrix = _build_opportunity_lineage_matrix(
        prior=prior_mechanism_snapshot,
        formal=formal,
        comprehensive=normalized,
    )
    reproducibility_ledger = _build_reproducibility_ledger(
        formal=formal,
        completed_supplements=completed_supplements,
        root=root_path,
    )
    json_path = _resolve(root_path, json_output)
    markdown_path = _resolve(root_path, markdown_output)
    if json_path == markdown_path:
        raise ValueError("JSON and Markdown outputs must differ")
    if not overwrite:
        existing = [str(path) for path in (json_path, markdown_path) if path.exists()]
        if existing:
            raise FileExistsError(f"refusing to overwrite report outputs: {existing}")

    payload = {
        "schema_version": 1,
        "analysis_type": ANALYSIS_TYPE,
        "report_id": normalized["report_id"],
        "status": "completed",
        "comprehensive_report_plan": dict(COMPREHENSIVE_REPORT_PLAN_IDENTITY),
        "evidence_admission": {
            "readiness": readiness,
            "formal_report": {
                "path": _display_path(root_path, formal_path),
                "sha256": sha256_file(formal_path),
                "analysis_type": formal["analysis_type"],
            },
            "supplement_registry": supplements["registry"],
            "supplement_registry_manifest": supplements["registry_manifest"],
            "supplements": list(completed_supplements.values()),
            "component_design_gate": {
                "path": completed_supplements[DESIGN_GATE_SUPPLEMENT]["path"],
                "sha256": completed_supplements[DESIGN_GATE_SUPPLEMENT]["sha256"],
                "cross_model_design_qualified_nodes": sorted(design_nodes),
            },
            "source_test_opened": False,
            "qrels_or_score_bundles_opened_by_this_builder": False,
        },
        "formal_execution_census": formal.get("execution_census", {}),
        "formal_layerwise_attenuation_profile": formal.get(
            "layerwise_attenuation_profile", {}
        ),
        "formal_attenuation_transition_profile": formal.get(
            "attenuation_transition_profile", {}
        ),
        "formal_architecture_opportunity_ranking": formal.get(
            "architecture_opportunity_ranking", []
        ),
        "execution_axis_census": EXECUTION_AXIS_CENSUS,
        "frozen_observation_scope_contract": list(
            FROZEN_OBSERVATION_SCOPE_CONTRACT
        ),
        "frozen_observation_evidence": [
            dict(identity) for identity in FROZEN_OBSERVATION_EVIDENCE_IDENTITIES
        ],
        "frozen_observation_machine_snapshot": frozen_observation_snapshot,
        "prior_mechanism_diagnosis_snapshot": prior_mechanism_snapshot,
        "opportunity_lineage_matrix": opportunity_lineage_matrix,
        "paper_method_stage_requirements": list(PAPER_METHOD_STAGE_REQUIREMENTS),
        "reproducibility_ledger": reproducibility_ledger,
        "functional_localization_contract": FUNCTIONAL_LOCALIZATION_CONTRACT,
        "localization_to_design_bridge": list(LOCALIZATION_TO_DESIGN_BRIDGE),
        "component_bidirectional_gate_matrix": component_gate_matrix,
        "necessity_direction_claim_boundary": dict(
            NECESSITY_DIRECTION_CLAIM_BOUNDARY
        ),
        "component_functional_questions": COMPONENT_FUNCTIONAL_QUESTIONS,
        "component_evidence_role_coverage": component_evidence_role_coverage,
        "transformer_internal_interface_coverage": interface_coverage,
        "history_signal_observation_scope_contract": list(
            HISTORY_SIGNAL_OBSERVATION_SCOPE_CONTRACT
        ),
        "frozen_model_architecture_audit": readiness[
            "frozen_model_architecture_audit"
        ],
        **normalized,
        "claim_invariants": dict(CLAIM_INVARIANTS),
    }
    payload["report_section_contract"] = _audit_report_section_contract(payload)
    markdown = render_comprehensive_report_markdown(payload)
    _atomic_write_pair(
        json_path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        markdown_path,
        markdown,
    )
    return payload


def _build_opportunity_lineage_matrix(
    *,
    prior: Mapping[str, Any],
    formal: Mapping[str, Any],
    comprehensive: Mapping[str, Any],
) -> dict[str, Any]:
    """Join prior hypotheses, formal ranks, and final dispositions by frozen ID."""

    prior_rows = {
        str(row.get("opportunity_id")): row
        for row in _require_list(
            prior, "architecture_opportunity_matrix"
        )
    }
    formal_rows = {
        str(row.get("opportunity_id")): row
        for row in _require_list(formal, "architecture_opportunity_ranking")
    }
    dispositions = _require_mapping(
        comprehensive.get("formal_opportunity_disposition"),
        "formal_opportunity_disposition",
    )
    if (
        set(prior_rows) != set(OPPORTUNITY_IDS)
        or set(formal_rows) != set(OPPORTUNITY_IDS)
        or set(dispositions) != set(OPPORTUNITY_IDS)
    ):
        raise ValueError("opportunity lineage predecessor coverage differs")
    comprehensive_targets = {
        str(row.get("opportunity_id")): row
        for row in _require_list(comprehensive, "optimization_opportunities")
    }
    if len(comprehensive_targets) != len(
        _require_list(comprehensive, "optimization_opportunities")
    ):
        raise ValueError("duplicate comprehensive opportunity lineage target")
    not_recommended_rows = _require_list(comprehensive, "not_recommended")
    not_recommended_targets = {
        str(row.get("direction")): row for row in not_recommended_rows
    }
    if len(not_recommended_targets) != len(not_recommended_rows):
        raise ValueError("duplicate not-recommended opportunity lineage target")

    rows = []
    for opportunity_id in OPPORTUNITY_IDS:
        prior_row = prior_rows[opportunity_id]
        formal_row = formal_rows[opportunity_id]
        disposition = _require_mapping(
            dispositions[opportunity_id],
            f"opportunity disposition {opportunity_id}",
        )
        target_id = _nonempty_string(
            disposition.get("target_id"), f"{opportunity_id} target_id"
        )
        disposition_kind = _nonempty_string(
            disposition.get("disposition"), f"{opportunity_id} disposition"
        )
        if disposition_kind == "mapped_to_comprehensive_opportunity":
            if target_id not in comprehensive_targets:
                raise ValueError(
                    f"opportunity lineage target is missing: {opportunity_id}/{target_id}"
                )
            target = comprehensive_targets[target_id]
            target_summary = {
                "target_kind": "optimization_opportunity",
                "target_id": target_id,
                "rank": int(target["rank"]),
                "design_priority": str(target["design_priority"]),
                "actual_evidence_level": str(target["actual_evidence_level"]),
                "basis": None,
                "functional_component": str(target["functional_component"]),
                "model_scope": list(target["model_scope"]),
                "supporting_findings": list(target["supporting_findings"]),
                "utility_gain_established": target.get("utility_gain_established")
                is True,
                "architecture_implemented": target.get("architecture_implemented")
                is True,
            }
        elif disposition_kind == "mapped_to_not_recommended":
            if target_id not in not_recommended_targets:
                raise ValueError(
                    f"not-recommended lineage target is missing: {opportunity_id}/{target_id}"
                )
            target = not_recommended_targets[target_id]
            target_summary = {
                "target_kind": "not_recommended",
                "target_id": target_id,
                "rank": None,
                "design_priority": "not_recommended",
                "actual_evidence_level": None,
                "basis": str(target["basis"]),
                "functional_component": str(target["functional_component"]),
                "model_scope": list(target["model_scope"]),
                "supporting_findings": list(target["supporting_findings"]),
                "utility_gain_established": False,
                "architecture_implemented": False,
            }
        else:
            raise ValueError(
                f"unknown opportunity lineage disposition: {opportunity_id}"
            )
        if (
            target_summary["utility_gain_established"]
            or target_summary["architecture_implemented"]
        ):
            raise ValueError("opportunity lineage exceeds mechanism-stage authority")
        rows.append(
            {
                "opportunity_id": opportunity_id,
                "prior_priority": str(prior_row["priority"]),
                "prior_bottleneck_hypotheses": list(
                    prior_row["bottleneck_hypotheses"]
                ),
                "prior_logical_evidence": list(prior_row["evidence_ids"]),
                "formal_rank": int(formal_row["rank"]),
                "formal_status": str(formal_row["status"]),
                "formal_evidence_deliverables": list(
                    formal_row["evidence_deliverables"]
                ),
                "disposition": disposition_kind,
                **target_summary,
            }
        )
    if sorted(row["formal_rank"] for row in rows) != list(
        range(1, len(OPPORTUNITY_IDS) + 1)
    ):
        raise ValueError("opportunity lineage formal ranks differ")
    return {
        "rows": rows,
        "prior_opportunity_count": len(prior_rows),
        "formal_opportunity_count": len(formal_rows),
        "disposition_count": len(dispositions),
        "all_predecessors_disposed_exactly_once": True,
        "utility_gain_established": False,
        "architecture_implemented": False,
        "interpretation_boundary": (
            "This matrix records lineage and changing evidence disposition, not a "
            "trained-method result. Prior priority, formal rank, and comprehensive "
            "component-gated priority have distinct evidentiary meanings."
        ),
    }


def _build_component_evidence_role_coverage(
    readiness: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Expose outcome-independent causal-role coverage for every component."""

    raw_rows = readiness.get("component_artifact_coverage")
    if not isinstance(raw_rows, list):
        raise ValueError("component artifact coverage is missing")
    rows_by_id: dict[str, Mapping[str, Any]] = {}
    for raw in raw_rows:
        row = _require_mapping(raw, "component artifact coverage row")
        component_id = str(row.get("component_id"))
        if component_id in rows_by_id:
            raise ValueError(f"duplicate component artifact coverage: {component_id}")
        rows_by_id[component_id] = row
    if set(rows_by_id) != set(COMPONENT_IDS):
        raise ValueError("component artifact coverage differs from the 18 components")
    if readiness.get("component_count") != len(COMPONENT_IDS):
        raise ValueError("component artifact coverage count differs")

    normalized: dict[str, dict[str, Any]] = {}
    for component_id in COMPONENT_IDS:
        row = rows_by_id[component_id]
        boolean_fields = (
            "any_evidence_artifact_completed",
            "causal_role_artifact_registered",
            "causal_role_artifact_completed",
            "q2_q3_causal_role_artifacts_registered",
            "q2_q3_causal_role_artifacts_completed",
        )
        booleans: dict[str, bool] = {}
        for field in boolean_fields:
            value = row.get(field)
            if not isinstance(value, bool):
                raise ValueError(
                    f"component artifact coverage {field} is not boolean: "
                    f"{component_id}"
                )
            booleans[field] = value
        registered_models = _model_scope(
            row.get("causal_role_model_scope_registered"), allow_empty=True
        )
        completed_models = _model_scope(
            row.get("causal_role_model_scope_completed"), allow_empty=True
        )
        if not set(completed_models).issubset(registered_models):
            raise ValueError(
                "completed causal-role model scope exceeds registration: "
                f"{component_id}"
            )
        if booleans["causal_role_artifact_completed"] != bool(completed_models):
            raise ValueError(
                "component causal-role completion disagrees with model scope: "
                f"{component_id}"
            )
        if booleans["causal_role_artifact_registered"] != bool(registered_models):
            raise ValueError(
                "component causal-role registration disagrees with model scope: "
                f"{component_id}"
            )
        if booleans["q2_q3_causal_role_artifacts_registered"] != (
            PRIMARY_DESIGN_MODELS.issubset(registered_models)
        ):
            raise ValueError(
                "component Q2/Q3 causal-role registration disagrees with model "
                f"scope: {component_id}"
            )
        if booleans["q2_q3_causal_role_artifacts_completed"] != (
            PRIMARY_DESIGN_MODELS.issubset(completed_models)
        ):
            raise ValueError(
                "component Q2/Q3 causal-role completion disagrees with model "
                f"scope: {component_id}"
            )
        normalized[component_id] = {
            **booleans,
            "causal_role_model_scope_registered": registered_models,
            "causal_role_model_scope_completed": completed_models,
            "scientific_support_inferred_from_completion": False,
        }

    without_registered = sorted(
        component_id
        for component_id, row in normalized.items()
        if not row["causal_role_artifact_registered"]
    )
    if without_registered != sorted(
        _string_list(
            readiness.get("components_without_registered_causal_role_artifact"),
            "components_without_registered_causal_role_artifact",
            allow_empty=True,
        )
    ):
        raise ValueError("component causal-role coverage summary differs")
    if sum(
        int(row["any_evidence_artifact_completed"])
        for row in normalized.values()
    ) != readiness.get("components_with_any_completed_artifact"):
        raise ValueError("component any-evidence completed count differs")
    if sum(
        int(row["causal_role_artifact_registered"])
        for row in normalized.values()
    ) != readiness.get("components_with_registered_causal_role_artifact"):
        raise ValueError("component causal-role registered count differs")
    if sum(
        int(row["causal_role_artifact_completed"])
        for row in normalized.values()
    ) != readiness.get("components_with_completed_causal_role_artifact"):
        raise ValueError("component causal-role completed count differs")
    if sum(
        int(row["q2_q3_causal_role_artifacts_completed"])
        for row in normalized.values()
    ) != readiness.get("components_with_completed_q2_q3_causal_role_artifacts"):
        raise ValueError("component Q2/Q3 causal-role completed count differs")
    return normalized


def validate_comprehensive_decisions(
    decisions: Mapping[str, Any],
    *,
    completed_formal: set[str],
    completed_supplements: set[str],
    supplement_metadata: Mapping[str, Mapping[str, Any]],
    design_qualified_nodes: set[str],
    necessity_supported_component_models: Mapping[str, set[str]],
) -> dict[str, Any]:
    """Validate the human synthesis without silently filling scientific claims."""

    if decisions.get("schema_version") != 1:
        raise ValueError("comprehensive decisions schema_version must be 1")
    if decisions.get("worksheet_status") != "final":
        raise ValueError("comprehensive decisions worksheet_status must be final")
    if set(supplement_metadata) != completed_supplements:
        raise ValueError("comprehensive supplement metadata coverage drift")
    if set(necessity_supported_component_models) != set(COMPONENT_IDS) or any(
        not set(model_scope).issubset(MODEL_IDS)
        for model_scope in necessity_supported_component_models.values()
    ):
        raise ValueError("necessity component/model support coverage drift")
    for evidence_id, metadata in supplement_metadata.items():
        components = metadata.get("components")
        model_scope = metadata.get("model_scope")
        if (
            not isinstance(components, (list, tuple, set))
            or not set(components).issubset(COMPONENT_IDS)
            or not isinstance(model_scope, (list, tuple, set))
            or not set(model_scope).issubset(MODEL_IDS)
        ):
            raise ValueError(
                f"comprehensive supplement metadata differs: {evidence_id}"
            )
    report_id = _nonempty_string(decisions.get("report_id"), "report_id")
    narratives = _exact_mapping(decisions, "narratives", REQUIRED_NARRATIVES)

    findings = _require_list(decisions, "findings")
    if not findings:
        raise ValueError("findings must not be empty")
    finding_ids: set[str] = set()
    normalized_findings = []
    for index, raw in enumerate(findings):
        row = _require_mapping(raw, f"findings[{index}]")
        finding_id = _nonempty_string(row.get("finding_id"), "finding_id")
        if finding_id in finding_ids:
            raise ValueError(f"duplicate finding_id: {finding_id}")
        finding_ids.add(finding_id)
        level = _evidence_level(row.get("evidence_level"))
        formal_refs = _reference_list(
            row, "supporting_formal_deliverables", completed_formal
        )
        supplement_refs = _reference_list(
            row, "supporting_supplements", completed_supplements
        )
        if not formal_refs and not supplement_refs:
            raise ValueError(f"finding {finding_id} has no admitted evidence")
        if level == "G" and (
            DESIGN_GATE_SUPPLEMENT not in supplement_refs
            or set(_model_scope(row.get("model_scope")))
            != PRIMARY_DESIGN_MODELS
        ):
            raise ValueError(
                f"finding {finding_id} level G lacks the cross-model design gate"
            )
        if level == "N" and NECESSITY_SUPPLEMENT not in supplement_refs:
            raise ValueError(
                f"finding {finding_id} level N lacks reverse-necessity evidence"
            )
        if level == "S" and not formal_refs:
            raise ValueError(
                f"finding {finding_id} level S lacks formal sufficiency evidence"
            )
        if (
            not formal_refs
            and set(supplement_refs).issubset(DESCRIPTIVE_SUPPLEMENTS)
            and level not in {"M", "D", "U"}
        ):
            raise ValueError(
                f"finding {finding_id} upgrades descriptive supplements"
            )
        finding_model_scope = _model_scope(row.get("model_scope"))
        normalized_findings.append(
            {
                "finding_id": finding_id,
                "title": _nonempty_string(row.get("title"), "finding title"),
                "evidence_level": level,
                "claim": _nonempty_string(row.get("claim"), "finding claim"),
                "model_scope": finding_model_scope,
                "dataset_scope": _dataset_scope(row.get("dataset_scope")),
                "supporting_formal_deliverables": formal_refs,
                "supporting_supplements": supplement_refs,
                "contradictory_evidence": _string_list(
                    row.get("contradictory_evidence"),
                    "finding contradictory_evidence",
                    allow_empty=True,
                ),
                "do_not_infer": _string_list(
                    row.get("do_not_infer"), "finding do_not_infer"
                ),
            }
        )

    findings_by_id = {row["finding_id"]: row for row in normalized_findings}
    normalized_narratives = {}
    for narrative_id in REQUIRED_NARRATIVES:
        row = _require_mapping(narratives[narrative_id], f"narratives.{narrative_id}")
        text = _nonempty_string(row.get("text"), f"narratives.{narrative_id}.text")
        level = _evidence_level(row.get("evidence_level"))
        narrative_findings = _finding_refs(row, finding_ids)
        if not narrative_findings:
            raise ValueError(
                f"narrative lacks supporting findings: {narrative_id}"
            )
        if level != "U" and not any(
            findings_by_id[finding_id]["evidence_level"] == level
            for finding_id in narrative_findings
        ):
            raise ValueError(
                "narrative evidence level lacks a matching finding: "
                f"{narrative_id}"
            )
        if _FREE_TEXT_METRIC_LITERAL.search(text):
            raise ValueError(
                "narrative cannot hand-copy a metric literal; use admitted tables: "
                f"{narrative_id}"
            )
        if narrative_id in {
            "executive_summary",
            "cross_model_boundary",
            "paper_claim_boundary",
        } and _ABSOLUTE_INTERNAL_INDEX.search(text):
            raise ValueError(
                f"absolute internal index forbidden in narrative: {narrative_id}"
            )
        normalized_narratives[narrative_id] = {
            "text": text,
            "evidence_level": level,
            "supporting_findings": narrative_findings,
            "do_not_infer": _string_list(
                row.get("do_not_infer"), f"narratives.{narrative_id}.do_not_infer"
            ),
        }

    component_matrix = _exact_mapping(decisions, "component_matrix", COMPONENT_IDS)
    design_qualified_components = set().union(
        *(
            DESIGN_NODE_COMPONENTS[node]
            for node in design_qualified_nodes
            if node in DESIGN_NODE_COMPONENTS
        ),
        set(),
    )
    normalized_components: dict[str, Any] = {}
    for component_id in COMPONENT_IDS:
        row = _require_mapping(component_matrix[component_id], component_id)
        status = str(row.get("status"))
        if status not in COMPONENT_STATUSES:
            raise ValueError(f"invalid component status: {component_id}={status}")
        component_level = _evidence_level(row.get("evidence_level"))
        component_model_scope = _model_scope(
            row.get("model_scope"), allow_empty=True
        )
        if status == "supported" and component_level not in {"S", "N", "G"}:
            raise ValueError(
                f"supported component lacks causal evidence level: {component_id}"
            )
        if status == "weakened" and component_level not in {"D", "S"}:
            raise ValueError(
                f"weakened component lacks registered negative evidence level: {component_id}"
            )
        if status == "unresolved" and component_level != "U":
            raise ValueError(f"unresolved component must use level U: {component_id}")
        if status == "untested" and component_level != "U":
            raise ValueError(f"untested component must use level U: {component_id}")
        if status == "mechanical_failure" and component_level != "M":
            raise ValueError(
                f"mechanical-failure component must use level M: {component_id}"
            )
        component_findings = _finding_refs(row, finding_ids)
        if status == "weakened":
            weakened_findings = [
                findings_by_id[finding_id]
                for finding_id in component_findings
                if findings_by_id[finding_id]["evidence_level"] == component_level
                and bool(
                    set(
                        findings_by_id[finding_id][
                            "supporting_formal_deliverables"
                        ]
                    )
                    & set(COMPONENT_ALLOWED_DELIVERABLES[component_id])
                )
            ]
            weakened_models = {
                model_id
                for finding in weakened_findings
                for model_id in finding["model_scope"]
            }
            if (
                not weakened_findings
                or not component_model_scope
                or not set(component_model_scope).issubset(weakened_models)
            ):
                raise ValueError(
                    "weakened component lacks level/component/model-matched "
                    f"registered evidence: {component_id}"
                )
        if status == "supported":
            if not component_findings:
                raise ValueError(
                    f"supported component lacks supporting findings: {component_id}"
                )
            referenced_findings = [
                findings_by_id[finding_id] for finding_id in component_findings
            ]
            matched_findings: list[Mapping[str, Any]] = []
            if component_level == "S":
                allowed_causal = set(
                    COMPONENT_SUPPORT_REQUIRED_CAUSAL_EVIDENCE[component_id]
                )
                matched_findings = [
                    finding
                    for finding in referenced_findings
                    if (
                    finding["evidence_level"] == "S"
                    and bool(
                        set(finding["supporting_formal_deliverables"])
                        & allowed_causal
                    )
                    )
                ]
                if not matched_findings:
                    raise ValueError(
                        "supported component lacks component-matched sufficiency "
                        f"evidence: {component_id}"
                    )
            elif component_level == "N":
                matched_findings = [
                    finding
                    for finding in referenced_findings
                    if (
                    finding["evidence_level"] == "N"
                    and NECESSITY_SUPPLEMENT
                    in finding["supporting_supplements"]
                    )
                ]
                necessity_models = set(
                    necessity_supported_component_models[component_id]
                )
                if (
                    component_id not in NECESSITY_COMPONENTS
                    or not matched_findings
                    or not set(component_model_scope).issubset(necessity_models)
                ):
                    raise ValueError(
                        "supported component lacks component/model-matched necessity "
                        f"evidence: {component_id}"
                    )
            elif component_level == "G":
                matched_findings = [
                    finding
                    for finding in referenced_findings
                    if (
                    finding["evidence_level"] == "G"
                    and DESIGN_GATE_SUPPLEMENT
                    in finding["supporting_supplements"]
                    )
                ]
                if (
                    component_id not in design_qualified_components
                    or not matched_findings
                    or set(component_model_scope) != PRIMARY_DESIGN_MODELS
                ):
                    raise ValueError(
                        "supported component lacks component-matched cross-model "
                        f"design gate: {component_id}"
                    )
            matched_model_scope = {
                model_id
                for finding in matched_findings
                for model_id in finding["model_scope"]
            }
            if not component_model_scope or not set(component_model_scope).issubset(
                matched_model_scope
            ):
                raise ValueError(
                    "supported component model scope exceeds matched causal "
                    f"findings: {component_id}"
                )
        for finding_id in component_findings:
            finding = findings_by_id[finding_id]
            if not any(
                _finding_matches_component_model(
                    finding,
                    component_id=component_id,
                    model_id=model_id,
                    supplement_metadata=supplement_metadata,
                )
                for model_id in set(component_model_scope)
                & set(finding["model_scope"])
            ):
                raise ValueError(
                    "component cites evidence outside its component/model scope: "
                    f"{component_id}/{finding_id}"
                )
        if status == "untested" and component_findings:
            raise ValueError(f"untested component cites evidence: {component_id}")
        normalized_components[component_id] = {
            "mechanism_question": COMPONENT_FUNCTIONAL_QUESTIONS[component_id],
            "claim_boundary": COMPONENT_PROBE_CLAIM_BOUNDARIES[component_id],
            "status": status,
            "evidence_level": component_level,
            "summary": _nonempty_string(row.get("summary"), "component summary"),
            "model_scope": component_model_scope,
            "supporting_findings": component_findings,
            "remaining_uncertainty": _nonempty_string(
                row.get("remaining_uncertainty"), "component remaining_uncertainty"
            ),
        }

    # The aggregate component row is useful for the causal-chain and opportunity
    # gates, but it cannot represent model heterogeneity.  Keep a second,
    # exhaustive 18 x 4 matrix so that a Q2/Q3 result never silently becomes a
    # Q0/Q1 claim and an uncovered model cannot disappear behind model_scope.
    component_model_matrix = _exact_mapping(
        decisions, "component_model_matrix", COMPONENT_IDS
    )
    component_model_registered_sources = {
        component_id: {
            model_id: sorted(
                [
                    deliverable
                    for deliverable in completed_formal
                    if deliverable in COMPONENT_ALLOWED_DELIVERABLES[component_id]
                    and model_id
                    in COMPONENT_DELIVERABLE_MODEL_COVERAGE[component_id].get(
                        deliverable, set()
                    )
                ]
                + [
                    supplement_id
                    for supplement_id, metadata in supplement_metadata.items()
                    if component_id in metadata["components"]
                    and model_id in metadata["model_scope"]
                ]
            )
            for model_id in MODEL_IDS
        }
        for component_id in COMPONENT_IDS
    }
    normalized_component_models: dict[str, dict[str, Any]] = {}
    for component_id in COMPONENT_IDS:
        model_rows = _exact_mapping(
            component_model_matrix,
            component_id,
            MODEL_IDS,
        )
        normalized_component_models[component_id] = {}
        for model_id in MODEL_IDS:
            item = _require_mapping(
                model_rows[model_id],
                f"component_model_matrix.{component_id}.{model_id}",
            )
            cell_status = str(item.get("status"))
            if cell_status not in COMPONENT_STATUSES:
                raise ValueError(
                    "invalid component-model status: "
                    f"{component_id}/{model_id}={cell_status}"
                )
            cell_level = _evidence_level(item.get("evidence_level"))
            if cell_status == "supported" and cell_level not in {"S", "N", "G"}:
                raise ValueError(
                    "supported component-model cell lacks causal evidence: "
                    f"{component_id}/{model_id}"
                )
            if cell_status == "weakened" and cell_level not in {"D", "S"}:
                raise ValueError(
                    "weakened component-model cell lacks registered negative "
                    f"evidence: {component_id}/{model_id}"
                )
            if cell_status in {"unresolved", "untested"} and cell_level != "U":
                raise ValueError(
                    f"{cell_status} component-model cell must use U: "
                    f"{component_id}/{model_id}"
                )
            if cell_status == "mechanical_failure" and cell_level != "M":
                raise ValueError(
                    "mechanical-failure component-model cell must use M: "
                    f"{component_id}/{model_id}"
                )
            registered_sources = component_model_registered_sources[component_id][
                model_id
            ]
            if registered_sources and cell_status == "untested":
                raise ValueError(
                    "covered component-model cell cannot be untested: "
                    f"{component_id}/{model_id}"
                )
            if not registered_sources and cell_status != "untested":
                raise ValueError(
                    "uncovered component-model cell must be untested: "
                    f"{component_id}/{model_id}"
                )

            cell_findings = _finding_refs(item, finding_ids)
            matched_cell_findings = [
                findings_by_id[finding_id]
                for finding_id in cell_findings
                if model_id in findings_by_id[finding_id]["model_scope"]
                and _finding_matches_component_model(
                    findings_by_id[finding_id],
                    component_id=component_id,
                    model_id=model_id,
                    supplement_metadata=supplement_metadata,
                )
            ]
            if cell_status == "untested" and cell_findings:
                raise ValueError(
                    "untested component-model cell cites evidence: "
                    f"{component_id}/{model_id}"
                )
            if cell_status in {"supported", "weakened", "mechanical_failure"}:
                level_matched = [
                    finding
                    for finding in matched_cell_findings
                    if finding["evidence_level"] == cell_level
                ]
                if not level_matched:
                    raise ValueError(
                        "component-model claim lacks level/component/model-matched "
                        f"evidence: {component_id}/{model_id}"
                    )
            if cell_level == "S" and cell_status == "supported":
                if not any(
                    set(finding["supporting_formal_deliverables"])
                    & set(COMPONENT_SUPPORT_REQUIRED_CAUSAL_EVIDENCE[component_id])
                    for finding in matched_cell_findings
                    if finding["evidence_level"] == "S"
                ):
                    raise ValueError(
                        "supported component-model cell lacks formal sufficiency "
                        f"evidence: {component_id}/{model_id}"
                    )
            if cell_level == "N" and cell_status == "supported":
                if (
                    component_id not in NECESSITY_COMPONENTS
                    or model_id
                    not in necessity_supported_component_models[component_id]
                    or not any(
                        finding["evidence_level"] == "N"
                        and NECESSITY_SUPPLEMENT
                        in finding["supporting_supplements"]
                        for finding in matched_cell_findings
                    )
                ):
                    raise ValueError(
                        "supported component-model cell lacks necessity evidence: "
                        f"{component_id}/{model_id}"
                    )
            if cell_level == "G" and cell_status == "supported":
                if (
                    component_id not in design_qualified_components
                    or model_id not in PRIMARY_DESIGN_MODELS
                    or not any(
                        finding["evidence_level"] == "G"
                        and DESIGN_GATE_SUPPLEMENT
                        in finding["supporting_supplements"]
                        for finding in matched_cell_findings
                    )
                ):
                    raise ValueError(
                        "supported component-model cell lacks cross-model design "
                        f"gate: {component_id}/{model_id}"
                    )
            if len(matched_cell_findings) != len(cell_findings):
                raise ValueError(
                    "component-model cell cites evidence outside its scope: "
                    f"{component_id}/{model_id}"
                )

            normalized_component_models[component_id][model_id] = {
                "status": cell_status,
                "evidence_level": cell_level,
                "registered_evidence_sources": registered_sources,
                "summary": _nonempty_string(
                    item.get("summary"), "component-model summary"
                ),
                "supporting_findings": cell_findings,
                "remaining_uncertainty": _nonempty_string(
                    item.get("remaining_uncertainty"),
                    "component-model remaining_uncertainty",
                ),
            }

        aggregate = normalized_components[component_id]
        cell_statuses = {
            row["status"]
            for row in normalized_component_models[component_id].values()
        }
        expected_aggregate_status = (
            "supported"
            if "supported" in cell_statuses
            else "weakened"
            if "weakened" in cell_statuses
            else "unresolved"
            if "unresolved" in cell_statuses
            else "mechanical_failure"
            if "mechanical_failure" in cell_statuses
            else "untested"
        )
        if aggregate["status"] != expected_aggregate_status:
            raise ValueError(
                "aggregate component status differs from its 18x4 cells: "
                f"{component_id}={aggregate['status']} expected "
                f"{expected_aggregate_status}"
            )
        if aggregate["status"] in {"supported", "weakened", "mechanical_failure"}:
            if not aggregate["model_scope"]:
                raise ValueError(
                    "aggregate component claim has empty model scope: "
                    f"{component_id}"
                )
            for model_id in aggregate["model_scope"]:
                cell = normalized_component_models[component_id][model_id]
                if (
                    cell["status"] != aggregate["status"]
                    or cell["evidence_level"] != aggregate["evidence_level"]
                ):
                    raise ValueError(
                        "aggregate component claim differs from its per-model cell: "
                        f"{component_id}/{model_id}"
                    )

    chain_rows = _require_list(decisions, "functional_causal_chain")
    if [row.get("node") for row in chain_rows if isinstance(row, Mapping)] != list(
        CAUSAL_CHAIN_NODES
    ):
        raise ValueError("functional causal chain node order or coverage drift")
    normalized_chain = []
    for row in chain_rows:
        item = _require_mapping(row, "functional causal chain row")
        node = str(item["node"])
        chain_level = _evidence_level(item.get("evidence_level"))
        chain_status = str(item.get("status"))
        chain_model_scope = _model_scope(
            item.get("model_scope"), allow_empty=True
        )
        if chain_status not in CAUSAL_CHAIN_STATUSES:
            raise ValueError(f"invalid functional causal-chain status: {node}")
        if chain_status == "unresolved" and chain_level != "U":
            raise ValueError(f"unresolved causal-chain node must use U: {node}")
        if chain_status == "mechanical_failure" and chain_level != "M":
            raise ValueError(
                f"mechanical-failure causal-chain node must use M: {node}"
            )
        chain_findings = _finding_refs(item, finding_ids)
        if chain_status in {"supported", "weakened"}:
            if chain_level not in {"S", "N", "G"}:
                raise ValueError(
                    f"{chain_status} causal-chain node lacks causal level: {node}"
                )
            level_matched_chain_findings = [
                findings_by_id[finding_id]
                for finding_id in chain_findings
                if findings_by_id[finding_id]["evidence_level"] == chain_level
            ]
            if not level_matched_chain_findings:
                raise ValueError(
                    f"{chain_status} causal-chain node lacks a level-matched finding: "
                    f"{node}"
                )
            required_component_status = (
                "supported" if chain_status == "supported" else "weakened"
            )
            matched_components = [
                normalized_components[component_id]
                for component_id in CAUSAL_CHAIN_COMPONENTS[node]
                if normalized_components[component_id]["status"]
                == required_component_status
                and normalized_components[component_id]["evidence_level"]
                == chain_level
            ]
            if not matched_components:
                raise ValueError(
                    f"{chain_status} causal-chain node lacks a level-matched "
                    f"{required_component_status} component: "
                    f"{node}"
                )
            finding_models = {
                model_id
                for finding in level_matched_chain_findings
                for model_id in finding["model_scope"]
            }
            component_models = {
                model_id
                for component in matched_components
                for model_id in component["model_scope"]
            }
            if not chain_model_scope or not set(chain_model_scope).issubset(
                finding_models & component_models
            ):
                raise ValueError(
                    f"{chain_status} causal-chain model scope exceeds level-matched "
                    f"findings/components: {node}"
                )
            if chain_level == "G" and set(chain_model_scope) != PRIMARY_DESIGN_MODELS:
                raise ValueError(
                    "G-level causal-chain scope must be exactly the two primary "
                    f"design models: {node}"
                )
        normalized_chain.append(
            {
                "node": node,
                "evidence_level": chain_level,
                "status": chain_status,
                "model_scope": chain_model_scope,
                "diagnosis": _nonempty_string(
                    item.get("diagnosis"), "chain diagnosis"
                ),
                "claim_boundary": CAUSAL_CHAIN_CLAIM_BOUNDARIES[node],
                "supporting_findings": chain_findings,
            }
        )

    failure_mode = _require_mapping(
        decisions.get("failure_mode_diagnosis"), "failure_mode_diagnosis"
    )
    primary_mode = str(failure_mode.get("primary_mode"))
    if primary_mode not in FAILURE_MODES:
        raise ValueError(f"unknown primary failure mode: {primary_mode}")
    failure_level = _evidence_level(failure_mode.get("evidence_level"))
    failure_findings = _finding_refs(failure_mode, finding_ids)
    if not failure_findings:
        raise ValueError("failure_mode_diagnosis requires supporting findings")
    failure_model_scope = _model_scope(failure_mode.get("model_scope"))
    referenced_failure_findings = [
        findings_by_id[finding_id] for finding_id in failure_findings
    ]
    level_matched_failure_findings = [
        finding
        for finding in referenced_failure_findings
        if finding["evidence_level"] == failure_level
    ]
    if primary_mode != "unresolved" and failure_level == "U":
        raise ValueError("a resolved failure mode cannot use unresolved evidence")
    if primary_mode != "unresolved" and failure_level == "M":
        raise ValueError("mechanical evidence cannot name a scientific failure mode")
    if primary_mode != "unresolved" and not level_matched_failure_findings:
        raise ValueError(
            "failure mode lacks an evidence-level-matched supporting finding"
        )
    matched_failure_model_scope = {
        model_id
        for finding in level_matched_failure_findings
        for model_id in finding["model_scope"]
    }
    if primary_mode != "unresolved" and not set(failure_model_scope).issubset(
        matched_failure_model_scope
    ):
        raise ValueError("failure-mode model scope exceeds level-matched findings")
    if failure_level == "G" and set(failure_model_scope) != PRIMARY_DESIGN_MODELS:
        raise ValueError(
            "G-level failure mode must cover exactly the two primary design models"
        )
    causal_erasure = failure_mode.get("causal_erasure_claim_authorized")
    causal_loss_of_use = failure_mode.get(
        "causal_loss_of_use_claim_authorized"
    )
    if not isinstance(causal_erasure, bool) or not isinstance(
        causal_loss_of_use, bool
    ):
        raise ValueError("failure-mode causal authorization flags must be booleans")
    if primary_mode == "unresolved" and (
        failure_level != "U" or causal_erasure or causal_loss_of_use
    ):
        raise ValueError("unresolved failure mode must remain level U and non-causal")
    if failure_level in {"M", "D", "S", "U"} and (
        causal_erasure or causal_loss_of_use
    ):
        raise ValueError("causal failure-mode claim requires bidirectional level G")
    if causal_erasure and primary_mode not in {
        "localized_state_attenuation",
        "distributed_state_attenuation",
        "multiple_bottlenecks",
    }:
        raise ValueError("causal erasure flag conflicts with primary failure mode")
    if causal_erasure:
        raise ValueError(
            "causal signal erasure is not authorized: no registered signal-erasure "
            "experiment distinguishes semantic destruction from behavioral "
            "sufficiency attenuation or state mediation"
        )
    if causal_loss_of_use and primary_mode not in {
        "state_present_but_readout_misaligned",
        "multiple_bottlenecks",
    }:
        raise ValueError("causal loss-of-use flag conflicts with primary failure mode")
    if causal_loss_of_use:
        supporting_formal = {
            ref
            for finding in normalized_findings
            if finding["finding_id"] in failure_findings
            for ref in finding["supporting_formal_deliverables"]
        }
        scoped_models = set(failure_model_scope)
        required_readouts = set()
        if MODEL_IDS[2] in scoped_models:
            required_readouts.add("d6_q2_native_readout")
        if MODEL_IDS[3] in scoped_models:
            required_readouts.add("d6_q3_native_readout")
        if not required_readouts or not required_readouts.issubset(supporting_formal):
            raise ValueError(
                "causal loss-of-use claim lacks model-scoped native readout evidence"
            )
        native_readout = normalized_components["native_readout"]
        native_score_chain = next(
            row for row in normalized_chain if row["node"] == "native_score"
        )
        if (
            native_readout["status"] != "supported"
            or not scoped_models.issubset(set(native_readout["model_scope"]))
            or native_score_chain["status"] != "supported"
        ):
            raise ValueError(
                "causal loss-of-use claim requires a model-scoped supported "
                "native-readout component and supported native-score chain"
            )
    failure_components = _component_list(
        failure_mode.get("functional_components"), allow_empty=True
    )
    if primary_mode == "unresolved":
        if failure_components:
            raise ValueError(
                "unresolved failure mode cannot assert functional components"
            )
        component_model_evidence: dict[str, list[str]] = {}
    else:
        if not failure_components:
            raise ValueError(
                "resolved failure mode requires functional components"
            )
        component_model_evidence = {
            component_id: [
                model_id
                for model_id in failure_model_scope
                if any(
                    _finding_matches_component_model(
                        finding,
                        component_id=component_id,
                        model_id=model_id,
                        supplement_metadata=supplement_metadata,
                    )
                    for finding in referenced_failure_findings
                )
            ]
            for component_id in failure_components
        }
        unmatched_components = [
            component_id
            for component_id, model_scope in component_model_evidence.items()
            if not model_scope
        ]
        if unmatched_components:
            raise ValueError(
                "failure mode cites components without component/model-matched "
                "findings: "
                + ", ".join(unmatched_components)
            )
        for model_id in failure_model_scope:
            matched_components = {
                component_id
                for component_id, model_scope in component_model_evidence.items()
                if model_id in model_scope
            }
            if primary_mode == "multiple_bottlenecks":
                matched_layers = {
                    layer_id
                    for layer_id, layer_components in SYSTEM_LAYER_COMPONENTS.items()
                    if matched_components & layer_components
                }
                if len(matched_layers) < 2:
                    raise ValueError(
                        "multiple-bottleneck failure mode lacks two independently "
                        f"evidenced system layers for model: {model_id}"
                    )
            elif not (
                matched_components
                & FAILURE_MODE_REQUIRED_COMPONENTS[primary_mode]
            ):
                raise ValueError(
                    "failure mode lacks a mode/component/model-matched finding: "
                    f"{primary_mode}/{model_id}"
                )
    competing = _require_list(failure_mode, "competing_modes")
    if not competing:
        raise ValueError("failure_mode_diagnosis must retain competing modes")
    normalized_competing = []
    competing_ids: set[str] = set()
    for raw in competing:
        row = _require_mapping(raw, "competing failure mode")
        mode = str(row.get("mode"))
        if mode not in FAILURE_MODES or mode == primary_mode or mode in competing_ids:
            raise ValueError("invalid, duplicate, or primary competing failure mode")
        competing_ids.add(mode)
        normalized_competing.append(
            {
                "mode": mode,
                "reason_remaining": _nonempty_string(
                    row.get("reason_remaining"), "competing mode reason"
                ),
            }
        )
    if failure_mode.get("exact_layer_index_used_for_design") is not False:
        raise ValueError("failure-mode design cannot use an exact layer index")
    normalized_failure_mode = {
        "primary_mode": primary_mode,
        "evidence_level": failure_level,
        "diagnostic_resolution": (
            "unresolved"
            if primary_mode == "unresolved"
            else FAILURE_DIAGNOSTIC_RESOLUTION_BY_LEVEL[failure_level]
        ),
        "summary": _nonempty_string(
            failure_mode.get("summary"), "failure-mode summary"
        ),
        "functional_components": failure_components,
        "component_model_evidence": component_model_evidence,
        "model_scope": failure_model_scope,
        "supporting_findings": failure_findings,
        "competing_modes": normalized_competing,
        "causal_erasure_claim_authorized": causal_erasure,
        "causal_loss_of_use_claim_authorized": causal_loss_of_use,
        "claim_boundary": dict(FAILURE_MODE_CLAIM_BOUNDARY),
        "exact_layer_index_used_for_design": False,
        "falsification_gate": _nonempty_string(
            failure_mode.get("falsification_gate"),
            "failure-mode falsification_gate",
        ),
    }

    system_layers = _exact_mapping(decisions, "system_layers", SYSTEM_LAYER_IDS)
    normalized_layers: dict[str, Any] = {}
    for layer_id in SYSTEM_LAYER_IDS:
        row = _require_mapping(system_layers[layer_id], layer_id)
        layer_status = str(row.get("status"))
        if layer_status not in SYSTEM_LAYER_STATUSES:
            raise ValueError(f"invalid system-layer status: {layer_id}")
        layer_level = _evidence_level(row.get("evidence_level"))
        layer_model_scope = _model_scope(
            row.get("model_scope"), allow_empty=True
        )
        if layer_status == "unresolved" and layer_level != "U":
            raise ValueError(
                f"unresolved system layer must use level U: {layer_id}"
            )
        if layer_status == "mechanical_failure" and layer_level != "M":
            raise ValueError(
                f"mechanical-failure system layer must use level M: {layer_id}"
            )
        if layer_status == "supported" and layer_level not in {"S", "N", "G"}:
            raise ValueError(
                f"supported system layer lacks causal evidence: {layer_id}"
            )
        if layer_status == "weakened" and layer_level not in {"D", "S"}:
            raise ValueError(
                f"weakened system layer lacks registered negative evidence: {layer_id}"
            )
        layer_findings = _finding_refs(row, finding_ids)
        for finding_id in layer_findings:
            finding = findings_by_id[finding_id]
            if not any(
                _finding_matches_component_model(
                    finding,
                    component_id=component_id,
                    model_id=model_id,
                    supplement_metadata=supplement_metadata,
                )
                for component_id in SYSTEM_LAYER_COMPONENTS[layer_id]
                for model_id in set(finding["model_scope"])
                & set(layer_model_scope)
            ):
                raise ValueError(
                    "system-layer diagnosis cites evidence from a different layer: "
                    f"{layer_id}/{finding_id}"
                )
        if layer_findings and not layer_model_scope:
            raise ValueError(
                f"system-layer diagnosis with findings has empty model scope: {layer_id}"
            )
        if layer_status != "unresolved":
            if not layer_findings or not layer_model_scope:
                raise ValueError(
                    f"resolved system layer lacks findings or model scope: {layer_id}"
                )
            for model_id in layer_model_scope:
                if not any(
                    findings_by_id[finding_id]["evidence_level"] == layer_level
                    and _finding_matches_component_model(
                        findings_by_id[finding_id],
                        component_id=component_id,
                        model_id=model_id,
                        supplement_metadata=supplement_metadata,
                    )
                    and normalized_component_models[component_id][model_id][
                        "status"
                    ]
                    == layer_status
                    and normalized_component_models[component_id][model_id][
                        "evidence_level"
                    ]
                    == layer_level
                    and finding_id
                    in normalized_component_models[component_id][model_id][
                        "supporting_findings"
                    ]
                    for finding_id in layer_findings
                    for component_id in SYSTEM_LAYER_COMPONENTS[layer_id]
                ):
                    raise ValueError(
                        "system-layer claim lacks status/level/component/model-matched "
                        f"evidence: {layer_id}/{model_id}"
                    )
        normalized_layers[layer_id] = {
            "functional_components": sorted(SYSTEM_LAYER_COMPONENTS[layer_id]),
            "status": layer_status,
            "evidence_level": layer_level,
            "model_scope": layer_model_scope,
            "diagnosis": _nonempty_string(row.get("diagnosis"), "layer diagnosis"),
            "supporting_findings": layer_findings,
            "remaining_uncertainty": _nonempty_string(
                row.get("remaining_uncertainty"), "layer remaining_uncertainty"
            ),
        }

    model_boundaries = _exact_mapping(decisions, "model_boundaries", MODEL_IDS)
    normalized_models: dict[str, Any] = {}
    for model_id in MODEL_IDS:
        row = _require_mapping(model_boundaries[model_id], model_id)
        model_findings = _finding_refs(row, finding_ids)
        model_has_completed_direct_evidence = any(
            model_id
            in COMPONENT_DELIVERABLE_MODEL_COVERAGE[component_id].get(
                deliverable, set()
            )
            for component_id in COMPONENT_IDS
            for deliverable in completed_formal
            if deliverable in COMPONENT_ALLOWED_DELIVERABLES[component_id]
        ) or any(
            model_id in metadata["model_scope"]
            for metadata in supplement_metadata.values()
        )
        if model_has_completed_direct_evidence and not model_findings:
            raise ValueError(
                "model boundary with completed direct evidence lacks findings: "
                f"{model_id}"
            )
        if any(
            model_id not in findings_by_id[finding_id]["model_scope"]
            for finding_id in model_findings
        ):
            raise ValueError(
                "model boundary cites a finding outside its model scope: "
                f"{model_id}"
            )
        uncovered = _component_list(
            row.get("uncovered_components"), allow_empty=True
        )
        matrix_uncovered = [
            component_id
            for component_id in COMPONENT_IDS
            if normalized_component_models[component_id][model_id]["status"]
            == "untested"
        ]
        if set(uncovered) != set(matrix_uncovered):
            raise ValueError(
                "model uncovered_components differs from the 18x4 matrix: "
                f"{model_id}"
            )
        normalized_models[model_id] = {
            "summary": _nonempty_string(row.get("summary"), "model summary"),
            "supporting_findings": model_findings,
            "uncovered_components": uncovered,
            "do_not_generalize": _nonempty_string(
                row.get("do_not_generalize"), "model do_not_generalize"
            ),
        }

    cross_model = _exact_mapping(
        decisions,
        "cross_model_synthesis",
        (
            "shared_patterns",
            "heterogeneous_patterns",
            "remaining_uncertainty",
            "absolute_index_alignment_used",
        ),
    )
    normalized_cross_model: dict[str, Any] = {}
    pattern_ids: set[str] = set()
    for pattern_kind in ("shared_patterns", "heterogeneous_patterns"):
        raw_patterns = _require_list(cross_model, pattern_kind)
        if not raw_patterns:
            raise ValueError(f"cross_model_synthesis.{pattern_kind} must not be empty")
        normalized_patterns = []
        for raw in raw_patterns:
            row = _require_mapping(raw, f"cross-model {pattern_kind} row")
            pattern_id = _nonempty_string(
                row.get("pattern_id"), "cross-model pattern_id"
            )
            if pattern_id in pattern_ids:
                raise ValueError(f"duplicate cross-model pattern_id: {pattern_id}")
            pattern_ids.add(pattern_id)
            level = _evidence_level(row.get("evidence_level"))
            model_scope = _model_scope(row.get("model_scope"))
            if len(model_scope) < 2:
                raise ValueError(
                    f"cross-model pattern covers fewer than two models: {pattern_id}"
                )
            pattern_findings = _finding_refs(row, finding_ids)
            if not pattern_findings:
                raise ValueError(
                    f"cross-model pattern lacks supporting findings: {pattern_id}"
                )
            functional_components = _component_list(
                row.get("functional_components")
            )
            for model_id in model_scope:
                if not (
                    set(pattern_findings)
                    & set(normalized_models[model_id]["supporting_findings"])
                ):
                    raise ValueError(
                        "cross-model pattern is absent from its model boundary: "
                        f"{pattern_id}/{model_id}"
                    )
                for component_id in functional_components:
                    if not any(
                        findings_by_id[finding_id]["evidence_level"] == level
                        and _finding_matches_component_model(
                            findings_by_id[finding_id],
                            component_id=component_id,
                            model_id=model_id,
                            supplement_metadata=supplement_metadata,
                        )
                        for finding_id in pattern_findings
                    ):
                        raise ValueError(
                            "cross-model pattern lacks component/model/level-matched "
                            f"evidence: {pattern_id}/{component_id}/{model_id}"
                        )
                    component_cell = normalized_component_models[component_id][
                        model_id
                    ]
                    if level in {"S", "N", "G"} and (
                        component_cell["status"] != "supported"
                        or component_cell["evidence_level"] != level
                    ):
                        raise ValueError(
                            "causal cross-model pattern conflicts with its component "
                            f"cell: {pattern_id}/{component_id}/{model_id}"
                        )
                    if level == "M" and (
                        component_cell["status"] != "mechanical_failure"
                        or component_cell["evidence_level"] != "M"
                    ):
                        raise ValueError(
                            "mechanical cross-model pattern conflicts with its component "
                            f"cell: {pattern_id}/{component_id}/{model_id}"
                        )
                    if level == "U" and component_cell["status"] not in {
                        "unresolved",
                        "untested",
                    }:
                        raise ValueError(
                            "unresolved cross-model pattern conflicts with its component "
                            f"cell: {pattern_id}/{component_id}/{model_id}"
                        )
            summary = _nonempty_string(
                row.get("summary"), "cross-model pattern summary"
            )
            do_not_generalize = _nonempty_string(
                row.get("do_not_generalize"),
                "cross-model pattern do_not_generalize",
            )
            if _ABSOLUTE_INTERNAL_INDEX.search(
                summary
            ) or _ABSOLUTE_INTERNAL_INDEX.search(do_not_generalize):
                raise ValueError(
                    f"absolute internal index forbidden in cross-model pattern: {pattern_id}"
                )
            normalized_patterns.append(
                {
                    "pattern_id": pattern_id,
                    "evidence_level": level,
                    "model_scope": model_scope,
                    "functional_components": functional_components,
                    "summary": summary,
                    "supporting_findings": pattern_findings,
                    "do_not_generalize": do_not_generalize,
                }
            )
        normalized_cross_model[pattern_kind] = normalized_patterns
    if cross_model.get("absolute_index_alignment_used") is not False:
        raise ValueError("cross-model synthesis cannot align by absolute index")
    normalized_cross_model.update(
        {
            "remaining_uncertainty": _nonempty_string(
                cross_model.get("remaining_uncertainty"),
                "cross-model remaining_uncertainty",
            ),
            "absolute_index_alignment_used": False,
        }
    )

    hypotheses = _exact_mapping(decisions, "hypothesis_matrix", HYPOTHESIS_IDS)
    normalized_hypotheses: dict[str, Any] = {}
    for hypothesis_id in HYPOTHESIS_IDS:
        row = _require_mapping(hypotheses[hypothesis_id], hypothesis_id)
        status = str(row.get("status"))
        if status not in HYPOTHESIS_STATUSES:
            raise ValueError(f"invalid hypothesis status: {hypothesis_id}={status}")
        hypothesis_level = _evidence_level(row.get("evidence_level"))
        if status in {"supported", "rejected"} and hypothesis_level not in {
            "S",
            "N",
            "G",
        }:
            raise ValueError(
                f"decisive hypothesis status lacks causal evidence level: {hypothesis_id}"
            )
        hypothesis_findings = _finding_refs(row, finding_ids)
        if status == "weakened":
            if hypothesis_level not in {"D", "S"}:
                raise ValueError(
                    "weakened hypothesis lacks a registered negative evidence level: "
                    f"{hypothesis_id}"
                )
            allowed_formal = set(HYPOTHESIS_ALLOWED_DELIVERABLES[hypothesis_id])
            if not any(
                findings_by_id[finding_id]["evidence_level"] == hypothesis_level
                and bool(
                    set(
                        findings_by_id[finding_id][
                            "supporting_formal_deliverables"
                        ]
                    )
                    & allowed_formal
                )
                for finding_id in hypothesis_findings
            ):
                raise ValueError(
                    "weakened hypothesis lacks level- and hypothesis-matched "
                    f"registered evidence: {hypothesis_id}"
                )
        if status in {"supported", "rejected"}:
            if hypothesis_id == "H5":
                raise ValueError(
                    "H5 cannot receive a decisive status without the unregistered "
                    "independent seed"
                )
            if hypothesis_level == "N":
                raise ValueError(
                    "component-state reverse necessity cannot by itself decide H0--H5"
                )
            if status == "rejected" and hypothesis_level != "S":
                raise ValueError(
                    "only the admitted formal S-level outcome can reject H0--H5; "
                    "a design-qualified component gate is not hypothesis refutation"
                )
            if not hypothesis_findings:
                raise ValueError(
                    f"decisive hypothesis lacks supporting findings: {hypothesis_id}"
                )
            referenced_findings = [
                findings_by_id[finding_id] for finding_id in hypothesis_findings
            ]
            allowed_formal = set(HYPOTHESIS_ALLOWED_DELIVERABLES[hypothesis_id])
            level_matched = [
                finding
                for finding in referenced_findings
                if finding["evidence_level"] == hypothesis_level
                and bool(
                    set(finding["supporting_formal_deliverables"])
                    & allowed_formal
                )
                and (
                    hypothesis_level != "G"
                    or DESIGN_GATE_SUPPLEMENT
                    in finding["supporting_supplements"]
                )
            ]
            if not level_matched:
                raise ValueError(
                    "decisive hypothesis lacks level- and hypothesis-matched "
                    f"evidence: {hypothesis_id}"
                )
            if status == "supported":
                cited_formal = {
                    deliverable
                    for finding in referenced_findings
                    for deliverable in finding["supporting_formal_deliverables"]
                }
                required_groups = HYPOTHESIS_SUPPORTED_REQUIRED_EVIDENCE_GROUPS[
                    hypothesis_id
                ]
                if not required_groups or any(
                    not (set(group) & cited_formal) for group in required_groups
                ):
                    raise ValueError(
                        "supported hypothesis lacks a preregistered evidence group: "
                        f"{hypothesis_id}"
                    )
                required_components = HYPOTHESIS_SUPPORTED_COMPONENT_REQUIREMENTS.get(
                    hypothesis_id, ()
                )
                if any(
                    normalized_components[component_id]["status"] != "supported"
                    for component_id in required_components
                ):
                    raise ValueError(
                        "supported hypothesis lacks its required supported components: "
                        f"{hypothesis_id}"
                    )
        negative_bases = _bounded_string_scope(
            row.get("negative_evidence_basis"),
            label="hypothesis negative_evidence_basis",
            allowed=HYPOTHESIS_NEGATIVE_EVIDENCE_BASES,
        )
        if status == "rejected" and "registered_refutation" not in negative_bases:
            raise ValueError(
                f"rejected hypothesis lacks registered_refutation basis: {hypothesis_id}"
            )
        if status == "weakened" and not (
            set(negative_bases)
            & {
                "registered_weakening",
                "cross_model_or_endpoint_conflict",
                "measurement_population_instability",
            }
        ):
            raise ValueError(
                f"weakened hypothesis lacks a negative evidence basis: {hypothesis_id}"
            )
        if status in {"supported", "unresolved"} and (
            "registered_refutation" in negative_bases
        ):
            raise ValueError(
                f"non-rejected hypothesis claims a refutation basis: {hypothesis_id}"
            )
        normalized_hypotheses[hypothesis_id] = {
            "status": status,
            "evidence_level": hypothesis_level,
            "summary": _nonempty_string(row.get("summary"), "hypothesis summary"),
            "supporting_findings": hypothesis_findings,
            "negative_evidence_basis": negative_bases,
            "contradictory_evidence": _string_list(
                row.get("contradictory_evidence"),
                "hypothesis contradictory_evidence",
            ),
            "remaining_uncertainty": _nonempty_string(
                row.get("remaining_uncertainty"), "hypothesis remaining_uncertainty"
            ),
        }

    negatives = _require_list(decisions, "negative_and_conflicting_results")
    if not negatives:
        raise ValueError("negative_and_conflicting_results must not be empty")
    normalized_negatives = []
    negative_ids: set[str] = set()
    for raw in negatives:
        row = _require_mapping(raw, "negative/conflicting result")
        result_id = _nonempty_string(row.get("result_id"), "negative result_id")
        if result_id in negative_ids:
            raise ValueError(f"duplicate negative result_id: {result_id}")
        negative_ids.add(result_id)
        negative_findings = _finding_refs(row, finding_ids)
        if not negative_findings:
            raise ValueError(
                f"negative/conflicting result lacks supporting findings: {result_id}"
            )
        negative_model_scope = _model_scope(row.get("model_scope"))
        finding_models = {
            model_id
            for finding_id in negative_findings
            for model_id in findings_by_id[finding_id]["model_scope"]
        }
        if not set(negative_model_scope).issubset(finding_models):
            raise ValueError(
                f"negative/conflicting result exceeds finding model scope: {result_id}"
            )
        endpoint_scope = _bounded_string_scope(
            row.get("endpoint_scope"),
            label="negative endpoint_scope",
            allowed=NEGATIVE_ENDPOINT_SCOPES,
        )
        surface_scope = _bounded_string_scope(
            row.get("surface_scope"),
            label="negative surface_scope",
            allowed=NEGATIVE_SURFACE_SCOPES,
        )
        contrast_scope = _bounded_string_scope(
            row.get("contrast_scope"),
            label="negative contrast_scope",
            allowed=NEGATIVE_CONTRAST_SCOPES,
        )
        fold_scope = _bounded_string_scope(
            row.get("fold_scope"),
            label="negative fold_scope",
            allowed=NEGATIVE_FOLD_SCOPES,
        )
        seed_scope = _string_list(row.get("seed_scope"), "negative seed_scope")
        if len(seed_scope) != len(set(seed_scope)):
            raise ValueError("negative seed_scope contains duplicates")
        normalized_negatives.append(
            {
                "result_id": result_id,
                "summary": _nonempty_string(row.get("summary"), "negative summary"),
                "model_scope": negative_model_scope,
                "endpoint_scope": endpoint_scope,
                "surface_scope": surface_scope,
                "contrast_scope": contrast_scope,
                "fold_scope": fold_scope,
                "seed_scope": seed_scope,
                "supporting_findings": negative_findings,
                "interpretation_boundary": _nonempty_string(
                    row.get("interpretation_boundary"),
                    "negative interpretation_boundary",
                ),
            }
        )
    retained_negative_findings = {
        finding_id
        for row in normalized_negatives
        for finding_id in row["supporting_findings"]
    }
    retained_negative_findings_by_model = {
        model_id: {
            finding_id
            for row in normalized_negatives
            if model_id in row["model_scope"]
            for finding_id in row["supporting_findings"]
        }
        for model_id in MODEL_IDS
    }
    for component_id, row in normalized_components.items():
        if row["status"] == "weakened" and not (
            set(row["supporting_findings"]) & retained_negative_findings
        ):
            raise ValueError(
                "weakened component is missing from negative/conflicting results: "
                f"{component_id}"
            )
    for hypothesis_id, row in normalized_hypotheses.items():
        if row["status"] in {"weakened", "rejected"} and not (
            set(row["supporting_findings"]) & retained_negative_findings
        ):
            raise ValueError(
                "negative hypothesis outcome is missing from negative/conflicting "
                f"results: {hypothesis_id}"
            )
    for component_id, model_rows in normalized_component_models.items():
        for model_id, row in model_rows.items():
            if row["status"] == "weakened" and not (
                set(row["supporting_findings"])
                & retained_negative_findings_by_model[model_id]
            ):
                raise ValueError(
                    "weakened component-model cell is missing from a model-scoped "
                    "negative/conflicting result: "
                    f"{component_id}/{model_id}"
                )

    expected_evidence_ids = tuple(
        sorted(set(completed_formal) | set(completed_supplements))
    )
    disposition_rows = _exact_mapping(
        decisions,
        "evidence_disposition",
        expected_evidence_ids,
    )
    normalized_disposition: dict[str, Any] = {}
    for evidence_id in expected_evidence_ids:
        row = _require_mapping(
            disposition_rows[evidence_id],
            f"evidence_disposition.{evidence_id}",
        )
        expected_kind = (
            "formal_deliverable"
            if evidence_id in completed_formal
            else "supplement"
        )
        if row.get("evidence_kind") != expected_kind:
            raise ValueError(
                f"evidence disposition kind differs: {evidence_id}"
            )
        disposition = str(row.get("disposition"))
        if disposition not in EVIDENCE_DISPOSITIONS:
            raise ValueError(
                f"invalid evidence disposition: {evidence_id}={disposition}"
            )
        disposition_findings = _finding_refs(row, finding_ids)
        if (
            disposition == "bounded_no_scientific_claim"
            and disposition_findings
        ):
            raise ValueError(
                "bounded evidence disposition cannot cite a scientific finding: "
                f"{evidence_id}"
            )
        all_citing_findings = [
            finding["finding_id"]
            for finding in normalized_findings
            if (
                evidence_id in finding["supporting_formal_deliverables"]
                if expected_kind == "formal_deliverable"
                else evidence_id in finding["supporting_supplements"]
            )
        ]
        if disposition == "bounded_no_scientific_claim" and all_citing_findings:
            raise ValueError(
                "evidence marked as bounded without a scientific claim is cited by "
                f"a finding: {evidence_id}"
            )
        exact_evidence_findings = []
        for finding_id in disposition_findings:
            finding = findings_by_id[finding_id]
            cited = (
                evidence_id in finding["supporting_formal_deliverables"]
                if expected_kind == "formal_deliverable"
                else evidence_id in finding["supporting_supplements"]
            )
            if not cited:
                raise ValueError(
                    "evidence disposition cites a finding that does not cite the "
                    f"same evidence: {evidence_id}/{finding_id}"
                )
            exact_evidence_findings.append(finding_id)
        if (
            disposition != "bounded_no_scientific_claim"
            and not exact_evidence_findings
        ):
            raise ValueError(
                f"evidence disposition lacks an exact supporting finding: {evidence_id}"
            )
        if disposition == "negative_or_conflicting" and not (
            set(exact_evidence_findings) & retained_negative_findings
        ):
            raise ValueError(
                "negative evidence disposition is absent from the registered "
                f"negative/conflicting table: {evidence_id}"
            )
        normalized_disposition[evidence_id] = {
            "evidence_id": evidence_id,
            "evidence_kind": expected_kind,
            "disposition": disposition,
            "supporting_findings": exact_evidence_findings,
            "summary": _nonempty_string(
                row.get("summary"),
                f"evidence disposition summary: {evidence_id}",
            ),
            "do_not_infer": _string_list(
                row.get("do_not_infer"),
                f"evidence disposition do_not_infer: {evidence_id}",
            ),
            "scientific_claim_emitted": disposition
            != "bounded_no_scientific_claim",
        }

    opportunities = _require_list(decisions, "optimization_opportunities")
    if not opportunities:
        raise ValueError("optimization_opportunities must not be empty")
    normalized_opportunities = []
    opportunity_ids: set[str] = set()
    formal_opportunity_targets: dict[str, dict[str, str]] = {}
    previous_priority_order = -1
    previous_opportunity_rank_key: tuple[int, int, int] | None = None
    for rank, raw in enumerate(opportunities, start=1):
        row = _require_mapping(raw, "optimization opportunity")
        opportunity_id = _nonempty_string(row.get("opportunity_id"), "opportunity_id")
        if opportunity_id in opportunity_ids:
            raise ValueError(f"duplicate opportunity_id: {opportunity_id}")
        opportunity_ids.add(opportunity_id)
        formal_predecessors = _bounded_string_scope(
            row.get("formal_predecessor_ids"),
            label="opportunity formal_predecessor_ids",
            allowed=set(OPPORTUNITY_IDS),
            allow_empty=True,
        )
        for formal_id in formal_predecessors:
            if formal_id in formal_opportunity_targets:
                raise ValueError(
                    f"formal opportunity is mapped more than once: {formal_id}"
                )
            formal_opportunity_targets[formal_id] = {
                "disposition": "mapped_to_comprehensive_opportunity",
                "target_id": opportunity_id,
            }
        component = _nonempty_string(
            row.get("functional_component"), "functional_component"
        )
        if component not in COMPONENT_IDS:
            raise ValueError(f"unknown opportunity component: {component}")
        mechanism_target = _nonempty_string(
            row.get("mechanism_target"), "mechanism_target"
        )
        reason = _nonempty_string(row.get("reason"), "opportunity reason")
        expected_benefit = _nonempty_string(
            row.get("expected_benefit"), "opportunity expected_benefit"
        )
        hypothesized_innovation = _nonempty_string(
            row.get("hypothesized_innovation"),
            "opportunity hypothesized_innovation",
        )
        training_signal_requirements = _string_list(
            row.get("training_signal_requirements"),
            "opportunity training_signal_requirements",
        )
        key_ablations = _string_list(
            row.get("key_ablations"), "opportunity key_ablations"
        )
        closest_baseline_families = _string_list(
            row.get("closest_baseline_families"),
            "opportunity closest_baseline_families",
        )
        baseline_differentiation = _nonempty_string(
            row.get("baseline_differentiation"),
            "opportunity baseline_differentiation",
        )
        key_risks = _string_list(row.get("key_risks"), "opportunity key_risks")
        for label, value in (
            ("opportunity_id", opportunity_id),
            ("functional_component", component),
            ("mechanism_target", mechanism_target),
            ("reason", reason),
            ("expected_benefit", expected_benefit),
            ("hypothesized_innovation", hypothesized_innovation),
            ("baseline_differentiation", baseline_differentiation),
            *(
                ("training_signal_requirement", requirement)
                for requirement in training_signal_requirements
            ),
            *(("key_ablation", ablation) for ablation in key_ablations),
            *(
                ("closest_baseline_family", baseline)
                for baseline in closest_baseline_families
            ),
            *(("key_risk", risk) for risk in key_risks),
        ):
            if _ABSOLUTE_INTERNAL_INDEX.search(value):
                raise ValueError(f"absolute internal index forbidden in {label}")
        priority = str(row.get("design_priority"))
        if priority not in DESIGN_PRIORITIES:
            raise ValueError(f"invalid design_priority: {priority}")
        priority_order = DESIGN_PRIORITY_ORDER[priority]
        if priority_order < previous_priority_order:
            raise ValueError(
                "optimization opportunities must be ordered design-qualified, "
                "candidate-to-test, deprioritized, then not-recommended"
            )
        previous_priority_order = priority_order
        intervention_polarity = str(row.get("intervention_polarity"))
        if intervention_polarity not in INTERVENTION_POLARITIES:
            raise ValueError(
                f"invalid opportunity intervention_polarity: {intervention_polarity}"
            )
        actual_level = _evidence_level(row.get("actual_evidence_level"))
        minimum_level = _evidence_level(row.get("minimum_evidence_level"))
        if intervention_polarity == "preserve_or_strengthen_beneficial_state":
            raise ValueError(
                "mechanism-stage opportunity cannot preserve or strengthen a state "
                "without established component benefit"
            )
        if actual_level == "M" and (
            priority not in {"deprioritized", "not_recommended"}
            or intervention_polarity != "diagnostic_only"
        ):
            raise ValueError(
                "mechanical evidence cannot rank a scientific optimization candidate"
            )
        supplement_refs = _reference_list(
            row, "supporting_supplements", completed_supplements
        )
        formal_refs = _reference_list(
            row, "supporting_formal_deliverables", completed_formal
        )
        for formal_id in formal_predecessors:
            if not (
                set(formal_refs) & OPPORTUNITY_ALLOWED_DELIVERABLES[formal_id]
            ):
                raise ValueError(
                    "formal opportunity predecessor lacks mechanism-family evidence "
                    f"in its comprehensive target: {formal_id}/{opportunity_id}"
                )
        if not supplement_refs and not formal_refs:
            raise ValueError(f"opportunity {opportunity_id} has no evidence")
        matched_formal_refs = [
            ref
            for ref in formal_refs
            if ref in COMPONENT_ALLOWED_DELIVERABLES[component]
        ]
        matched_supplement_refs = [
            ref
            for ref in supplement_refs
            if component in set(supplement_metadata[ref]["components"])
        ]
        if not matched_formal_refs and not matched_supplement_refs:
            raise ValueError(
                "optimization opportunity lacks component-matched evidence: "
                f"{opportunity_id}"
            )
        functional_node = row.get("functional_node")
        if functional_node is not None:
            functional_node = _nonempty_string(functional_node, "functional_node")
            if functional_node not in DESIGN_NODE_COMPONENTS:
                raise ValueError(f"unknown functional_node: {functional_node}")
        if priority == "design_qualified":
            if actual_level != "G":
                raise ValueError("design-qualified opportunity requires evidence level G")
            if DESIGN_GATE_SUPPLEMENT not in supplement_refs:
                raise ValueError("design-qualified opportunity lacks design-gate supplement")
            if functional_node not in design_qualified_nodes:
                raise ValueError("opportunity node did not pass cross-model design gate")
            if component not in DESIGN_NODE_COMPONENTS[str(functional_node)]:
                raise ValueError("opportunity component does not match qualified node")
        elif actual_level == "G":
            raise ValueError("evidence level G is reserved for design-qualified entries")
        if priority == "candidate_to_test" and actual_level not in {"D", "S", "N"}:
            raise ValueError(
                "candidate-to-test opportunity requires D, S, or N evidence"
            )
        if actual_level == "S" and not (
            set(matched_formal_refs)
            & set(COMPONENT_SUPPORT_REQUIRED_CAUSAL_EVIDENCE[component])
        ):
            raise ValueError(
                "sufficiency-level opportunity lacks component causal evidence"
            )
        if actual_level == "N" and (
            component not in NECESSITY_COMPONENTS
            or NECESSITY_SUPPLEMENT not in matched_supplement_refs
            or not set(_model_scope(row.get("model_scope"))).issubset(
                set(necessity_supported_component_models[component])
            )
        ):
            raise ValueError(
                "necessity-level opportunity lacks component necessity evidence"
            )
        if (
            actual_level in {"N", "G"}
            and intervention_polarity
            not in HARM_MEDIATOR_INTERVENTION_POLARITIES
        ):
            raise ValueError(
                "N/G harmful-mediator opportunity must suppress, reroute, or "
                "recalibrate rather than strengthen or remain diagnostic-only"
            )
        opportunity_model_scope = _model_scope(row.get("model_scope"))
        opportunity_rank_key = (
            priority_order,
            -OPPORTUNITY_EVIDENCE_STRENGTH[actual_level],
            -len(opportunity_model_scope),
        )
        if (
            previous_opportunity_rank_key is not None
            and opportunity_rank_key < previous_opportunity_rank_key
        ):
            raise ValueError(
                "optimization opportunities must be ordered by priority, evidence "
                "strength, then model coverage"
            )
        previous_opportunity_rank_key = opportunity_rank_key
        if actual_level == "G" and set(opportunity_model_scope) != PRIMARY_DESIGN_MODELS:
            raise ValueError(
                "G-level opportunity must cover exactly the two primary design models"
            )
        if actual_level in {"S", "N", "G"}:
            component_claim = normalized_components[component]
            if (
                component_claim["status"] != "supported"
                or component_claim["evidence_level"] != actual_level
                or not set(opportunity_model_scope).issubset(
                    set(component_claim["model_scope"])
                )
            ):
                raise ValueError(
                    "causal opportunity conflicts with the component evidence matrix: "
                    f"{opportunity_id}/{component}"
                )
        opportunity_findings = _finding_refs(row, finding_ids)
        if not opportunity_findings:
            raise ValueError(
                f"optimization opportunity lacks supporting findings: {opportunity_id}"
            )
        matched_evidence_models = {
            model_id
            for ref in matched_formal_refs
            for model_id in COMPONENT_DELIVERABLE_MODEL_COVERAGE[component][ref]
        } | {
            model_id
            for ref in matched_supplement_refs
            for model_id in supplement_metadata[ref]["model_scope"]
        }
        if not set(opportunity_model_scope).issubset(matched_evidence_models):
            raise ValueError(
                "optimization opportunity model scope exceeds component evidence"
            )
        finding_formal_refs = {
            evidence_id
            for finding_id in opportunity_findings
            for evidence_id in findings_by_id[finding_id][
                "supporting_formal_deliverables"
            ]
        }
        finding_supplement_refs = {
            evidence_id
            for finding_id in opportunity_findings
            for evidence_id in findings_by_id[finding_id]["supporting_supplements"]
        }
        if not set(formal_refs).issubset(finding_formal_refs) or not set(
            supplement_refs
        ).issubset(finding_supplement_refs):
            raise ValueError(
                "optimization opportunity cites evidence not interpreted by its "
                f"supporting findings: {opportunity_id}"
            )
        for model_id in opportunity_model_scope:
            if not any(
                findings_by_id[finding_id]["evidence_level"] == actual_level
                and model_id in findings_by_id[finding_id]["model_scope"]
                and _finding_matches_component_model(
                    findings_by_id[finding_id],
                    component_id=component,
                    model_id=model_id,
                    supplement_metadata=supplement_metadata,
                )
                for finding_id in opportunity_findings
            ):
                raise ValueError(
                    "optimization opportunity lacks evidence-level/component/model-"
                    f"matched findings: {opportunity_id}/{model_id}"
                )
        if row.get("source_test_opened") is not False:
            raise ValueError("opportunity must preserve source_test_opened=false")
        if row.get("utility_gain_established") is not False:
            raise ValueError(
                "diagnostic mechanism evidence cannot establish opportunity utility gain"
            )
        if row.get("diagnostic_patch_promoted_as_method") is not False:
            raise ValueError("diagnostic patch cannot be promoted as the method")
        if row.get("architecture_implemented") is not False:
            raise ValueError(
                "mechanism-stage opportunity cannot claim an implemented architecture"
            )
        normalized_opportunities.append(
            {
                "opportunity_id": opportunity_id,
                "rank": rank,
                "formal_predecessor_ids": formal_predecessors,
                "functional_component": component,
                "functional_node": functional_node,
                "mechanism_target": mechanism_target,
                "minimum_evidence_level": minimum_level,
                "actual_evidence_level": actual_level,
                "supporting_formal_deliverables": formal_refs,
                "supporting_supplements": supplement_refs,
                "supporting_findings": opportunity_findings,
                "contradictory_evidence": _string_list(
                    row.get("contradictory_evidence"),
                    "opportunity contradictory_evidence",
                ),
                "model_scope": opportunity_model_scope,
                "dataset_scope": _dataset_scope(row.get("dataset_scope")),
                "source_test_opened": False,
                "utility_gain_established": False,
                "design_priority": priority,
                "evidence_strength_tier": OPPORTUNITY_EVIDENCE_STRENGTH[
                    actual_level
                ],
                "ranking_basis": (
                    "design_priority_then_G_over_SN_over_D_over_U_over_M_then_"
                    "model_coverage"
                ),
                "intervention_polarity": intervention_polarity,
                "reason": reason,
                "expected_benefit": expected_benefit,
                "hypothesized_innovation": hypothesized_innovation,
                "training_signal_requirements": training_signal_requirements,
                "key_ablations": key_ablations,
                "closest_baseline_families": closest_baseline_families,
                "baseline_differentiation": baseline_differentiation,
                "key_risks": key_risks,
                "falsification_gate": _nonempty_string(
                    row.get("falsification_gate"), "falsification_gate"
                ),
                "do_not_infer": _string_list(
                    row.get("do_not_infer"), "opportunity do_not_infer"
                ),
                "diagnostic_patch_promoted_as_method": False,
                "architecture_implemented": False,
            }
        )

    not_recommended = _require_list(decisions, "not_recommended")
    if not not_recommended:
        raise ValueError("not_recommended must not be empty")
    normalized_not_recommended = []
    for raw in not_recommended:
        row = _require_mapping(raw, "not_recommended row")
        basis = str(row.get("basis"))
        if basis not in NOT_RECOMMENDED_BASES:
            raise ValueError(f"invalid not_recommended basis: {basis}")
        component = _nonempty_string(
            row.get("functional_component"),
            "not_recommended functional_component",
        )
        if component not in COMPONENT_IDS:
            raise ValueError(
                f"unknown not_recommended component: {component}"
            )
        model_scope = _model_scope(row.get("model_scope"))
        basis_findings = _finding_refs(row, finding_ids)
        level_matched_findings = [
            finding_id
            for finding_id in basis_findings
            if findings_by_id[finding_id]["evidence_level"]
            in NOT_RECOMMENDED_BASIS_LEVELS[basis]
        ]
        if not level_matched_findings:
            raise ValueError(
                "not_recommended basis lacks a matching evidence level: "
                f"{basis}"
            )
        for model_id in model_scope:
            if not any(
                model_id in findings_by_id[finding_id]["model_scope"]
                and _finding_matches_component_model(
                    findings_by_id[finding_id],
                    component_id=component,
                    model_id=model_id,
                    supplement_metadata=supplement_metadata,
                )
                for finding_id in level_matched_findings
            ):
                raise ValueError(
                    "not_recommended direction lacks component/model-matched "
                    f"evidence: {component}/{model_id}"
                )
            component_cell = normalized_component_models[component][model_id]
            if basis == "registered_refutation" and (
                component_cell["status"] != "weakened"
                or component_cell["evidence_level"] != "S"
                or not (
                    set(level_matched_findings)
                    & retained_negative_findings_by_model[model_id]
                )
            ):
                raise ValueError(
                    "registered-refutation recommendation lacks a weakened "
                    f"component-model outcome retained as negative: {component}/{model_id}"
                )
            if basis == "mechanical_non_result" and (
                component_cell["status"] != "mechanical_failure"
                or component_cell["evidence_level"] != "M"
            ):
                raise ValueError(
                    "mechanical-non-result recommendation lacks a mechanical-failure "
                    f"component-model outcome: {component}/{model_id}"
                )
            if basis == "insufficient_causal_evidence" and (
                component_cell["status"] == "supported"
                and component_cell["evidence_level"] == "G"
            ):
                raise ValueError(
                    "insufficient-causal-evidence basis conflicts with a G-level "
                    f"component-model outcome: {component}/{model_id}"
                )
        if row.get("source_test_opened") is not False:
            raise ValueError(
                "not_recommended direction must preserve source_test_opened=false"
            )
        formal_predecessors = _bounded_string_scope(
            row.get("formal_predecessor_ids"),
            label="not_recommended formal_predecessor_ids",
            allowed=set(OPPORTUNITY_IDS),
            allow_empty=True,
        )
        direction = _nonempty_string(row.get("direction"), "direction")
        basis_formal_evidence = {
            evidence_id
            for finding_id in level_matched_findings
            for evidence_id in findings_by_id[finding_id][
                "supporting_formal_deliverables"
            ]
        }
        for formal_id in formal_predecessors:
            if not (
                basis_formal_evidence
                & OPPORTUNITY_ALLOWED_DELIVERABLES[formal_id]
            ):
                raise ValueError(
                    "formal opportunity predecessor lacks mechanism-family evidence "
                    f"in its not-recommended target: {formal_id}/{direction}"
                )
            if formal_id in formal_opportunity_targets:
                raise ValueError(
                    f"formal opportunity is mapped more than once: {formal_id}"
                )
            formal_opportunity_targets[formal_id] = {
                "disposition": "mapped_to_not_recommended",
                "target_id": direction,
            }
        normalized_not_recommended.append(
            {
                "direction": direction,
                "formal_predecessor_ids": formal_predecessors,
                "functional_component": component,
                "reason": _nonempty_string(row.get("reason"), "reason"),
                "supporting_findings": basis_findings,
                "basis": basis,
                "model_scope": model_scope,
                "dataset_scope": _dataset_scope(row.get("dataset_scope")),
                "source_test_opened": False,
            }
        )

    if set(formal_opportunity_targets) != set(OPPORTUNITY_IDS):
        missing = sorted(set(OPPORTUNITY_IDS) - set(formal_opportunity_targets))
        raise ValueError(
            "formal opportunity disposition coverage differs; missing: "
            + ", ".join(missing)
        )

    normalized = {
        "worksheet_status": "final",
        "report_id": report_id,
        "narratives": normalized_narratives,
        "findings": normalized_findings,
        "component_matrix": normalized_components,
        "component_model_matrix": normalized_component_models,
        "functional_causal_chain": normalized_chain,
        "failure_mode_diagnosis": normalized_failure_mode,
        "system_layers": normalized_layers,
        "model_boundaries": normalized_models,
        "cross_model_synthesis": normalized_cross_model,
        "hypothesis_matrix": normalized_hypotheses,
        "negative_and_conflicting_results": normalized_negatives,
        "evidence_disposition": normalized_disposition,
        "optimization_opportunities": normalized_opportunities,
        "not_recommended": normalized_not_recommended,
        "formal_opportunity_disposition": {
            formal_id: formal_opportunity_targets[formal_id]
            for formal_id in OPPORTUNITY_IDS
        },
    }
    _audit_human_interpretation_text(normalized)
    return normalized


def build_comprehensive_decision_template(
    *, report_id: str = "motivation_transformer_comprehensive_v1"
) -> dict[str, Any]:
    """Return an exhaustive, deliberately non-final human worksheet skeleton."""

    required = "__REQUIRED_FROM_ADMITTED_EVIDENCE__"
    return {
        "schema_version": 1,
        "worksheet_status": "incomplete",
        "report_id": report_id,
        "narratives": {
            key: {
                "text": required,
                "evidence_level": "U",
                "supporting_findings": [],
                "do_not_infer": [required],
            }
            for key in REQUIRED_NARRATIVES
        },
        "findings": [],
        "component_matrix": {
            component_id: {
                "status": "unresolved",
                "evidence_level": "U",
                "summary": required,
                "model_scope": [],
                "supporting_findings": [],
                "remaining_uncertainty": required,
            }
            for component_id in COMPONENT_IDS
        },
        "component_model_matrix": {
            component_id: {
                model_id: {
                    "status": "unresolved",
                    "evidence_level": "U",
                    "summary": required,
                    "supporting_findings": [],
                    "remaining_uncertainty": required,
                }
                for model_id in MODEL_IDS
            }
            for component_id in COMPONENT_IDS
        },
        "functional_causal_chain": [
            {
                "node": node,
                "evidence_level": "U",
                "status": "unresolved",
                "model_scope": [],
                "diagnosis": required,
                "supporting_findings": [],
            }
            for node in CAUSAL_CHAIN_NODES
        ],
        "failure_mode_diagnosis": {
            "primary_mode": "unresolved",
            "evidence_level": "U",
            "summary": required,
            "functional_components": [],
            "model_scope": list(MODEL_IDS),
            "supporting_findings": [],
            "competing_modes": [
                {
                    "mode": "candidate_transport_failure",
                    "reason_remaining": required,
                },
                {
                    "mode": "state_present_but_readout_misaligned",
                    "reason_remaining": required,
                },
            ],
            "causal_erasure_claim_authorized": False,
            "causal_loss_of_use_claim_authorized": False,
            "exact_layer_index_used_for_design": False,
            "falsification_gate": required,
        },
        "system_layers": {
            layer_id: {
                "status": "unresolved",
                "evidence_level": "U",
                "model_scope": [],
                "diagnosis": required,
                "supporting_findings": [],
                "remaining_uncertainty": required,
            }
            for layer_id in SYSTEM_LAYER_IDS
        },
        "model_boundaries": {
            model_id: {
                "summary": required,
                "supporting_findings": [],
                "uncovered_components": list(COMPONENT_IDS),
                "do_not_generalize": required,
            }
            for model_id in MODEL_IDS
        },
        "cross_model_synthesis": {
            "shared_patterns": [],
            "heterogeneous_patterns": [],
            "remaining_uncertainty": required,
            "absolute_index_alignment_used": False,
        },
        "hypothesis_matrix": {
            hypothesis_id: {
                "status": "unresolved",
                "evidence_level": "U",
                "summary": required,
                "supporting_findings": [],
                "negative_evidence_basis": ["insufficient_causal_evidence"],
                "contradictory_evidence": [required],
                "remaining_uncertainty": required,
            }
            for hypothesis_id in HYPOTHESIS_IDS
        },
        "negative_and_conflicting_results": [],
        "evidence_disposition": {
            evidence_id: {
                "evidence_kind": (
                    "formal_deliverable"
                    if evidence_id in set(EXPECTED_DELIVERABLES)
                    else "supplement"
                ),
                "disposition": "bounded_no_scientific_claim",
                "supporting_findings": [],
                "summary": required,
                "do_not_infer": [required],
            }
            for evidence_id in sorted(
                set(EXPECTED_DELIVERABLES) | set(EXPECTED_SUPPLEMENT_IDS)
            )
        },
        "optimization_opportunities": [],
        "not_recommended": [],
        "template_instructions": {
            "set_worksheet_status_to_final_only_after_all_placeholders_are_replaced": True,
            "absolute_layer_head_or_neuron_forbidden_in_design": True,
            "source_test_must_remain_closed": True,
            "metrics_must_be_copied_from_admitted_evaluator_outputs": True,
            "cross_model_patterns_require_component_model_level_matched_findings": True,
            "cross_model_alignment_uses_functional_components_not_absolute_indices": True,
            "each_opportunity_requires_training_signal_ablations_and_baseline_differentiation": True,
            "opportunities_are_hypotheses_not_implemented_architectures": True,
            "every_formal_and_supplemental_evidence_item_requires_an_explicit_disposition": True,
        },
    }


def populate_registered_component_model_coverage(
    template: Mapping[str, Any],
    *,
    registered_formal: set[str],
    registered_supplements: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Prefill only outcome-independent 18x4 coverage in a worksheet template."""

    value = copy.deepcopy(dict(template))
    component_matrix = _exact_mapping(value, "component_matrix", COMPONENT_IDS)
    component_models = _exact_mapping(
        value, "component_model_matrix", COMPONENT_IDS
    )
    model_boundaries = _exact_mapping(value, "model_boundaries", MODEL_IDS)
    for model_id in MODEL_IDS:
        row = _require_mapping(model_boundaries[model_id], model_id)
        row["uncovered_components"] = []
    for component_id in COMPONENT_IDS:
        model_rows = _exact_mapping(component_models, component_id, MODEL_IDS)
        for model_id in MODEL_IDS:
            formal_covered = any(
                deliverable in COMPONENT_ALLOWED_DELIVERABLES[component_id]
                and model_id
                in COMPONENT_DELIVERABLE_MODEL_COVERAGE[component_id].get(
                    deliverable, set()
                )
                for deliverable in registered_formal
            )
            supplement_covered = any(
                component_id in metadata.get("components", [])
                and model_id in metadata.get("model_scope", [])
                for metadata in registered_supplements.values()
            )
            cell = _require_mapping(
                model_rows[model_id],
                f"component_model_matrix.{component_id}.{model_id}",
            )
            if formal_covered or supplement_covered:
                cell["status"] = "unresolved"
                cell["evidence_level"] = "U"
            else:
                cell["status"] = "untested"
                cell["evidence_level"] = "U"
                model_boundaries[model_id]["uncovered_components"].append(
                    component_id
                )
        aggregate = _require_mapping(component_matrix[component_id], component_id)
        aggregate["status"] = (
            "untested"
            if all(model_rows[model_id]["status"] == "untested" for model_id in MODEL_IDS)
            else "unresolved"
        )
        aggregate["evidence_level"] = "U"
        aggregate["model_scope"] = []
        aggregate["supporting_findings"] = []
    instructions = value.get("template_instructions")
    if isinstance(instructions, dict):
        instructions["component_model_coverage_prefilled_from_registry"] = True
        instructions["scientific_status_inferred_during_prefill"] = False
    return value


def render_comprehensive_report_markdown(payload: Mapping[str, Any]) -> str:
    """Render the thirteen required comprehensive-report sections."""

    narratives = payload["narratives"]
    readiness = payload["evidence_admission"]["readiness"]
    execution = payload["formal_execution_census"]
    axes = payload["execution_axis_census"]
    evidence_disposition = payload["evidence_disposition"]
    reproducibility = payload["reproducibility_ledger"]
    section_contract = payload.get("report_section_contract")
    if not isinstance(section_contract, Mapping):
        section_contract = _audit_report_section_contract(payload)
    scope_contract = payload["frozen_observation_scope_contract"]
    frozen_observation_evidence = payload["frozen_observation_evidence"]
    frozen_observation_snapshot = payload["frozen_observation_machine_snapshot"]
    prior_mechanism_snapshot = payload["prior_mechanism_diagnosis_snapshot"]
    opportunity_lineage = payload["opportunity_lineage_matrix"]
    layer_profile = payload["formal_layerwise_attenuation_profile"]
    transition_profile = payload["formal_attenuation_transition_profile"]
    component_role_coverage = payload["component_evidence_role_coverage"]
    interface_coverage = payload["transformer_internal_interface_coverage"]
    history_signal_scopes = payload["history_signal_observation_scope_contract"]
    architecture = payload["frozen_model_architecture_audit"]
    topology = architecture["frozen_topology"]
    lines = [
        "# Transformer 机制全面探索报告",
        "",
        f"状态：`{payload['status']}`  ",
        f"报告 ID：`{payload['report_id']}`",
        "",
        str(narratives["executive_summary"]["text"]),
        _narrative_evidence_line(narratives["executive_summary"]),
        "",
        "## 1. 执行与证据总表",
        "",
        f"- Declared runs：`{execution['run_declaration_count']}`",
        "- Run status counts：`"
        + json.dumps(execution["run_status_counts"], ensure_ascii=False, sort_keys=True)
        + "`",
        f"- Result-eligible completed runs：`{execution['completed_result_eligible_runs']}`",
        f"- Formal deliverables：`{readiness['formal']['completed']}/{readiness['formal']['registered']}`",
        f"- Supplemental evidence：`{readiness['supplements']['completed']}/{readiness['supplements']['registered']}`",
        f"- D2 fixed bundles：`{readiness['d2_causal_core']['fixed_completed']}/{readiness['d2_causal_core']['fixed_registered']}`",
        f"- Components with any completed artifact：`{readiness['components_with_any_completed_artifact']}/{readiness['component_count']}`",
        f"- Components with a registered causal-role artifact：`{readiness['components_with_registered_causal_role_artifact']}/{readiness['component_count']}`",
        "- Components without a registered causal-role artifact："
        + ", ".join(
            f"`{component}`"
            for component in readiness[
                "components_without_registered_causal_role_artifact"
            ]
        ),
        f"- Components with a completed causal-role artifact：`{readiness['components_with_completed_causal_role_artifact']}/{readiness['component_count']}`",
        f"- Components with completed causal-role artifacts in both Q2/Q3：`{readiness['components_with_completed_q2_q3_causal_role_artifacts']}/{readiness['component_count']}`",
        f"- Coverage boundary：{readiness['component_coverage_interpretation']}",
        f"- Exact Transformer interfaces with any completed evidence：`{interface_coverage['interfaces_with_any_completed_evidence']}/{interface_coverage['interface_count']}`",
        f"- Exact Transformer interfaces with completed causal-role evidence：`{interface_coverage['interfaces_with_completed_causal_role_evidence']}/{interface_coverage['interface_count']}`",
        f"- Operator attribution inferred from artifact availability：`{interface_coverage['operator_attribution_inferred_from_artifact_availability_count']}/{interface_coverage['interface_count']}`",
        f"- Interfaces where operator attribution remains unresolved from artifact availability：`{interface_coverage['operator_attribution_unresolved_from_artifact_availability_count']}/{interface_coverage['interface_count']}`",
        "- Registered artifact claim ceilings：`"
        + json.dumps(
            interface_coverage["registered_claim_ceiling_counts"],
            ensure_ascii=False,
            sort_keys=True,
        )
        + "`",
        "- Completed artifact claim ceilings：`"
        + json.dumps(
            interface_coverage["completed_artifact_claim_ceiling_counts"],
            ensure_ascii=False,
            sort_keys=True,
        )
        + "`",
        "- Exact Transformer interfaces without registered causal-role evidence："
        + ", ".join(
            f"`{interface_id}`"
            for interface_id in interface_coverage[
                "interfaces_without_registered_causal_role_evidence"
            ]
        ),
        f"- Exact-interface boundary：{interface_coverage['interpretation']}",
        f"- Frozen architecture audit：`{architecture['status']}`; failures=`{len(architecture['failures'])}`",
        "- Frozen Qwen topology："
        f"layers=`{topology['num_hidden_layers']}`, hidden=`{topology['hidden_size']}`, "
        f"SwiGLU intermediate=`{topology['intermediate_size']}`, "
        f"Q heads/KV heads=`{topology['num_attention_heads']}/{topology['num_key_value_heads']}`, "
        f"head dim=`{topology['head_dim']}`, RoPE theta=`{topology['rope_theta']}`, "
        f"RMSNorm eps=`{topology['rms_norm_eps']}`, tied rows=`{topology['tie_word_embeddings']}`",
        f"- Config-backed exact interfaces：`{architecture['config_backed_interface_count']}/{architecture['exact_interface_inventory_count']}`; all present=`{architecture['config_backed_interfaces_present_in_inventory']}`",
        f"- Dynamic runtime/source-backed exact interfaces：`{architecture['dynamic_runtime_or_source_backed_interface_count']}/{architecture['exact_interface_inventory_count']}`",
        f"- Exact-interface implementation provenance：`{architecture['implementation_provenance_covered_interface_count']}/{architecture['exact_interface_inventory_count']}`; exhaustive=`{architecture['all_exact_interfaces_have_config_or_runtime_source_provenance']}`",
        f"- Frozen forward primitive coverage：primitives=`{architecture['forward_primitive_count']}`, inference interfaces=`{architecture['forward_mapped_interface_count']}/{architecture['forward_inference_interface_count']}`, missing=`{len(architecture['forward_missing_interface_ids'])}`, exhaustive=`{architecture['forward_primitive_interface_coverage_complete']}`",
        f"- Installed forward-source bindings：`{architecture['forward_source_binding_count']}` @ frozen Transformers `{architecture['transformers_version']}` and PEFT `{architecture['forward_peft_version']}`; checkpoint environment=`{architecture['forward_source_environment_is_frozen_checkpoint_environment']}`",
        f"- Frozen inactive architecture paths：`{sum(row['inactive_verified'] for row in architecture['inactive_architecture_paths'])}/{architecture['inactive_architecture_path_count']}`; exhaustive=`{architecture['all_inactive_architecture_paths_verified']}`",
        f"- Forward-coverage boundary：semantic primitive census=`{architecture['forward_coverage_is_semantic_primitive_census']}`, kernel-instruction census=`{architecture['forward_coverage_is_kernel_instruction_census']}`, operator attribution inferred=`{architecture['operator_attribution_inferred_from_forward_coverage']}`",
        f"- Frozen training-update primitive coverage：primitives=`{architecture['training_primitive_count']}`, training interfaces=`{architecture['training_mapped_interface_count']}/{architecture['training_exact_interface_count']}`, missing=`{len(architecture['training_missing_interface_ids'])}`, exhaustive=`{architecture['training_primitive_interface_coverage_complete']}`",
        f"- Installed training-source/artifact bindings：`{architecture['training_source_binding_count']}/{architecture['training_artifact_binding_count']}` @ Torch `{architecture['training_torch_version']}`, Transformers `{architecture['training_transformers_version']}`, PEFT `{architecture['training_peft_version']}`",
        f"- Training-coverage boundary：single-step semantic primitive census=`{architecture['training_coverage_is_single_step_semantic_primitive_census']}`, multiseed causal attribution=`{architecture['training_coverage_is_multiseed_causal_attribution']}`, operator attribution inferred=`{architecture['operator_attribution_inferred_from_training_coverage']}`",
        f"- Frozen inactive training paths：`{sum(row['inactive_verified'] for row in architecture['inactive_training_paths'])}/{architecture['inactive_training_path_count']}`; exhaustive=`{architecture['all_inactive_training_paths_verified']}`",
        f"- Runtime instrumentation identity：models=`{architecture['runtime_identity_smoke_count']}`, hook nodes/model=`{architecture['runtime_hook_node_count']}`, backend=`{architecture['runtime_attention_backend']}`, exact identity and BF16-bounded recomposition=`{architecture['runtime_identity_and_recomposition_validated']}`",
        f"- Frozen model pathways/base artifacts：`{architecture['frozen_model_pathway_count']}/{architecture['frozen_base_artifact_count']}`",
        "- Q0–Q3 adaptation boundary："
        + "; ".join(
            f"{model_id}=`{row['base_artifact']}/{row['adaptation']}` "
            f"steps=`{row['optimizer_steps']}` objective=`{row['objective']}`"
            for model_id, row in architecture["model_pathways"].items()
        ),
        "- Q3 LoRA details："
        f"rank=`{architecture['model_pathways']['q3_tallrec_generalqwen']['lora_rank']}` "
        f"alpha=`{architecture['model_pathways']['q3_tallrec_generalqwen']['lora_alpha']}` "
        f"training dropout=`{architecture['model_pathways']['q3_tallrec_generalqwen']['lora_dropout']}` "
        f"targets=`{','.join(architecture['model_pathways']['q3_tallrec_generalqwen']['lora_targets'])}`",
        f"- Retained mechanical non-results：`{readiness['mechanical_nonresults']['retained']}`",
        f"- Mechanical boundary：{readiness['mechanical_nonresults']['interpretation']}",
        "- Registered models：" + ", ".join(f"`{value}`" for value in axes["models"]),
        "- Registered endpoints："
        + "; ".join(
            f"`{row['endpoint']}` ({row['role']})"
            for row in axes["registered_endpoints"]
        ),
        "- Registered fold roles："
        + "; ".join(
            f"fold-{row['fold']} ({row['role']})"
            for row in axes["normalized_query_folds"]
        ),
        f"- Axis boundary：{axes['boundary']}",
        f"- Source test opened：`{payload['evidence_admission']['source_test_opened']}`",
        "",
        "### 五层精确接口因果角色覆盖",
        "",
        "这里的 causal-role 只表示存在注册干预证据，不等于该算子已经被归因；debt 是没有任何 causal-role artifact 的下界。",
        f"按产物可用性推断 operator attribution：`{interface_coverage['operator_attribution_inferred_from_artifact_availability_count']}`；仍未由产物可用性解决：`{interface_coverage['operator_attribution_unresolved_from_artifact_availability_count']}`。科学归因只能由最终逐项效应解释给出，不能由完成标志生成。",
        "",
        "| System layer | Interfaces | Any completed | Causal-role registered | Causal-role completed | Lower-bound causal debt | Registered ceiling counts | Completed artifact ceiling counts |",
        "|---|---:|---:|---:|---:|---:|---|---|",
        *(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    system_layer,
                    row["interface_count"],
                    row["interfaces_with_any_completed_evidence"],
                    row["interfaces_with_registered_causal_role_evidence"],
                    row["interfaces_with_completed_causal_role_evidence"],
                    row["operator_causal_debt_count"],
                    json.dumps(
                        row["registered_claim_ceiling_counts"], sort_keys=True
                    ),
                    json.dumps(
                        row["completed_artifact_claim_ceiling_counts"],
                        sort_keys=True,
                    ),
                )
            )
            + " |"
            for system_layer, row in interface_coverage[
                "system_layer_coverage"
            ].items()
        ),
        "",
        "### 冻结的 13 项报告结构覆盖",
        "",
        f"- Plan：`{section_contract['plan']['path']}` @ `{section_contract['plan']['sha256']}`",
        f"- Covered：`{section_contract['covered_sections']}/{section_contract['registered_sections']}`",
        "",
        "| Required section | Coverage | Required payload paths |",
        "|---|---|---|",
        *(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    row["section_id"],
                    row["status"],
                    ", ".join(row["required_payload_paths"]),
                )
            )
            + " |"
            for row in section_contract["sections"]
        ),
        "",
        "### 19+21 项证据逐项处置账本",
        "",
        "每个已接纳证据必须进入有证据绑定的 finding、进入负面／冲突表，或明确声明仅用于边界而不产生科学 claim；不允许只留在复现附录中而静默忽略。",
        "",
        "| Evidence | Kind | Disposition | Findings | Scientific claim emitted | Evidence byte | Summary | Do not infer |",
        "|---|---|---|---|---:|---|---|---|",
        *(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    evidence_id,
                    row["evidence_kind"],
                    row["disposition"],
                    ", ".join(row["supporting_findings"]) or "none",
                    row["scientific_claim_emitted"],
                    f"{row['evidence_identity']['path']}@{row['evidence_identity']['sha256']}",
                    row["summary"],
                    "; ".join(row["do_not_infer"]),
                )
            )
            + " |"
            for evidence_id, row in evidence_disposition.items()
        ),
        "",
        "## 2. 冻结观察与问题边界",
        "",
        str(narratives["frozen_observation_and_scope"]["text"]),
        _narrative_evidence_line(narratives["frozen_observation_and_scope"]),
        "",
        "| Scope | Fixed definition | Claim boundary |",
        "|---|---|---|",
    ]
    for row in scope_contract:
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    row["scope_id"],
                    row["definition"],
                    row["boundary"],
                )
            )
            + " |"
        )
    lines.extend(["", "### Frozen observation evidence bytes", ""])
    for identity in frozen_observation_evidence:
        lines.append(
            f"- `{identity['evidence_id']}`：`{identity['path']}` @ "
            f"`{identity['sha256']}`"
        )
    lines.extend(
        [
            "",
            "### Frozen shared-evaluator observation snapshot",
            "",
            f"- Population：`{frozen_observation_snapshot['dataset_version']}`; counts=`"
            + json.dumps(
                frozen_observation_snapshot["surface_counts"],
                ensure_ascii=False,
                sort_keys=True,
            )
            + "`",
            f"- Pilot seed：`{frozen_observation_snapshot['pilot_seed']}`; second seed run=`{frozen_observation_snapshot['second_seed_run']}`",
            f"- Bootstrap：`{json.dumps(frozen_observation_snapshot['bootstrap'], sort_keys=True)}`",
            f"- Boundary：{frozen_observation_snapshot['interpretation_boundary']}",
            "",
            "| Model | Full NDCG@10 | Full-null overall | Full-null recurrence | Full-null strict transfer | Full-null other overlap | Recurrence contribution | Strict contribution | Evaluator evidence |",
            "|---|---:|---|---|---|---|---:|---:|---|",
        ]
    )
    for method in frozen_observation_snapshot["methods"]:
        cells = method["full_minus_null"]
        contribution = method[
            "full_minus_null_population_weighted_contribution"
        ]
        evidence = method["evaluator_evidence"]
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    method["method_id"],
                    method["full_ndcg_at_10"],
                    _mean_ci_cell(cells["overall"]),
                    _mean_ci_cell(cells["recurrence"]),
                    _mean_ci_cell(cells["strict_transfer"]),
                    _mean_ci_cell(cells["other_overlap"]),
                    contribution["recurrence"],
                    contribution["strict_transfer"],
                    f"{evidence['path']}@{evidence['sha256']}",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 3. 逐层轨迹：定位而非层号设计",
            "",
            str(narratives["layer_trajectory_interpretation"]["text"]),
            _narrative_evidence_line(
                narratives["layer_trajectory_interpretation"]
            ),
            "",
            "层扫描只用于界定后续因果分解的前／后接口；它本身不证明 attention、MLP、residual、norm 或 readout 的因果责任。",
            "",
            "跨模型对齐单位是功能节点及其有符号因果行为；绝对 block/layer index 只保留为 lineage metadata，不进入优化方向。",
            "",
            str(layer_profile["interpretation_boundary"]),
            "",
            f"层扫描来源：`{layer_profile['source']['path']}` @ `{layer_profile['source']['sha256']}`",
            "",
            "### 完整层扫描形态",
            "",
            "| Model | Endpoint | Shape | Attenuation steps | Amplification steps | Distributed | Registered follow-up |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    for row in layer_profile["shape_summary"]:
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    row["method_id"],
                    row["endpoint"],
                    row["shape"],
                    row["significant_attenuation_steps"],
                    row["significant_amplification_steps"],
                    row["distributed_attenuation_pattern_established"],
                    row["registered_followup"],
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "### 全部 post-block sufficiency 格",
            "",
            "| Model | Endpoint | Block | Mean | 95% CI | BH q | Direction |",
            "|---|---|---:|---:|---|---:|---|",
        ]
    )
    for row in layer_profile["all_layer_rows"]:
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    row["method_id"],
                    row["endpoint"],
                    row["block_zero_based"],
                    row["mean"],
                    row["ci95"],
                    row["bh_q"],
                    row["directional_description"],
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "### 全部相邻层变化",
            "",
            "| Model | Endpoint | Transition | Mean | 95% CI | BH q | Direction |",
            "|---|---|---|---:|---|---:|---|",
        ]
    )
    for row in layer_profile["adjacent_layer_rows"]:
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    row["method_id"],
                    row["endpoint"],
                    row["transition"],
                    row["mean"],
                    row["ci95"],
                    row["bh_q"],
                    row["directional_description"],
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "### 全部相邻功能节点变化",
            "",
            str(transition_profile["interpretation_boundary"]),
            "",
            f"功能节点来源：`{transition_profile['source']['path']}` @ `{transition_profile['source']['sha256']}`",
            "",
            "| Model | Endpoint | Transition | Mean | 95% CI | BH q | Direction | Evidence role |",
            "|---|---|---|---:|---|---:|---|---|",
        ]
    )
    for row in transition_profile["rows"]:
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    row["method_id"],
                    row["endpoint"],
                    row["transition"],
                    row["mean"],
                    row["ci95"],
                    row["bh_q"],
                    row["directional_description"],
                    row["evidence_role"],
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "### 从层扫描到设计证据的桥接",
            "",
            "| Stage | Question | Required evidence | Authorized consequence | Design authority |",
            "|---|---|---|---|---:|",
        ]
    )
    for row in payload["localization_to_design_bridge"]:
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    row["stage"],
                    row["question"],
                    row["required_evidence"],
                    row["authorized_consequence"],
                    row["design_authority"],
                )
            )
            + " |"
        )
    necessity_direction = payload["necessity_direction_claim_boundary"]
    lines.extend(
        [
            "",
            "### Necessity direction boundary",
            "",
            str(necessity_direction["interpretation"]),
            "",
            "Reverse-removal support targets the registered harmful full-history response; it is not evidence that the component benefits transfer or should be strengthened.",
            "",
            "## 4. 18 组件 × 4 模型证据矩阵",
            "",
            "### 历史序列信号的观测尺度与盲点",
            "",
            "下表区分完整 history span、summary endpoint、candidate native readout 与模型特定序列阶段。它回答每类证据实际观察了什么，也明确哪些逐事件或 operator 结论没有被观察；不能用 endpoint 向量扫描冒充逐事件序列归因。",
            "",
            "| Scope | Evidence | Models | Sequence/token scope | Granularity | Question answered | Not observed |",
            "|---|---|---|---|---|---|---|",
            *(
                "| "
                + " | ".join(
                    _cell(value)
                    for value in (
                        row["scope_id"],
                        ", ".join(row["evidence_ids"]),
                        ", ".join(row["model_scope"]),
                        row["sequence_or_token_scope"],
                        row["granularity"],
                        row["question_answered"],
                        row["not_observed"],
                    )
                )
                + " |"
                for row in history_signal_scopes
            ),
            "",
            "### 精确 Transformer 实现接口清单",
            "",
            "18 类科学组件之外，下表逐项公开真实实现接口，避免 Q/K normalization、RoPE、softmax edge、SwiGLU 子阶段、KV-cache phase、final RMSNorm 或训练路径被聚合标签隐藏。完成仅表示证据可用，不表示科学支持。",
            "",
            "| Interface | System layer | Implementation surface | Components | Evidence complete | Causal role registered / completed | Artifact claim ceiling registered / completed | Models registered / completed | Registered evidence roles | Claim boundary |",
            "|---|---|---|---|---:|---|---|---|---|---|",
            *(
                "| "
                + " | ".join(
                    _cell(value)
                    for value in (
                        row["interface_id"],
                        row["system_layer"],
                        row["implementation_surface"],
                        ", ".join(row["component_ids"]),
                        f"{row['completed_evidence_count']}/{row['registered_evidence_count']}",
                        f"{row['causal_role_registered']} / {row['causal_role_completed']}",
                        f"{row['registered_claim_ceiling']} / {row['completed_artifact_claim_ceiling']}",
                        (", ".join(row["model_scope_registered"]) or "none")
                        + " / "
                        + (", ".join(row["model_scope_completed"]) or "none"),
                        "; ".join(
                            f"{evidence['evidence_id']}:{evidence['role']}:{'done' if evidence['completed'] else 'pending'}"
                            for evidence in row["registered_evidence"]
                        ),
                        row["claim_boundary"],
                    )
                )
                + " |"
                for row in interface_coverage["interfaces"]
            ),
            "",
            "### 跨接口或模型范围证据",
            "",
            f"精确接口直接处置 `{interface_coverage['direct_interface_evidence_count']}` 项，跨接口处置 `{interface_coverage['cross_interface_evidence_count']}` 项；合计必须恰好覆盖全部 `{interface_coverage['registered_evidence_count']}` 项正式与补充证据。跨接口证据不能被强行归因给单一 operator。",
            "",
            "| Evidence | Kind | Completed | Models | Components | Disposition | Claim boundary |",
            "|---|---|---:|---|---|---|---|",
            *(
                "| "
                + " | ".join(
                    _cell(value)
                    for value in (
                        row["evidence_id"],
                        row["evidence_kind"],
                        row["completed"],
                        ", ".join(row["model_scope"]),
                        ", ".join(row["component_scope"]),
                        row["disposition"],
                        row["claim_boundary"],
                    )
                )
                + " |"
                for row in interface_coverage["cross_interface_evidence"]
            ),
            "",
            "### Finding evidence ledger",
            "",
            "| Finding | Level | Models | Claim | Formal evidence | Supplemental evidence | Evidence bytes | Contradiction | Do not infer |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for row in payload["findings"]:
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    f"{row['finding_id']}: {row['title']}",
                    row["evidence_level"],
                    ", ".join(row["model_scope"]),
                    row["claim"],
                    ", ".join(row["supporting_formal_deliverables"])
                    or "none",
                    ", ".join(row["supporting_supplements"]) or "none",
                    "; ".join(
                        f"{identity['evidence_id']}@{identity['sha256']}"
                        for identity in row["supporting_evidence_identities"]
                    ),
                    "; ".join(row["contradictory_evidence"]) or "none",
                    "; ".join(row["do_not_infer"]),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "### Component aggregate overview",
            "",
            "| Component | Status | Level | Models | Causal role registered | Causal role completed | Q2/Q3 causal completed | Findings | Summary | Remaining uncertainty |",
            "|---|---|---|---|---:|---:|---:|---|---|---|",
        ]
    )
    for component_id in COMPONENT_IDS:
        row = payload["component_matrix"][component_id]
        coverage = component_role_coverage[component_id]
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    component_id,
                    row["status"],
                    row["evidence_level"],
                    ", ".join(row["model_scope"]) or "none",
                    coverage["causal_role_artifact_registered"],
                    coverage["causal_role_artifact_completed"],
                    coverage["q2_q3_causal_role_artifacts_completed"],
                    ", ".join(row["supporting_findings"]) or "none",
                    row["summary"],
                    row["remaining_uncertainty"],
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "### Component-model cells",
            "",
            "| Component | Model | Mechanism question | Registered evidence | Causal role registered | Causal role completed | Findings | Status | Level | Summary | Claim boundary | Remaining uncertainty |",
            "|---|---|---|---|---:|---:|---|---|---|---|---|---|",
        ]
    )
    for component_id in COMPONENT_IDS:
        aggregate = payload["component_matrix"][component_id]
        coverage = component_role_coverage[component_id]
        for model_id in MODEL_IDS:
            row = payload["component_model_matrix"][component_id][model_id]
            lines.append(
                "| "
                + " | ".join(
                    _cell(value)
                    for value in (
                        component_id,
                        model_id,
                        aggregate["mechanism_question"],
                        ", ".join(row["registered_evidence_sources"])
                        or "not-directly-registered",
                        model_id
                        in coverage["causal_role_model_scope_registered"],
                        model_id
                        in coverage["causal_role_model_scope_completed"],
                        ", ".join(row["supporting_findings"]) or "none",
                        row["status"],
                        row["evidence_level"],
                        row["summary"],
                        aggregate["claim_boundary"],
                        row["remaining_uncertainty"],
                    )
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## 5. 功能因果链",
            "",
            "| Node | Status | Level | Models | Diagnosis | Claim boundary | Findings |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for row in payload["functional_causal_chain"]:
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    row["node"],
                    row["status"],
                    row["evidence_level"],
                    ", ".join(row["model_scope"]) or "none",
                    row["diagnosis"],
                    row["claim_boundary"],
                    ", ".join(row["supporting_findings"]),
                )
                )
            + " |"
        )
    gate_matrix = payload["component_bidirectional_gate_matrix"]
    lines.extend(
        [
            "",
            "### S / N / specificity / controls / G 原始门矩阵",
            "",
            str(gate_matrix["interpretation_boundary"]),
            "",
            f"来源：`{gate_matrix['source']['path']}` @ `{gate_matrix['source']['sha256']}`；主endpoint：`{gate_matrix['primary_endpoint']}`；重新计算效应：`{gate_matrix['scientific_effect_values_recomputed']}`",
            "",
            "| Model | Functional node | Role | S same-request | N position-preserving | Same-minus-wrong | Cross-request | Norm/direction/random | Combined state gate | Design eligible | G gate |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in gate_matrix["rows"]:
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    row["method_id"],
                    row["functional_node"],
                    row["claim_role"],
                    row["sufficiency_S_same_request"],
                    row["necessity_N_position_preserving_removal"],
                    row["history_specificity_same_minus_wrong"],
                    row["cross_request_stress_control"],
                    row["norm_direction_random_controls"],
                    row["combined_component_state_gate"],
                    row["functional_node_design_target_eligible"],
                    row["design_G_gate"],
                )
            )
            + " |"
        )
    cross_gate = gate_matrix["cross_model"]
    lines.extend(
        [
            "",
            "- 跨Q2/Q3 component-state节点："
            + (
                ", ".join(
                    f"`{node}`"
                    for node in cross_gate["component_state_supported_nodes"]
                )
                or "none"
            ),
            "- 跨Q2/Q3 G级设计节点："
            + (
                ", ".join(
                    f"`{node}`"
                    for node in cross_gate["design_prioritized_nodes"]
                )
                or "none"
            ),
            "",
        ]
    )
    failure = payload["failure_mode_diagnosis"]
    lines.extend(
        [
            "",
            "### Erasure / transport / loss-of-use classification",
            "",
            f"- Primary mode: `{failure['primary_mode']}` (`{failure['evidence_level']}`)",
            f"- Diagnostic resolution: `{failure['diagnostic_resolution']}`",
            f"- Diagnosis: {failure['summary']}",
            "- Functional components: "
            + (
                ", ".join(f"`{value}`" for value in failure["functional_components"])
                or "none"
            ),
            "- Component/model evidence: "
            + (
                "; ".join(
                    f"`{component}`="
                    + ",".join(f"`{model}`" for model in models)
                    for component, models in failure["component_model_evidence"].items()
                )
                or "none"
            ),
            f"- Causal erasure authorized: `{failure['causal_erasure_claim_authorized']}`",
            f"- Causal loss-of-use authorized: `{failure['causal_loss_of_use_claim_authorized']}`",
            "- Models: " + ", ".join(f"`{value}`" for value in failure["model_scope"]),
            "- Evidence findings: "
            + ", ".join(f"`{value}`" for value in failure["supporting_findings"]),
            f"- Claim boundary: {failure['claim_boundary']['interpretation']}",
            f"- Falsification gate: {failure['falsification_gate']}",
            "- Competing modes: "
            + "; ".join(
                f"`{row['mode']}` — {row['reason_remaining']}"
                for row in failure["competing_modes"]
            ),
        ]
    )
    lines.extend(["", "## 6. 输入／表示／路由／readout／训练五层解释", ""])
    for layer_id, row in payload["system_layers"].items():
        lines.extend(
            [
                f"### {layer_id}",
                "",
                f"状态／证据等级：`{row['status']}` / `{row['evidence_level']}`",
                "",
                "模型范围："
                + (
                    ", ".join(f"`{value}`" for value in row["model_scope"])
                    or "none"
                ),
                "",
                "功能组件："
                + ", ".join(f"`{value}`" for value in row["functional_components"]),
                "",
                str(row["diagnosis"]),
                "",
                "证据 findings："
                + (
                    ", ".join(
                        f"`{value}`" for value in row["supporting_findings"]
                    )
                    or "none"
                ),
                "",
                f"剩余不确定性：{row['remaining_uncertainty']}",
                "",
            ]
        )
    lines.extend(
        [
            "## 7. Q0–Q3 横向边界",
            "",
            str(narratives["cross_model_boundary"]["text"]),
            _narrative_evidence_line(narratives["cross_model_boundary"]),
            "",
        ]
    )
    for model_id, row in payload["model_boundaries"].items():
        lines.extend(
            [
                f"- `{model_id}`：{row['summary']}",
                "  - Evidence findings："
                + (
                    ", ".join(
                        f"`{value}`" for value in row["supporting_findings"]
                    )
                    or "none"
                ),
                "  - Untested components："
                + (
                    ", ".join(
                        f"`{value}`" for value in row["uncovered_components"]
                    )
                    or "none"
                ),
                f"  - 不可外推：{row['do_not_generalize']}",
                "",
            ]
        )
    cross_model = payload["cross_model_synthesis"]
    for pattern_kind, title in (
        ("shared_patterns", "跨模型共享功能现象"),
        ("heterogeneous_patterns", "跨模型异质功能现象"),
    ):
        lines.extend(
            [
                f"### {title}",
                "",
                "| Pattern | Level | Models | Functional components | Findings | Summary | Do not generalize |",
                "|---|---|---|---|---|---|---|",
            ]
        )
        for row in cross_model[pattern_kind]:
            lines.append(
                "| "
                + " | ".join(
                    _cell(value)
                    for value in (
                        row["pattern_id"],
                        row["evidence_level"],
                        ", ".join(row["model_scope"]),
                        ", ".join(row["functional_components"]),
                        ", ".join(row["supporting_findings"]),
                        row["summary"],
                        row["do_not_generalize"],
                    )
                )
                + " |"
            )
        lines.append("")
    lines.extend(
        [
            f"- 跨模型剩余不确定性：{cross_model['remaining_uncertainty']}",
            f"- 使用绝对层号对齐：`{cross_model['absolute_index_alignment_used']}`",
            "",
        ]
    )
    lines.extend(
        [
            "## 8. H0–H5 证据矩阵",
            "",
            f"首轮M0--M3边界：{prior_mechanism_snapshot['interpretation_boundary']}",
            "",
            "### Prior M0--M3 hypothesis state",
            "",
            "| Hypothesis | Prior status | Claim level | Statement | Rationale | Contradictions | Remaining uncertainty |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for row in prior_mechanism_snapshot["hypothesis_status_matrix"]:
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    row["hypothesis_id"],
                    row["status"],
                    row["claim_level"],
                    row["statement_verbatim"],
                    row["rationale"],
                    ", ".join(row["contradiction_ids"]),
                    "; ".join(row["remaining_uncertainty"]),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "### Prior M0--M3 contradiction ledger",
            "",
            "| Contradiction | Evidence | Description | Interpretation | Mechanical failure |",
            "|---|---|---|---|---:|",
        ]
    )
    for row in prior_mechanism_snapshot["contradictions"]:
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    row["contradiction_id"],
                    ", ".join(row["evidence_ids"]),
                    row["description"],
                    row["interpretation"],
                    row["mechanical_failure"],
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "### Prior logical-evidence to artifact mapping",
            "",
            "| Logical evidence | Stage | Artifact IDs | Summary | Scope |",
            "|---|---|---|---|---|",
        ]
    )
    for row in prior_mechanism_snapshot["evidence_index"]:
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    row["evidence_id"],
                    row["stage"],
                    ", ".join(row["artifact_ids"]),
                    row["summary"],
                    row["scope"],
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "### Prior M0--M3 artifact identities",
            "",
            "| Artifact | Stage | Kind | Run | Evidence byte | Verbatim/no recomputation |",
            "|---|---|---|---|---|---:|",
        ]
    )
    for row in prior_mechanism_snapshot["artifact_registry"]:
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    row["evidence_id"],
                    row["stage"],
                    row["kind"],
                    row["run_id"],
                    f"{row['path']}@{row['sha256']}",
                    row["copied_verbatim_without_statistical_recomputation"],
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            f"首轮M0--M3原始产物字节：`{len(prior_mechanism_snapshot['artifact_registry'])}`项；本报告未重算其scientific effect。",
            "",
            "### Final deep-dive-updated hypothesis state",
            "",
            "| Hypothesis | Status | Level | Findings | Negative-evidence basis | Summary | Contradiction | Remaining uncertainty |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    for hypothesis_id, row in payload["hypothesis_matrix"].items():
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    hypothesis_id,
                    row["status"],
                    row["evidence_level"],
                    ", ".join(row["supporting_findings"]) or "none",
                    ", ".join(row["negative_evidence_basis"]),
                    row["summary"],
                    "; ".join(row["contradictory_evidence"]),
                    row["remaining_uncertainty"],
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 9. 全部负结果与冲突",
            "",
            "| Result | Models | Endpoints | Surfaces | Contrasts | Folds | Seeds | Findings | Summary | Interpretation boundary |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for row in payload["negative_and_conflicting_results"]:
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    row["result_id"],
                    ", ".join(row["model_scope"]),
                    ", ".join(row["endpoint_scope"]),
                    ", ".join(row["surface_scope"]),
                    ", ".join(row["contrast_scope"]),
                    ", ".join(row["fold_scope"]),
                    ", ".join(row["seed_scope"]),
                    ", ".join(row["supporting_findings"]),
                    row["summary"],
                    row["interpretation_boundary"],
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "### 保留的机械 non-results",
            "",
            str(readiness["mechanical_nonresults"]["interpretation"]),
            "",
        ]
    )
    for run_id in readiness["mechanical_nonresults"]["run_ids"]:
        lines.append(f"- `{run_id}`")
    lines.extend(
        [
            "",
            "## 10. 优化机会排序与证伪门",
            "",
            "### 首轮假设 → formal 排序 → 综合处置",
            "",
            str(opportunity_lineage["interpretation_boundary"]),
            "",
            "| Opportunity | Prior priority | Prior hypotheses | Formal rank / status | Formal evidence | Disposition | Final target kind / ID | Component | Final priority / basis | Level | Models | Findings | Utility gain | Architecture implemented |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---:|---:|",
            *(
                "| "
                + " | ".join(
                    _cell(value)
                    for value in (
                        row["opportunity_id"],
                        row["prior_priority"],
                        ", ".join(row["prior_bottleneck_hypotheses"]),
                        f"{row['formal_rank']} / {row['formal_status']}",
                        ", ".join(row["formal_evidence_deliverables"]),
                        row["disposition"],
                        f"{row['target_kind']} / {row['target_id']}",
                        row["functional_component"],
                        (
                            row["design_priority"]
                            if row["basis"] is None
                            else f"{row['design_priority']} / {row['basis']}"
                        ),
                        row["actual_evidence_level"] or "n/a",
                        ", ".join(row["model_scope"]),
                        ", ".join(row["supporting_findings"]),
                        row["utility_gain_established"],
                        row["architecture_implemented"],
                    )
                )
                + " |"
                for row in opportunity_lineage["rows"]
            ),
            "",
            "### 首轮 M0--M3 冻结的五项机会假设",
            "",
            "以下是deep-dive开始前已经冻结的设计假设、模块、训练信号、消融和可证伪预测。它们不是最终排序或已实现方法；正式与综合排序只能用后续admitted evidence更新其处置，不能静默改写原始设计理由。",
            "",
        ]
    )
    for row in prior_mechanism_snapshot["architecture_opportunity_matrix"]:
        lines.extend(
            [
                f"#### {row['opportunity_id']}",
                "",
                f"- Prior priority：`{row['priority']}`",
                "- Bottleneck hypotheses："
                + ", ".join(
                    f"`{value}`" for value in row["bottleneck_hypotheses"]
                ),
                "- Logical evidence："
                + ", ".join(f"`{value}`" for value in row["evidence_ids"]),
                f"- Innovation target：{row['innovation_target']}",
                f"- Architecture requirement：{row['architecture_requirement']}",
                "- Necessary modules：" + "; ".join(row["necessary_modules"]),
                "- Training signals：" + "; ".join(row["training_signals"]),
                "- Train-only data requirements："
                + "; ".join(row["train_only_data_requirements"]),
                "- Key ablations：" + "; ".join(row["key_ablations"]),
                "- Falsifiable predictions："
                + "; ".join(row["falsifiable_predictions"]),
                f"- Implementation status：`{row['implementation_status']}`",
                f"- Evaluation contract unchanged：`{row['evaluation_contract_unchanged']}`",
                "- Prior-work differentiation：",
            ]
        )
        for comparator, comparison in row["prior_work_differentiation"].items():
            lines.append(
                f"  - `{comparator}`：shared={comparison['shared_ground']} "
                f"difference={comparison['substantive_difference']} "
                f"source={comparison['source_ref']}"
            )
        lines.append("")
    lines.extend(
        [
            "### 正式深挖报告冻结的五项机会",
            "",
            "下表逐字保留formal closeout中的五项排序；后续组件门控排序是更严格的综合更新，不能用省略来表示降级或反证。",
            "",
            "| Formal rank | Opportunity | Status | Models | Evidence | Rationale | Falsification gate |",
            "|---:|---|---|---|---|---|---|",
        ]
    )
    for row in payload["formal_architecture_opportunity_ranking"]:
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    row["rank"],
                    row["opportunity_id"],
                    row["status"],
                    ", ".join(row["model_scope"]),
                    ", ".join(row["evidence_deliverables"]),
                    row["rationale"],
                    row["falsification_gate"],
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "### 冻结机会到综合结论的精确处置",
            "",
            "| Formal opportunity | Disposition | Comprehensive target |",
            "|---|---|---|",
        ]
    )
    for formal_id, row in payload["formal_opportunity_disposition"].items():
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    formal_id,
                    row["disposition"],
                    row["target_id"],
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "### 组件门控后的综合机会排序",
            "",
            "| Rank | Opportunity | Formal predecessors | Component / node | Mechanism target | Minimum / actual level | Priority | Evidence tier / ranking basis | Polarity | Utility gain established | Scope | Evidence bytes | Findings | Contradictory evidence | Hypothesized innovation | Training signal requirements | Key ablations | Closest baseline families | Baseline differentiation | Expected benefit | Key risks | Reason | Falsification gate | Do not infer | Diagnostic patch promoted | Architecture implemented |",
            "|---:|---|---|---|---|---|---|---|---|---:|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---:|---:|",
        ]
    )
    for row in payload["optimization_opportunities"]:
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    row["rank"],
                    row["opportunity_id"],
                    ", ".join(row["formal_predecessor_ids"]) or "new_direction",
                    row["functional_component"]
                    + (
                        f" / {row['functional_node']}"
                        if row["functional_node"] is not None
                        else ""
                    ),
                    row["mechanism_target"],
                    f"{row['minimum_evidence_level']} / {row['actual_evidence_level']}",
                    row["design_priority"],
                    f"{row['evidence_strength_tier']} / {row['ranking_basis']}",
                    row["intervention_polarity"],
                    row["utility_gain_established"],
                    ", ".join(row["model_scope"])
                    + f"; {row['dataset_scope']}; source_test={row['source_test_opened']}",
                    "; ".join(
                        f"{identity['evidence_id']}@{identity['sha256']}"
                        for identity in row["supporting_evidence_identities"]
                    ),
                    ", ".join(row["supporting_findings"]),
                    "; ".join(row["contradictory_evidence"]),
                    row["hypothesized_innovation"],
                    "; ".join(row["training_signal_requirements"]),
                    "; ".join(row["key_ablations"]),
                    "; ".join(row["closest_baseline_families"]),
                    row["baseline_differentiation"],
                    row["expected_benefit"],
                    "; ".join(row["key_risks"]),
                    row["reason"],
                    row["falsification_gate"],
                    "; ".join(row["do_not_infer"]),
                    row["diagnostic_patch_promoted_as_method"],
                    row["architecture_implemented"],
                )
            )
            + " |"
        )
    lines.extend(["", "## 11. 明确不建议的方向", ""])
    for row in payload["not_recommended"]:
        lines.append(
            f"- **{row['direction']}** [`{row['functional_component']}`]："
            f"{row['reason']}（{row['basis']}；models: "
            f"{', '.join(row['model_scope'])}；dataset: {row['dataset_scope']}；"
            f"source_test={row['source_test_opened']}；findings: "
            f"{', '.join(row['supporting_findings'])}；formal predecessors: "
            f"{', '.join(row['formal_predecessor_ids']) or 'none'}）"
        )
    lines.extend(
        [
            "",
            "## 12. 论文 claim 边界",
            "",
            str(narratives["paper_claim_boundary"]["text"]),
            _narrative_evidence_line(narratives["paper_claim_boundary"]),
            "",
            "### 固定 claim invariants",
            "",
        ]
    )
    for key, value in sorted(payload["claim_invariants"].items()):
        lines.append(f"- `{key}`：`{value}`")
    lines.extend(
        [
            "",
            "### 进入论文方法阶段仍需通过的门槛",
            "",
            "| Requirement | Evidence needed | Current-stage boundary |",
            "|---|---|---|",
        ]
    )
    for row in payload["paper_method_stage_requirements"]:
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    row["requirement_id"],
                    row["evidence_needed"],
                    row["current_stage_boundary"],
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "### 尚无注册 operator-level 因果检验的精确接口",
            "",
            "这些接口可以拥有描述性或训练动力学证据，但当前报告不能把它们写成已定位的因果瓶颈；若未来将其作为方法核心，需要新的预注册干预与跨模型复现。",
            "该清单只是“连 causal-role artifact 都没有”的下界；不在表内的接口也不能据此自动获得 operator attribution。",
            "",
            f"- 结构化因果债务：`{interface_coverage['operator_causal_debt_count']}`",
            "- 债务类型：`"
            + json.dumps(
                interface_coverage["operator_causal_debt_class_counts"],
                ensure_ascii=False,
                sort_keys=True,
            )
            + "`",
            f"- 本账本授权新增实验 family：`{interface_coverage['new_experiment_family_authorized_by_debt_ledger']}`",
            "",
            "| Interface | System layer | Debt class | Components | Current evidence roles | Minimum future evidence | Smallest falsification gate | Current-stage disposition | Active experiment authorized |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for row in interface_coverage["operator_causal_debt"]:
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    row["interface_id"],
                    row["system_layer"],
                    row["debt_class"],
                    ", ".join(row["component_ids"]),
                    ", ".join(row["registered_evidence_roles"]),
                    row["minimum_future_evidence"],
                    row["smallest_falsification_gate"],
                    row["current_stage_disposition"],
                    row["active_experiment_authorized"],
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "这些机制证据最多授权在冻结范围内排序下一轮设计候选；诊断 patch 不是方法，"
            "target-margin mediation不是NDCG收益，进入论文方法阶段仍需新的用户授权、独立实现、"
            "多seed确认、预注册效用评估、强基线与组件消融。",
            "",
            "## 13. 可复现附录",
            "",
            f"- Formal report：`{payload['evidence_admission']['formal_report']['path']}` @ `{payload['evidence_admission']['formal_report']['sha256']}`",
            f"- Supplemental registry：`{payload['evidence_admission']['supplement_registry']['path']}` @ `{payload['evidence_admission']['supplement_registry']['sha256']}`",
            f"- Supplemental manifest：`{payload['evidence_admission']['supplement_registry_manifest']['path']}` @ `{payload['evidence_admission']['supplement_registry_manifest']['sha256']}`",
            "",
            "### Frozen model/config identities",
            "",
        ]
    )
    for identity in architecture["files"].values():
        lines.append(f"- `{identity['path']}` @ `{identity['sha256']}`")
    lines.extend(["", "### Dynamic interface implementation provenance", ""])
    for row in architecture["dynamic_runtime_or_source_backed_interfaces"]:
        source_bytes = "; ".join(
            f"{source['path']}@{source['sha256']}"
            for source in row["source_identities"]
        )
        runtime_suffix = (
            f"; runtime node=`{row['runtime_identity_node']}`; identity models="
            + ",".join(f"`{model_id}`" for model_id in row["runtime_identity_models"])
            if row["runtime_identity_node"] is not None
            else ""
        )
        algebra_suffix = (
            f"; algebra=`{row['runtime_algebra_key']}`; algebra models="
            + ",".join(f"`{model_id}`" for model_id in row["runtime_algebra_models"])
            if row.get("runtime_algebra_key") is not None
            else ""
        )
        lines.append(
            f"- `{row['interface_id']}`：binding=`{row['binding_kind']}`; "
            f"status=`{row['status']}`; sources={source_bytes}{runtime_suffix}"
            f"{algebra_suffix}; "
            f"scientific support inferred=`{row['scientific_support_inferred']}`"
        )
    lines.extend(["", "### Frozen inference forward primitive map", ""])
    lines.extend(
        [
            "| Order | Primitive | Exact interfaces | Model scope | Status | Operator attribution inferred |",
            "|---:|---|---|---|---|---:|",
        ]
    )
    for row in architecture["forward_primitives"]:
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    row["execution_order"],
                    row["primitive_id"],
                    ", ".join(row["interface_ids"]),
                    ", ".join(row["model_scope"]),
                    row["status"],
                    row["operator_attribution_inferred"],
                )
            )
            + " |"
        )
    lines.extend(["", "### Installed Transformer forward source bindings", ""])
    for row in architecture["forward_source_bindings"]:
        lines.append(
            f"- `{row['source_id']}` / `{row['object']}`："
            f"`{row['package_relative_path']}` @ `{row['source_file_sha256']}`; "
            f"object source=`{row['object_source_sha256']}`, "
            f"sentinels=`{row['required_fragment_count']}`, status=`{row['status']}`"
        )
    lines.extend(["", "### Frozen training-update primitive map", ""])
    lines.extend(
        [
            "| Order | Primitive | Exact interfaces | Model scope | Status | Operator attribution inferred |",
            "|---:|---|---|---|---|---:|",
        ]
    )
    for row in architecture["training_primitives"]:
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    row["execution_order"],
                    row["primitive_id"],
                    ", ".join(row["interface_ids"]),
                    ", ".join(row["model_scope"]),
                    row["status"],
                    row["operator_attribution_inferred"],
                )
            )
            + " |"
        )
    lines.extend(["", "### Installed training-update source bindings", ""])
    for row in architecture["training_source_bindings"]:
        lines.append(
            f"- `{row['source_id']}` / `{row['object']}`："
            f"`{row['package_relative_path']}` @ `{row['source_file_sha256']}`; "
            f"object source=`{row['object_source_sha256']}`, "
            f"sentinels=`{row['required_fragment_count']}`, status=`{row['status']}`"
        )
    lines.extend(["", "### Frozen training artifact bindings", ""])
    for row in architecture["training_artifact_bindings"]:
        detail_parts = []
        if "dropout_executes_before_a_down_projection" in row:
            detail_parts.extend(
                [
                    f"dropout before A=`{row['dropout_executes_before_a_down_projection']}`",
                    f"eval identity=`{row['dropout_identity_at_evaluation']}`",
                ]
            )
        if "current_project_source_matches_frozen_training_identity" in row:
            detail_parts.append(
                "current source matches frozen training identity=`"
                f"{row['current_project_source_matches_frozen_training_identity']}`"
            )
        detail_suffix = ", ".join(detail_parts)
        if detail_suffix:
            detail_suffix += ", "
        lines.append(
            f"- `{row['binding_id']}`：kind=`{row['binding_kind']}`, "
            f"`{row['path']}` @ `{row['sha256']}`; status=`{row['status']}`, "
            f"{detail_suffix}"
            f"scientific support inferred=`{row['scientific_support_inferred']}`"
        )
    lines.extend(["", "### Frozen inactive training paths", ""])
    lines.extend(
        [
            "| Path | Evidence | Observed | Expected | Inactive verified | Boundary |",
            "|---|---|---|---|---:|---|",
        ]
    )
    for row in architecture["inactive_training_paths"]:
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    row["path_id"],
                    row["evidence_kind"],
                    json.dumps(row["observed"], ensure_ascii=False, sort_keys=True),
                    json.dumps(row["expected"], ensure_ascii=False, sort_keys=True),
                    row["inactive_verified"],
                    row["explanation"],
                )
            )
            + " |"
        )
    lines.extend(["", "### Frozen inactive architecture paths", ""])
    lines.extend(
        [
            "| Path | Evidence | Observed | Expected | Inactive verified | Boundary |",
            "|---|---|---|---|---:|---|",
        ]
    )
    for row in architecture["inactive_architecture_paths"]:
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    row["path_id"],
                    row["evidence_kind"],
                    json.dumps(row["observed"], ensure_ascii=False, sort_keys=True),
                    row["expected"],
                    row["inactive_verified"],
                    row["explanation"],
                )
            )
            + " |"
        )
    lines.extend(["", "### Runtime instrumentation identity smokes", ""])
    for row in architecture["runtime_identity_smokes"]:
        lines.append(
            f"- `{row['method_id']}`：`{row['path']}` @ `{row['sha256']}`; "
            f"backend=`{row['attention_backend']}`, hook nodes=`{row['hook_nodes_validated']}`, "
            f"max identity error=`{row['maximum_identity_error']}`, "
            f"algebra recomposition=`{row['algebra_recomposition_passed']}`"
        )
    lines.extend(["", "### Superseded runtime mechanical lineage", ""])
    for row in architecture["superseded_runtime_lineage"]:
        lines.append(
            f"- `{row['run_id']}`：`{row['path']}` @ `{row['sha256']}`; "
            f"status=`{row['status']}`, failure=`{row['failure_kind']}`, "
            f"max identity error=`{row['maximum_identity_error']}` > "
            f"tolerance=`{row['identity_tolerance']}`, canonical replacement="
            f"`{row['canonical_replacement']}`, scientific result eligible="
            f"`{row['scientific_result_eligible']}`"
        )
    lines.append("")
    for label, key in (
        ("Frozen asset", "frozen_assets"),
        ("Formal deliverable", "formal_deliverables"),
        ("Supplement", "supplements"),
    ):
        lines.append(f"### {label} identities")
        lines.append("")
        for identity in reproducibility[key]:
            command_suffix = (
                " — command: `" + " ".join(identity["command"]) + "`"
                if key == "supplements"
                else ""
            )
            lines.append(
                f"- `{identity['evidence_id']}`：`{identity['path']}` @ `{identity['sha256']}`{command_suffix}"
            )
        lines.append("")
    lines.extend(
        [
            "### Audited run metadata and commands",
            "",
            "| Run | Stage | Model | Status | Eligible | Metadata bytes | Command |",
            "|---|---|---|---|---:|---|---|",
        ]
    )
    for row in reproducibility["run_declarations"]:
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    row["evidence_id"],
                    row["analysis_stage"],
                    row["method_id"],
                    row["status"],
                    row["result_eligible"],
                    f"{row['path']}@{row['sha256']}",
                    " ".join(row["command"]) or "not-declared-for-nonformal-run",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "- Dev-eval ledger：`"
            + json.dumps(
                reproducibility["dev_eval_ledger"],
                ensure_ascii=False,
                sort_keys=True,
            )
            + "`",
            "- Commands copied from audited metadata：`"
            + str(reproducibility["commands_are_copied_from_audited_run_metadata"])
            + "`",
            "- 本 builder 未读取 qrels 或 score bundle；paper metrics 只来自已审计 evaluator 输出。",
            "- Source test 保持关闭；未切数据集；未实现或晋升诊断 patch 为论文方法。",
            "",
        ]
    )
    return "\n".join(lines)


def _build_reproducibility_ledger(
    *,
    formal: Mapping[str, Any],
    completed_supplements: Mapping[str, Mapping[str, Any]],
    root: Path | None = None,
) -> dict[str, Any]:
    """Normalize every audited input/output/run identity needed by the appendix."""

    admission = formal.get("evidence_admission")
    if not isinstance(admission, Mapping):
        raise ValueError("formal report evidence admission is missing")
    frozen_assets = admission.get("frozen_assets")
    deliverables = admission.get("deliverables")
    declarations = admission.get("run_declarations")
    if not isinstance(frozen_assets, Mapping) or not frozen_assets:
        raise ValueError("formal report frozen-asset identities are missing")
    if not isinstance(deliverables, Mapping) or set(deliverables) != set(
        EXPECTED_DELIVERABLES
    ):
        raise ValueError("formal report deliverable identity coverage differs")
    if not isinstance(declarations, list) or not declarations:
        raise ValueError("formal report run declarations are missing")

    normalized_assets = []
    for path, sha256 in sorted(frozen_assets.items()):
        normalized_assets.append(
            _normalized_evidence_identity(
                evidence_id=str(path),
                evidence_kind="frozen_asset",
                identity={"path": path, "sha256": sha256},
            )
        )
    if not any(
        row["path"] == COMPREHENSIVE_REPORT_PLAN_IDENTITY["path"]
        for row in normalized_assets
    ):
        normalized_assets.append(
            _normalized_evidence_identity(
                evidence_id=COMPREHENSIVE_REPORT_PLAN_IDENTITY["path"],
                evidence_kind="frozen_asset",
                identity=COMPREHENSIVE_REPORT_PLAN_IDENTITY,
            )
        )
    required_static_identities = (
        {
            "evidence_id": COMPREHENSIVE_REPORT_PLAN_IDENTITY["path"],
            "evidence_kind": "frozen_asset",
            **COMPREHENSIVE_REPORT_PLAN_IDENTITY,
        },
        *FROZEN_OBSERVATION_EVIDENCE_IDENTITIES,
    )
    assets_by_path = {row["path"]: row for row in normalized_assets}
    for declared in required_static_identities:
        normalized = _normalized_evidence_identity(
            evidence_id=str(declared["evidence_id"]),
            evidence_kind=str(declared["evidence_kind"]),
            identity=declared,
        )
        existing = assets_by_path.get(normalized["path"])
        if existing is not None and existing["sha256"] != normalized["sha256"]:
            raise ValueError(
                "frozen observation evidence conflicts with an admitted frozen asset: "
                f"{normalized['path']}"
            )
        if existing is None:
            normalized_assets.append(normalized)
            assets_by_path[normalized["path"]] = normalized
        if root is not None:
            _validate_repository_evidence_identity(root, normalized)
    normalized_deliverables = []
    for evidence_id, identity in sorted(deliverables.items()):
        if not isinstance(identity, Mapping) or identity.get("status") != "completed":
            raise ValueError(f"formal deliverable is not completed: {evidence_id}")
        normalized_deliverables.append(
            _normalized_evidence_identity(
                evidence_id=str(evidence_id),
                evidence_kind="formal_deliverable",
                identity=identity,
            )
        )
    normalized_supplements = []
    for evidence_id, identity in sorted(completed_supplements.items()):
        if not isinstance(identity, Mapping) or identity.get("status") != "completed":
            raise ValueError(f"supplement is not completed: {evidence_id}")
        command_value = identity.get("command")
        if not isinstance(command_value, list) or not command_value or any(
            not isinstance(value, str) or not value.strip() for value in command_value
        ):
            raise ValueError(f"completed supplement lacks its command: {evidence_id}")
        normalized_supplements.append(
            {
                **_normalized_evidence_identity(
                evidence_id=evidence_id,
                evidence_kind="supplement",
                identity=identity,
                ),
                "command": [value.strip() for value in command_value],
            }
        )

    normalized_runs = []
    run_ids: set[str] = set()
    for raw in declarations:
        row = _require_mapping(raw, "formal run declaration")
        run_id = _nonempty_string(row.get("run_id"), "formal run_id")
        if run_id in run_ids:
            raise ValueError(f"duplicate formal run declaration: {run_id}")
        run_ids.add(run_id)
        identity = _normalized_evidence_identity(
            evidence_id=run_id,
            evidence_kind="run_metadata",
            identity=row,
        )
        command_value = row.get("command")
        if isinstance(command_value, str):
            command = [_nonempty_string(command_value, "run command")]
        elif isinstance(command_value, list):
            command = [
                _nonempty_string(value, "run command") for value in command_value
            ]
        elif command_value is None:
            command = []
        else:
            raise ValueError(f"invalid run command declaration: {run_id}")
        if (
            row.get("status") == "completed"
            and row.get("result_eligible") is True
            and not command
        ):
            raise ValueError(f"completed formal run lacks its command: {run_id}")
        normalized_runs.append(
            {
                **identity,
                "analysis_stage": row.get("analysis_stage"),
                "method_id": row.get("method_id"),
                "status": str(row.get("status")),
                "result_eligible": row.get("result_eligible") is True,
                "command": command,
            }
        )

    dev_eval_ledger = admission.get("dev_eval_ledger")
    if not isinstance(dev_eval_ledger, Mapping):
        raise ValueError("formal report dev-eval ledger identity is missing")
    return {
        "frozen_assets": normalized_assets,
        "formal_deliverables": normalized_deliverables,
        "supplements": normalized_supplements,
        "run_declarations": normalized_runs,
        "dev_eval_ledger": dict(dev_eval_ledger),
        "source_test_content_read_by_closeout_audit": admission.get(
            "source_test_content_read_by_closeout_audit"
        ),
        "qrels_content_read_by_closeout_audit": admission.get(
            "qrels_content_read_by_closeout_audit"
        ),
        "commands_are_copied_from_audited_run_metadata": True,
    }


def _validate_repository_evidence_identity(
    root: Path, identity: Mapping[str, Any]
) -> None:
    """Fail closed when a frozen report source is missing, escaping, or stale."""

    root_path = root.resolve()
    raw_path = _nonempty_string(identity.get("path"), "repository evidence path")
    relative = Path(raw_path)
    if relative.is_absolute():
        raise ValueError(f"repository evidence path must be relative: {raw_path}")
    path = (root_path / relative).resolve()
    try:
        path.relative_to(root_path)
    except ValueError as exc:
        raise ValueError(f"repository evidence path escapes root: {raw_path}") from exc
    if not path.is_file():
        raise ValueError(f"repository evidence path is missing: {raw_path}")
    expected = _nonempty_string(
        identity.get("sha256"), f"{raw_path} repository evidence SHA-256"
    )
    if sha256_file(path) != expected:
        raise ValueError(f"repository evidence SHA-256 drift: {raw_path}")


def _audit_report_section_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Prove that each frozen comprehensive-plan section has report payload."""

    sections = []
    for registered in REPORT_SECTION_CONTRACT:
        required_paths = tuple(registered["required_payload_paths"])
        for path in required_paths:
            value = _payload_value_at_path(payload, path)
            if value is None or (
                isinstance(value, (str, list, tuple, dict)) and not value
            ):
                raise ValueError(
                    "comprehensive report section lacks required payload: "
                    f"{registered['section_id']}.{path}"
                )
        sections.append(
            {
                "section_id": registered["section_id"],
                "status": "covered",
                "required_payload_paths": list(required_paths),
            }
        )
    return {
        "plan": dict(COMPREHENSIVE_REPORT_PLAN_IDENTITY),
        "registered_sections": len(REPORT_SECTION_CONTRACT),
        "covered_sections": len(sections),
        "sections": sections,
        "scientific_effect_values_read_for_coverage": False,
    }


def _payload_value_at_path(payload: Mapping[str, Any], dotted_path: str) -> Any:
    value: Any = payload
    for key in dotted_path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _audit_human_interpretation_text(value: Mapping[str, Any]) -> None:
    """Keep human prose qualitative and absolute-index-free outside lineage."""

    allowed_index_path = (
        "narratives",
        "layer_trajectory_interpretation",
        "text",
    )
    for path, text in _iter_human_interpretation_strings(value):
        display_path = ".".join(path)
        if _FREE_TEXT_METRIC_LITERAL.search(text):
            raise ValueError(
                "human interpretation cannot hand-copy a metric literal; "
                f"use admitted evaluator tables: {display_path}"
            )
        if path != allowed_index_path and _ABSOLUTE_INTERNAL_INDEX.search(text):
            raise ValueError(
                "absolute internal index is allowed only in trajectory lineage: "
                f"{display_path}"
            )


def _iter_human_interpretation_strings(
    value: Any, path: tuple[str, ...] = ()
):
    if isinstance(value, Mapping):
        for key, item in value.items():
            child_path = (*path, str(key))
            if str(key) in HUMAN_INTERPRETATION_TEXT_FIELDS:
                yield from _iter_strings(item, child_path)
            else:
                yield from _iter_human_interpretation_strings(item, child_path)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _iter_human_interpretation_strings(
                item, (*path, str(index))
            )


def _iter_strings(value: Any, path: tuple[str, ...]):
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _iter_strings(item, (*path, str(key)))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _iter_strings(item, (*path, str(index)))


def _bind_opportunity_evidence_identities(
    normalized: Mapping[str, Any],
    *,
    formal: Mapping[str, Any],
    completed_supplements: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Attach audited file identities to every finding and design reference."""

    admission = formal.get("evidence_admission")
    formal_deliverables = (
        admission.get("deliverables") if isinstance(admission, Mapping) else None
    )
    if not isinstance(formal_deliverables, Mapping):
        raise ValueError("formal report deliverable identities are missing")

    def bind_rows(rows: Any, *, label: str) -> list[dict[str, Any]]:
        if not isinstance(rows, list):
            raise ValueError(f"normalized {label} rows are missing")
        bound_rows = []
        for raw in rows:
            row = _require_mapping(raw, f"normalized {label}")
            identities = []
            for evidence_id in row.get("supporting_formal_deliverables", []):
                identity = formal_deliverables.get(evidence_id)
                if (
                    not isinstance(identity, Mapping)
                    or identity.get("status") != "completed"
                ):
                    raise ValueError(
                        f"{label} formal evidence identity is not completed: "
                        f"{evidence_id}"
                    )
                identities.append(
                    _normalized_evidence_identity(
                        evidence_id=evidence_id,
                        evidence_kind="formal_deliverable",
                        identity=identity,
                    )
                )
            for evidence_id in row.get("supporting_supplements", []):
                identity = completed_supplements.get(evidence_id)
                if (
                    not isinstance(identity, Mapping)
                    or identity.get("status") != "completed"
                ):
                    raise ValueError(
                        f"{label} supplemental evidence identity is not completed: "
                        f"{evidence_id}"
                    )
                identities.append(
                    _normalized_evidence_identity(
                        evidence_id=evidence_id,
                        evidence_kind="supplement",
                        identity=identity,
                    )
                )
            expected_ids = list(row.get("supporting_formal_deliverables", [])) + list(
                row.get("supporting_supplements", [])
            )
            if [identity["evidence_id"] for identity in identities] != expected_ids:
                raise ValueError(f"{label} evidence identity order or coverage drift")
            bound_rows.append({**row, "supporting_evidence_identities": identities})
        return bound_rows

    bound_findings = bind_rows(normalized.get("findings"), label="finding")
    bound_findings_by_id = {row["finding_id"]: row for row in bound_findings}
    raw_narratives = _exact_mapping(
        normalized, "narratives", REQUIRED_NARRATIVES
    )
    bound_narratives = {}
    for narrative_id in REQUIRED_NARRATIVES:
        row = _require_mapping(
            raw_narratives[narrative_id], f"normalized narrative.{narrative_id}"
        )
        identities = []
        seen_identities: set[tuple[str, str]] = set()
        for finding_id in row.get("supporting_findings", []):
            finding = bound_findings_by_id.get(str(finding_id))
            if finding is None:
                raise ValueError(
                    f"narrative finding identity is missing: {narrative_id}"
                )
            for identity in finding["supporting_evidence_identities"]:
                key = (identity["evidence_kind"], identity["evidence_id"])
                if key in seen_identities:
                    continue
                seen_identities.add(key)
                identities.append(dict(identity))
        if not identities:
            raise ValueError(
                f"narrative has no evidence identities: {narrative_id}"
            )
        bound_narratives[narrative_id] = {
            **row,
            "supporting_evidence_identities": identities,
        }
    bound_opportunities = bind_rows(
        normalized.get("optimization_opportunities"),
        label="opportunity",
    )
    expected_disposition_ids = tuple(
        sorted(set(formal_deliverables) | set(completed_supplements))
    )
    raw_disposition = _exact_mapping(
        normalized,
        "evidence_disposition",
        expected_disposition_ids,
    )
    bound_disposition = {}
    for evidence_id in expected_disposition_ids:
        row = _require_mapping(
            raw_disposition[evidence_id],
            f"normalized evidence_disposition.{evidence_id}",
        )
        evidence_kind = str(row.get("evidence_kind"))
        if evidence_kind == "formal_deliverable":
            identity = formal_deliverables.get(evidence_id)
        elif evidence_kind == "supplement":
            identity = completed_supplements.get(evidence_id)
        else:
            raise ValueError(
                f"invalid normalized evidence disposition kind: {evidence_id}"
            )
        if not isinstance(identity, Mapping) or identity.get("status") != "completed":
            raise ValueError(
                f"evidence disposition identity is not completed: {evidence_id}"
            )
        bound_disposition[evidence_id] = {
            **row,
            "evidence_identity": _normalized_evidence_identity(
                evidence_id=evidence_id,
                evidence_kind=evidence_kind,
                identity=identity,
            ),
        }
    return {
        **normalized,
        "narratives": bound_narratives,
        "findings": bound_findings,
        "evidence_disposition": bound_disposition,
        "optimization_opportunities": bound_opportunities,
    }


def _normalized_evidence_identity(
    *, evidence_id: str, evidence_kind: str, identity: Mapping[str, Any]
) -> dict[str, str]:
    path = _nonempty_string(identity.get("path"), f"{evidence_id} evidence path")
    sha256 = _nonempty_string(
        identity.get("sha256"), f"{evidence_id} evidence sha256"
    )
    if re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
        raise ValueError(f"invalid evidence SHA-256: {evidence_id}")
    return {
        "evidence_id": evidence_id,
        "evidence_kind": evidence_kind,
        "path": path,
        "sha256": sha256,
    }


def _audit_formal_report(payload: Mapping[str, Any]) -> None:
    if payload.get("analysis_type") != REPORT_ANALYSIS_TYPE or payload.get(
        "status"
    ) != "completed":
        raise ValueError("formal deep-dive report is not a completed admitted report")
    admission = payload.get("evidence_admission")
    if not isinstance(admission, Mapping) or admission.get(
        "source_test_content_read_by_closeout_audit"
    ) is not False:
        raise ValueError("formal report source-test boundary differs")
    census = payload.get("execution_census")
    if not isinstance(census, Mapping) or census.get("completed_deliverables") != len(
        EXPECTED_DELIVERABLES
    ):
        raise ValueError("formal report does not contain all 19 deliverables")
    primary_rows = payload.get("primary_loss_attribution")
    if not isinstance(primary_rows, list) or {
        str(row.get("method_id"))
        for row in primary_rows
        if isinstance(row, Mapping)
    } != {MODEL_IDS[2], MODEL_IDS[3]}:
        raise ValueError("formal report primary attribution coverage differs")
    for row in primary_rows:
        if (
            not isinstance(row, Mapping)
            or row.get("component_erasure_boundary_established") is not False
            or row.get("history_token_flow_directly_observed_by_layer_scan") is not False
            or row.get("exact_layer_index_is_architecture_evidence") is not False
        ):
            raise ValueError("formal report overstates the layer-scan erasure boundary")
    _audit_formal_layer_profiles(payload)


def _audit_formal_layer_profiles(payload: Mapping[str, Any]) -> None:
    """Require every registered localization row before comprehensive rendering."""

    endpoints = ("target_margin", "ndcg@10")
    primary_models = tuple(MODEL_IDS[2:])
    profile = payload.get("layerwise_attenuation_profile")
    if not isinstance(profile, Mapping):
        raise ValueError("formal report layerwise attenuation profile is missing")
    shape_rows = profile.get("shape_summary")
    all_rows = profile.get("all_layer_rows")
    adjacent_rows = profile.get("adjacent_layer_rows")
    if not all(isinstance(rows, list) for rows in (shape_rows, all_rows, adjacent_rows)):
        raise ValueError("formal report layerwise profile rows are missing")
    expected_shape = {
        (model_id, endpoint)
        for model_id in primary_models
        for endpoint in endpoints
    }
    observed_shape = {
        (str(row.get("method_id")), str(row.get("endpoint")))
        for row in shape_rows
        if isinstance(row, Mapping)
    }
    if len(shape_rows) != len(expected_shape) or observed_shape != expected_shape:
        raise ValueError("formal report layerwise shape coverage differs")
    if any(
        not isinstance(row, Mapping)
        or row.get("exact_layer_index_is_architecture_evidence") is not False
        or row.get("layer_scan_alone_authorizes_design") is not False
        for row in shape_rows
    ):
        raise ValueError("formal report layerwise shape overstates design authority")

    def audit_layer_rows(
        rows: list[Any], *, blocks: Sequence[int], label: str
    ) -> None:
        expected = {
            (model_id, endpoint, block)
            for model_id in primary_models
            for endpoint in endpoints
            for block in blocks
        }
        observed = {
            (
                str(row.get("method_id")),
                str(row.get("endpoint")),
                row.get("block_zero_based"),
            )
            for row in rows
            if isinstance(row, Mapping)
        }
        if len(rows) != len(expected) or observed != expected:
            raise ValueError(f"formal report {label} coverage differs")
        if any(
            not isinstance(row, Mapping)
            or row.get("exact_layer_index_is_architecture_evidence") is not False
            or row.get("used_as_primary_component_attribution") is not False
            for row in rows
        ):
            raise ValueError(f"formal report {label} overstates component authority")

    audit_layer_rows(
        all_rows,
        blocks=POSTBLOCK_BLOCKS,
        label="all-layer profile",
    )
    audit_layer_rows(
        adjacent_rows,
        blocks=POSTBLOCK_BLOCKS[1:],
        label="adjacent-layer profile",
    )

    transition_profile = payload.get("attenuation_transition_profile")
    if not isinstance(transition_profile, Mapping) or not isinstance(
        transition_profile.get("rows"), list
    ):
        raise ValueError("formal report adjacent-node profile is missing")
    transition_rows = transition_profile["rows"]
    contrasts = tuple(
        f"adjacent__{left}__to__{right}"
        for left, right in zip(SELECTED_NODES[:-1], SELECTED_NODES[1:])
    )
    expected_transitions = {
        (model_id, endpoint, contrast_id)
        for model_id in primary_models
        for endpoint in endpoints
        for contrast_id in contrasts
    }
    observed_transitions = {
        (
            str(row.get("method_id")),
            str(row.get("endpoint")),
            str(row.get("contrast_id")),
        )
        for row in transition_rows
        if isinstance(row, Mapping)
    }
    if (
        len(transition_rows) != len(expected_transitions)
        or observed_transitions != expected_transitions
    ):
        raise ValueError("formal report adjacent-node profile coverage differs")
    if any(
        not isinstance(row, Mapping)
        or row.get("literal_hidden_state_sign_reversal_claimed") is not False
        or row.get("used_as_primary_component_attribution") is not False
        for row in transition_rows
    ):
        raise ValueError("formal report adjacent-node profile overstates causality")


def _audit_comprehensive_against_formal(
    normalized: Mapping[str, Any], formal: Mapping[str, Any]
) -> None:
    """Prevent the comprehensive synthesis from changing admitted formal outcomes."""

    formal_component_rows = formal.get("component_evidence_matrix")
    if not isinstance(formal_component_rows, list):
        raise ValueError("formal report component evidence matrix is missing")
    formal_components = {
        str(row.get("component_id")): row
        for row in formal_component_rows
        if isinstance(row, Mapping)
    }
    if set(formal_components) != set(COMPONENT_IDS):
        raise ValueError("formal report component evidence coverage differs")

    comprehensive_components = _require_mapping(
        normalized.get("component_matrix"), "normalized component matrix"
    )
    for component_id, row in comprehensive_components.items():
        status = row.get("status")
        evidence_level = row.get("evidence_level")
        if status not in {"supported", "weakened"}:
            continue
        formal_row = formal_components[component_id]
        if status == "supported" and evidence_level != "S":
            continue
        if formal_row.get("status") != status:
            raise ValueError(
                "comprehensive component changes the formal outcome: "
                f"{component_id}"
            )
        formal_scope = set(_model_scope(formal_row.get("model_scope")))
        if not set(row.get("model_scope", [])).issubset(formal_scope):
            raise ValueError(
                "comprehensive component exceeds the formal model scope: "
                f"{component_id}"
            )
        formal_evidence = set(
            _string_list(
                formal_row.get("evidence_deliverables"),
                "formal component evidence_deliverables",
            )
        )
        comprehensive_evidence = {
            deliverable
            for finding in normalized.get("findings", [])
            if finding.get("finding_id") in set(row.get("supporting_findings", []))
            for deliverable in finding.get("supporting_formal_deliverables", [])
        }
        if not (formal_evidence & comprehensive_evidence):
            raise ValueError(
                "comprehensive component lacks a formal outcome-matched "
                f"deliverable: {component_id}"
            )

    formal_hypothesis_rows = formal.get("hypothesis_status_matrix")
    if not isinstance(formal_hypothesis_rows, list):
        raise ValueError("formal report hypothesis status matrix is missing")
    formal_hypotheses = {
        str(row.get("hypothesis_id")): row
        for row in formal_hypothesis_rows
        if isinstance(row, Mapping)
    }
    if set(formal_hypotheses) != set(HYPOTHESIS_IDS):
        raise ValueError("formal report hypothesis coverage differs")
    comprehensive_hypotheses = _require_mapping(
        normalized.get("hypothesis_matrix"), "normalized hypothesis matrix"
    )
    for hypothesis_id, row in comprehensive_hypotheses.items():
        status = row.get("status")
        if status == "weakened":
            if formal_hypotheses[hypothesis_id].get("status") != status:
                raise ValueError(
                    "comprehensive weakened hypothesis changes the formal outcome: "
                    f"{hypothesis_id}"
                )
            continue
        if status == "rejected":
            if formal_hypotheses[hypothesis_id].get("status") != status:
                raise ValueError(
                    "comprehensive rejected hypothesis changes the formal outcome: "
                    f"{hypothesis_id}"
                )
            continue
        if row.get("evidence_level") != "S" or status != "supported":
            continue
        if formal_hypotheses[hypothesis_id].get("status") != status:
            raise ValueError(
                "comprehensive S-level hypothesis upgrades or contradicts the "
                f"formal outcome: {hypothesis_id}"
            )

    failure_mode = _require_mapping(
        normalized.get("failure_mode_diagnosis"), "normalized failure mode"
    )
    if failure_mode.get("causal_loss_of_use_claim_authorized") is True:
        formal_readout = formal_components["native_readout"]
        if formal_readout.get("status") != "supported" or not set(
            failure_mode.get("model_scope", [])
        ).issubset(set(_model_scope(formal_readout.get("model_scope")))):
            raise ValueError(
                "causal loss-of-use claim exceeds the formal native-readout outcome"
            )


def _audit_design_gate_payload(payload: Mapping[str, Any]) -> set[str]:
    if payload.get("analysis_type") != "transformer_component_design_gate_synthesis" or payload.get(
        "status"
    ) != "completed":
        raise ValueError("component design-gate synthesis is not completed")
    boundary = payload.get("claim_boundary")
    if not isinstance(boundary, Mapping) or boundary.get(
        "exact_layer_index_is_architecture_evidence"
    ) is not False or boundary.get("operator_necessity_authorized") is not False or boundary.get(
        "block_output_state_ceiling_authorizes_residual_operator_claim"
    ) is not False or boundary.get(
        "registered_behavior"
    ) != "harmful_full_history_target_margin_response" or boundary.get(
        "positive_neutral_removal_means_harm_reduction"
    ) is not True or boundary.get(
        "component_is_beneficial_for_transfer_authorized"
    ) is not False or boundary.get(
        "strengthen_or_preserve_component_authorized"
    ) is not False:
        raise ValueError("component design-gate claim boundary differs")
    cross = payload.get("cross_model_functional_support")
    if not isinstance(cross, Mapping):
        raise ValueError("component design-gate cross-model summary missing")
    if payload.get("models") != list(MODEL_IDS[2:]) or payload.get(
        "nodes"
    ) != list(DESIGN_NODE_COMPONENTS) or payload.get("primary_endpoint") != "target_margin":
        raise ValueError("component design-gate model/node/endpoint scope differs")
    lineage = payload.get("shared_parent_lineage")
    if not isinstance(lineage, Mapping) or set(lineage) != set(MODEL_IDS[2:]) or any(
        not isinstance(lineage[model_id], Mapping)
        or lineage[model_id].get("shared_parent_bytes_verified") is not True
        or lineage[model_id].get("exact_layer_index_is_architecture_evidence")
        not in {None, False}
        for model_id in MODEL_IDS[2:]
    ):
        raise ValueError("component design-gate shared-parent lineage differs")
    rows = payload.get("rows")
    expected_keys = {
        (model_id, node)
        for model_id in MODEL_IDS[2:]
        for node in DESIGN_NODE_COMPONENTS
    }
    if not isinstance(rows, list) or len(rows) != len(expected_keys):
        raise ValueError("component design-gate row coverage differs")
    by_key = {
        (str(row.get("method_id")), str(row.get("node"))): row
        for row in rows
        if isinstance(row, Mapping)
    }
    if set(by_key) != expected_keys or len(by_key) != len(rows):
        raise ValueError("component design-gate row identities differ")
    for row in by_key.values():
        node = str(row.get("node"))
        if row.get("claim_role") != DESIGN_NODE_CLAIM_ROLES[node]:
            raise ValueError("component design-gate claim role differs")
        if type(row.get("primary_target_margin_component_state_gate_passed")) is not bool or type(
            row.get("primary_target_margin_design_gate_passed")
        ) is not bool:
            raise ValueError("component design-gate row flags are not booleans")
        endpoint_gates = row.get("endpoint_gates")
        target_gate = (
            endpoint_gates.get("target_margin")
            if isinstance(endpoint_gates, Mapping)
            else None
        )
        if not isinstance(target_gate, Mapping) or any(
            type(target_gate.get(key)) is not bool
            for key in (
                "position_preserving_removal_gate_passed",
                "parent_same_request_sufficiency_gate_passed",
                "parent_history_specificity_gate_passed",
                "parent_cross_request_stress_gate_passed",
                "parent_direction_scale_controls_passed",
                "registered_component_state_gate_passed",
                "functional_node_design_target_eligible",
                "robust_design_prioritization_gate_passed",
            )
        ):
            raise ValueError("component design-gate target-margin cells differ")
        expected_design_eligibility = node != "block_output_residual"
        if (
            target_gate["functional_node_design_target_eligible"]
            is not expected_design_eligibility
        ):
            raise ValueError("component design-gate target eligibility differs")
        if (
            node == "block_output_residual"
            and row["primary_target_margin_design_gate_passed"]
        ):
            raise ValueError("block-output state ceiling cannot receive design priority")
        derived_state = bool(
            target_gate["position_preserving_removal_gate_passed"]
            and target_gate["parent_same_request_sufficiency_gate_passed"]
            and target_gate["parent_history_specificity_gate_passed"]
        )
        derived_design = bool(
            target_gate["functional_node_design_target_eligible"]
            and derived_state
            and target_gate["parent_cross_request_stress_gate_passed"]
            and target_gate["parent_direction_scale_controls_passed"]
        )
        if (
            target_gate["registered_component_state_gate_passed"] is not derived_state
            or target_gate["robust_design_prioritization_gate_passed"]
            is not derived_design
            or row["primary_target_margin_component_state_gate_passed"]
            is not derived_state
            or row["primary_target_margin_design_gate_passed"] is not derived_design
        ):
            raise ValueError("component design-gate row is not primitive-gate-derived")
        if row["primary_target_margin_design_gate_passed"] and not row[
            "primary_target_margin_component_state_gate_passed"
        ]:
            raise ValueError("component design priority lacks its state gate")
    derived_state_nodes = [
        node
        for node in DESIGN_NODE_COMPONENTS
        if all(
            by_key[(model_id, node)][
                "primary_target_margin_component_state_gate_passed"
            ]
            for model_id in MODEL_IDS[2:]
        )
    ]
    derived_design_nodes = [
        node
        for node in DESIGN_NODE_COMPONENTS
        if all(
            by_key[(model_id, node)][
                "primary_target_margin_design_gate_passed"
            ]
            for model_id in MODEL_IDS[2:]
        )
    ]
    nodes = cross.get("design_prioritized_nodes")
    if (
        cross.get("component_state_supported_nodes") != derived_state_nodes
        or nodes != derived_design_nodes
        or cross.get("any_shared_component_state_node")
        is not bool(derived_state_nodes)
        or cross.get("any_shared_design_prioritized_node")
        is not bool(derived_design_nodes)
        or cross.get("component_path_design_ranking_eligible")
        is not bool(derived_design_nodes)
    ):
        raise ValueError("component design-gate cross-model summary is not row-derived")
    if not isinstance(nodes, list) or not set(nodes).issubset(DESIGN_NODE_COMPONENTS):
        raise ValueError("component design-gate node coverage differs")
    return {str(node) for node in nodes}


def _build_component_bidirectional_gate_matrix(
    payload: Mapping[str, Any],
    *,
    evidence_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Expose every audited S/N/specificity/control/G primitive without effects."""

    design_nodes = _audit_design_gate_payload(payload)
    source = _normalized_evidence_identity(
        evidence_id=DESIGN_GATE_SUPPLEMENT,
        evidence_kind="supplement",
        identity=evidence_identity,
    )
    by_key = {
        (str(row["method_id"]), str(row["node"])): row
        for row in payload["rows"]
    }
    rows = []
    for model_id in MODEL_IDS[2:]:
        for node in DESIGN_NODE_COMPONENTS:
            source_row = by_key[(model_id, node)]
            gate = source_row["endpoint_gates"]["target_margin"]
            rows.append(
                {
                    "method_id": model_id,
                    "functional_node": node,
                    "claim_role": source_row["claim_role"],
                    "primary_endpoint": "target_margin",
                    "sufficiency_S_same_request": gate[
                        "parent_same_request_sufficiency_gate_passed"
                    ],
                    "history_specificity_same_minus_wrong": gate[
                        "parent_history_specificity_gate_passed"
                    ],
                    "necessity_N_position_preserving_removal": gate[
                        "position_preserving_removal_gate_passed"
                    ],
                    "cross_request_stress_control": gate[
                        "parent_cross_request_stress_gate_passed"
                    ],
                    "norm_direction_random_controls": gate[
                        "parent_direction_scale_controls_passed"
                    ],
                    "combined_component_state_gate": gate[
                        "registered_component_state_gate_passed"
                    ],
                    "functional_node_design_target_eligible": gate[
                        "functional_node_design_target_eligible"
                    ],
                    "design_G_gate": gate[
                        "robust_design_prioritization_gate_passed"
                    ],
                    "exact_layer_index_is_architecture_evidence": False,
                }
            )
    cross = payload["cross_model_functional_support"]
    return {
        "source": source,
        "primary_endpoint": "target_margin",
        "rows": rows,
        "cross_model": {
            "component_state_supported_nodes": list(
                cross["component_state_supported_nodes"]
            ),
            "design_prioritized_nodes": sorted(design_nodes),
            "any_shared_component_state_node": cross[
                "any_shared_component_state_node"
            ],
            "any_shared_design_prioritized_node": cross[
                "any_shared_design_prioritized_node"
            ],
            "component_path_design_ranking_eligible": cross[
                "component_path_design_ranking_eligible"
            ],
        },
        "interpretation_boundary": (
            "S is same-request state sufficiency; N is position-preserving removal of "
            "the registered harmful full-history response; G additionally requires "
            "wrong-user specificity, cross-request stress, and norm/direction/random "
            "controls in both primary models. These gates do not establish operator "
            "necessity, component benefit, ranking utility, or an exact-index design."
        ),
        "scientific_effect_values_recomputed": False,
    }


def _derive_necessity_component_models(
    payload: Mapping[str, Any]
) -> dict[str, set[str]]:
    """Derive N-level component scope from actual neutral-removal gate rows."""

    support = {component_id: set() for component_id in COMPONENT_IDS}
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("component design-gate rows missing for necessity derivation")
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("component design-gate necessity row is not an object")
        model_id = str(row.get("method_id"))
        node = str(row.get("node"))
        endpoint_gates = row.get("endpoint_gates")
        target_gate = (
            endpoint_gates.get("target_margin")
            if isinstance(endpoint_gates, Mapping)
            else None
        )
        if (
            model_id not in MODEL_IDS[2:]
            or node not in DESIGN_NODE_COMPONENTS
            or not isinstance(target_gate, Mapping)
        ):
            raise ValueError("component design-gate necessity scope differs")
        if target_gate.get("position_preserving_removal_gate_passed") is True:
            for component_id in DESIGN_NODE_COMPONENTS[node]:
                support[component_id].add(model_id)
    return support


def _exact_mapping(
    source: Mapping[str, Any], key: str, expected_keys: Sequence[str]
) -> Mapping[str, Any]:
    value = source.get(key)
    if not isinstance(value, Mapping) or set(value) != set(expected_keys):
        raise ValueError(f"{key} coverage drift")
    return value


def _require_list(source: Mapping[str, Any], key: str) -> list[Any]:
    value = source.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    return value


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{label} must be a finite number")
    return normalized


def _string_list(value: Any, label: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ValueError(f"{label} must be a {'possibly empty ' if allow_empty else ''}list")
    return [_nonempty_string(item, label) for item in value]


def _bounded_string_scope(
    value: Any, *, label: str, allowed: set[str], allow_empty: bool = False
) -> list[str]:
    values = _string_list(value, label, allow_empty=allow_empty)
    if len(values) != len(set(values)) or not set(values).issubset(allowed):
        raise ValueError(f"{label} contains duplicate or unknown values")
    return values


def _reference_list(
    row: Mapping[str, Any], key: str, admitted: set[str]
) -> list[str]:
    values = _string_list(row.get(key), key, allow_empty=True)
    if len(values) != len(set(values)) or not set(values).issubset(admitted):
        raise ValueError(f"{key} contains duplicate or unadmitted references")
    return values


def _finding_refs(row: Mapping[str, Any], finding_ids: set[str]) -> list[str]:
    values = _string_list(row.get("supporting_findings"), "supporting_findings", allow_empty=True)
    if len(values) != len(set(values)) or not set(values).issubset(finding_ids):
        raise ValueError("supporting_findings contains duplicate or unknown IDs")
    return values


def _finding_matches_component_model(
    finding: Mapping[str, Any],
    *,
    component_id: str,
    model_id: str,
    supplement_metadata: Mapping[str, Mapping[str, Any]],
) -> bool:
    """Return whether a finding directly covers one component/model cell."""

    if model_id not in finding.get("model_scope", []):
        return False
    for deliverable in finding.get("supporting_formal_deliverables", []):
        if deliverable not in COMPONENT_ALLOWED_DELIVERABLES[component_id]:
            continue
        if model_id in COMPONENT_DELIVERABLE_MODEL_COVERAGE[component_id].get(
            deliverable, set()
        ):
            return True
    for supplement_id in finding.get("supporting_supplements", []):
        metadata = supplement_metadata.get(supplement_id)
        if not isinstance(metadata, Mapping):
            continue
        if component_id in metadata.get("components", []) and model_id in metadata.get(
            "model_scope", []
        ):
            return True
    return False


def _model_scope(value: Any, *, allow_empty: bool = False) -> list[str]:
    values = _string_list(value, "model_scope", allow_empty=allow_empty)
    if len(values) != len(set(values)) or not set(values).issubset(MODEL_IDS):
        raise ValueError("model_scope contains duplicate or unknown models")
    return values


def _component_list(value: Any, *, allow_empty: bool = False) -> list[str]:
    values = _string_list(value, "component list", allow_empty=allow_empty)
    if len(values) != len(set(values)) or not set(values).issubset(COMPONENT_IDS):
        raise ValueError("component list contains duplicate or unknown components")
    return values


def _evidence_level(value: Any) -> str:
    level = str(value)
    if level not in EVIDENCE_LEVELS:
        raise ValueError(f"invalid evidence level: {level}")
    return level


def _dataset_scope(value: Any) -> str:
    if value != "kuaisearch_dev":
        raise ValueError("dataset_scope must be kuaisearch_dev")
    return str(value)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _resolve(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def _display_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _mean_ci_cell(cell: Mapping[str, Any]) -> str:
    interval = cell["query_cluster_ci95"]
    return f"{cell['mean']} [{interval[0]}, {interval[1]}]"


def _narrative_evidence_line(narrative: Mapping[str, Any]) -> str:
    identities = narrative.get("supporting_evidence_identities")
    evidence_bytes = ""
    if isinstance(identities, list) and identities:
        evidence_bytes = "; evidence bytes: " + ", ".join(
            f"{identity['evidence_id']}@{identity['sha256']}"
            for identity in identities
            if isinstance(identity, Mapping)
        )
    return (
        f"Narrative evidence：`{narrative['evidence_level']}`; findings: `"
        + "`, `".join(narrative["supporting_findings"])
        + "`; do not infer: "
        + "; ".join(narrative["do_not_infer"])
        + evidence_bytes
    )


def _atomic_write_pair(
    json_path: Path, json_text: str, markdown_path: Path, markdown_text: str
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    temporary: list[Path] = []
    try:
        for target, text in ((json_path, json_text), (markdown_path, markdown_text)):
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
                temporary.append(Path(handle.name))
        os.replace(temporary[0], json_path)
        os.replace(temporary[1], markdown_path)
    finally:
        for path in temporary:
            if path.exists():
                path.unlink()
