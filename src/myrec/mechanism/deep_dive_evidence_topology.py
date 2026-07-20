"""Outcome-independent model/deliverable topology for deep-dive evidence."""

from __future__ import annotations


MODEL_IDS = (
    "q0_qwen3_reranker_06b",
    "q1_instructrec_generalqwen",
    "q2_recranker_generalqwen",
    "q3_tallrec_generalqwen",
)

# This is a protocol/topology declaration, not an outcome-derived coverage claim.
DELIVERABLE_MODEL_COVERAGE = {
    "d1_representation": {MODEL_IDS[2], MODEL_IDS[3]},
    "d2_q3_native_gate": {MODEL_IDS[3]},
    "d2_postblock": {MODEL_IDS[2], MODEL_IDS[3]},
    "d2_selected_branches": {MODEL_IDS[2], MODEL_IDS[3]},
    "d3_attention_edges": {MODEL_IDS[2], MODEL_IDS[3]},
    "d3_attention_heads": {MODEL_IDS[2], MODEL_IDS[3]},
    "d3_attention_groups": {MODEL_IDS[2], MODEL_IDS[3]},
    "d4_mlp_groups": {MODEL_IDS[2], MODEL_IDS[3]},
    "d5_context": {MODEL_IDS[2], MODEL_IDS[3]},
    "d5_rope": {MODEL_IDS[2], MODEL_IDS[3]},
    "d6_q2_native_readout": {MODEL_IDS[2]},
    "d6_q3_native_readout": {MODEL_IDS[3]},
    "d6_q0_trajectory": {MODEL_IDS[0]},
    "d6_q1_trajectory": {MODEL_IDS[1]},
    "d6_q0_q1_branches": {MODEL_IDS[0], MODEL_IDS[1]},
    "d6_q0_q1_readouts": {MODEL_IDS[0], MODEL_IDS[1]},
    "d7_q2_objective": {MODEL_IDS[2]},
    "d7_q3_lora_path": {MODEL_IDS[3]},
    "d7_optimizer_replay": {MODEL_IDS[2], MODEL_IDS[3]},
}
