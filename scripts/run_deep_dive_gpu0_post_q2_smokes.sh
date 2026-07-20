#!/usr/bin/env bash
set -euo pipefail

python_bin="/home/gkl/miniconda3/envs/pps-kuaisearch/bin/python"
standardized="data/standardized/kuaisearch/full_confirm_preceding40k_v11"
q2_config="configs/methods/kuaisearch_motivation_v12_q2_recranker_generalqwen.yaml"
q3_config="configs/methods/kuaisearch_motivation_v12_q3_tallrec_generalqwen.yaml"
q0_config="configs/methods/kuaisearch_motivation_v12_q0_qwen3_reranker_06b.yaml"
q1_config="configs/methods/kuaisearch_motivation_v12_q1_instructrec_generalqwen.yaml"
q2_checkpoint="artifacts/motivation_v1_2/checkpoints/q2_recranker_generalqwen_seed20260714"
q3_checkpoint="artifacts/motivation_v1_2/checkpoints/q3_tallrec_generalqwen_seed20260714"
q0_checkpoint="artifacts/motivation_v1_2/checkpoints/q0_qwen3_reranker_06b_seed20260714"
q1_checkpoint="artifacts/motivation_v1_2/checkpoints/q1_instructrec_generalqwen_seed20260714"

"$python_bin" scripts/score_deep_dive_q3_native_readout.py \
  --standardized-dir "$standardized" \
  --config "$q3_config" \
  --checkpoint-root "$q3_checkpoint" \
  --run-id 20260718_kuaisearch_mech_d6_q3_native_readout_smoke_gpu_v1 \
  --device cuda:0 \
  --max-requests 1

"$python_bin" scripts/score_deep_dive_mlp_groups.py \
  --standardized-dir "$standardized" \
  --config "$q3_config" \
  --checkpoint-root "$q3_checkpoint" \
  --run-id 20260718_kuaisearch_mech_d4_q3_b13_smoke_gpu_v1 \
  --block 13 \
  --device cuda:0 \
  --max-rows 1

"$python_bin" scripts/score_deep_dive_mlp_groups.py \
  --standardized-dir "$standardized" \
  --config "$q2_config" \
  --checkpoint-root "$q2_checkpoint" \
  --run-id 20260718_kuaisearch_mech_d4_q2_b13_smoke_gpu_v1 \
  --block 13 \
  --device cuda:0 \
  --max-rows 1

"$python_bin" scripts/observe_deep_dive_attention_heads.py \
  --standardized-dir "$standardized" \
  --config "$q2_config" \
  --checkpoint-root "$q2_checkpoint" \
  --run-id 20260718_kuaisearch_mech_d3_q2_attention_heads_b13_smoke_gpu_v1 \
  --block 13 \
  --device cuda:0 \
  --max-rows 1

"$python_bin" scripts/observe_deep_dive_attention_heads.py \
  --standardized-dir "$standardized" \
  --config "$q3_config" \
  --checkpoint-root "$q3_checkpoint" \
  --run-id 20260718_kuaisearch_mech_d3_q3_attention_heads_b13_smoke_gpu_v1 \
  --block 13 \
  --device cuda:0 \
  --max-rows 1

"$python_bin" scripts/score_deep_dive_attention_groups.py \
  --standardized-dir "$standardized" \
  --config "$q2_config" \
  --checkpoint-root "$q2_checkpoint" \
  --run-id 20260718_kuaisearch_mech_d3_q2_attention_groups_b13_smoke_gpu_v1 \
  --block 13 \
  --device cuda:0 \
  --max-rows 1

"$python_bin" scripts/score_deep_dive_attention_groups.py \
  --standardized-dir "$standardized" \
  --config "$q3_config" \
  --checkpoint-root "$q3_checkpoint" \
  --run-id 20260718_kuaisearch_mech_d3_q3_attention_groups_b13_smoke_gpu_v1 \
  --block 13 \
  --device cuda:0 \
  --max-rows 1

"$python_bin" scripts/run_deep_dive_q2_optimizer_replay.py \
  --standardized-dir "$standardized" \
  --config "$q2_config" \
  --run-id 20260718_kuaisearch_mech_d7_q2_step501_replay_smoke_gpu_v1 \
  --device cuda:0 \
  --max-tasks 1

"$python_bin" scripts/run_deep_dive_q3_optimizer_replay.py \
  --standardized-dir "$standardized" \
  --config "$q3_config" \
  --run-id 20260718_kuaisearch_mech_d7_q3_step501_replay_smoke_gpu_v1 \
  --device cuda:0 \
  --max-tasks 1

for block in 13 27; do
  smoke_run="20260718_kuaisearch_mech_d2_q2_postblock_b${block}_reuse_smoke_v1"
  audit_run="20260718_kuaisearch_mech_d2_q2_postblock_b${block}_reuse_audit_v1"
  "$python_bin" scripts/score_deep_dive_postblock_sweep.py \
    --standardized-dir "$standardized" \
    --config "$q2_config" \
    --checkpoint-root "$q2_checkpoint" \
    --run-id "$smoke_run" \
    --block "$block" \
    --fold 0 \
    --device cuda:0 \
    --max-requests 32
  "$python_bin" scripts/audit_deep_dive_q2_postblock_reuse.py \
    --standardized-dir "$standardized" \
    --smoke-bundle "runs/$smoke_run" \
    --identity-dir "runs/20260717_kuaisearch_mech_m2_q2_patch_identity_b${block}" \
    --same-dir "runs/20260717_kuaisearch_mech_m2_q2_patch_same_b${block}" \
    --cross-dir "runs/20260717_kuaisearch_mech_m2_q2_patch_cross_b${block}" \
    --output-dir "runs/$audit_run" \
    --analysis-run-id "$audit_run" \
    --block "$block"
done

for model in q2 q3; do
  if [[ "$model" == q2 ]]; then
    config="$q2_config"
    checkpoint="$q2_checkpoint"
  else
    config="$q3_config"
    checkpoint="$q3_checkpoint"
  fi
  "$python_bin" scripts/score_deep_dive_rope.py \
    --standardized-dir "$standardized" \
    --config "$config" \
    --checkpoint-root "$checkpoint" \
    --run-id "20260718_kuaisearch_mech_d5_${model}_rope_b13_smoke_gpu_v1" \
    --block 13 \
    --device cuda:0 \
    --max-requests 1
  "$python_bin" scripts/score_deep_dive_contextual_controls.py \
    --standardized-dir "$standardized" \
    --config "$config" \
    --checkpoint-root "$checkpoint" \
    --run-id "20260718_kuaisearch_mech_d5_${model}_context_smoke_gpu_v1" \
    --device cuda:0 \
    --max-requests 1
done

"$python_bin" scripts/score_deep_dive_q0_branches.py \
  --standardized-dir "$standardized" \
  --config "$q0_config" \
  --checkpoint-root "$q0_checkpoint" \
  --run-id 20260718_kuaisearch_mech_d6_q0_branch_b13_smoke_gpu_v1 \
  --block 13 \
  --device cuda:0 \
  --max-requests 1

"$python_bin" scripts/score_deep_dive_breadth_readout.py \
  --standardized-dir "$standardized" \
  --config "$q0_config" \
  --checkpoint-root "$q0_checkpoint" \
  --run-id 20260718_kuaisearch_mech_d6_q0_final_readout_smoke_gpu_v1 \
  --device cuda:0 \
  --max-requests 1

"$python_bin" scripts/score_deep_dive_breadth_readout.py \
  --standardized-dir "$standardized" \
  --config "$q1_config" \
  --checkpoint-root "$q1_checkpoint" \
  --run-id 20260718_kuaisearch_mech_d6_q1_final_readout_smoke_gpu_v1 \
  --device cuda:0 \
  --max-requests 1

"$python_bin" scripts/score_deep_dive_q1_branches.py \
  --standardized-dir "$standardized" \
  --config "$q1_config" \
  --checkpoint-root "$q1_checkpoint" \
  --run-id 20260718_kuaisearch_mech_d6_q1_branch_b13_smoke_gpu_v1 \
  --block 13 \
  --device cuda:0 \
  --max-requests 1
