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
