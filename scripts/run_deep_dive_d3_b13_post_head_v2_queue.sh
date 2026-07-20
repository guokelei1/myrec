#!/usr/bin/env bash
set -euo pipefail

python_bin="/home/gkl/miniconda3/envs/pps-kuaisearch/bin/python"
standardized="data/standardized/kuaisearch/full_confirm_preceding40k_v11"
q2_config="configs/methods/kuaisearch_motivation_v12_q2_recranker_generalqwen.yaml"
q3_config="configs/methods/kuaisearch_motivation_v12_q3_tallrec_generalqwen.yaml"
q2_checkpoint="artifacts/motivation_v1_2/checkpoints/q2_recranker_generalqwen_seed20260714"
q3_checkpoint="artifacts/motivation_v1_2/checkpoints/q3_tallrec_generalqwen_seed20260714"
fixed_digest="40bffa52cca427d6756a3e972e1ecc7388e85729cca2cd7d12ded00e829de20d"
mlp_digest="0fbf6d77eddebdde602ec6a8af250cc05ebe61a94fd9ac99e8ded242368b2895"

run_resumable() {
  local run_id="$1"
  shift
  scripts/run_deep_dive_resume_loop.sh \
    "runs/$run_id/metadata.json" \
    "tmp/${run_id}_resume_loop.log" \
    -- \
    "$@"
}

wait_completed() {
  local path="$1"
  while true; do
    local status="missing"
    if [[ -f "$path" ]]; then
      status="$(jq -r '.status // "missing"' "$path")"
    fi
    case "$status" in
      completed) return 0 ;;
      missing|initializing|running|wall_time_exhausted) sleep 30 ;;
      *) echo "upstream terminal status=$status path=$path" >&2; return 3 ;;
    esac
  done
}

for model in q3 q2; do
  metadata="runs/20260718_kuaisearch_mech_d3_${model}_attention_heads_b13_v2/metadata.json"
  [[ "$(jq -r '.status' "$metadata")" == completed ]]
  [[ "$(jq -r '.complete_finite_observation_coverage' "$metadata")" == true ]]
  [[ "$(jq -r '.implementation_identity.digest' "$metadata")" == "$fixed_digest" ]]
done

for model in q3 q2; do
  metadata="runs/20260718_kuaisearch_mech_d4_${model}_mlp_groups_b13_smoke_gpu_v2/metadata.json"
  [[ "$(jq -r '.status' "$metadata")" == completed ]]
  [[ "$(jq -r '.identity_passed' "$metadata")" == true ]]
  [[ "$(jq -r '.implementation_identity.digest' "$metadata")" == "$mlp_digest" ]]
  [[ "$(jq -r '.permutation_recomposition_dtype' "$metadata")" == float32 ]]
  [[ "$(jq -r '.permutation_bound_reference_dtype' "$metadata")" == native_swiglu_product_dtype ]]
done

for model in q3 q2; do
  if [[ "$model" == q3 ]]; then
    model_config="$q3_config"
    model_checkpoint="$q3_checkpoint"
  else
    model_config="$q2_config"
    model_checkpoint="$q2_checkpoint"
  fi
  run_id="20260718_kuaisearch_mech_d4_${model}_mlp_groups_b13_v2"
  run_resumable "$run_id" \
    "$python_bin" scripts/score_deep_dive_mlp_groups.py \
      --standardized-dir "$standardized" \
      --config "$model_config" \
      --checkpoint-root "$model_checkpoint" \
      --run-id "$run_id" \
      --block 13 \
      --device cuda:0 \
      --max-wall-seconds 13500
done

wait_completed "runs/20260718_kuaisearch_mech_d3_q2_attention_groups_b13_smoke_gpu_v2/metadata.json"
wait_completed "runs/20260718_kuaisearch_mech_d3_q3_attention_groups_b13_smoke_gpu_v2/metadata.json"
for model in q3 q2; do
  if [[ "$model" == q3 ]]; then
    model_config="$q3_config"
    model_checkpoint="$q3_checkpoint"
  else
    model_config="$q2_config"
    model_checkpoint="$q2_checkpoint"
  fi
  run_id="20260718_kuaisearch_mech_d3_${model}_attention_groups_b13_v1"
  run_resumable "$run_id" \
    "$python_bin" scripts/score_deep_dive_attention_groups.py \
      --standardized-dir "$standardized" \
      --config "$model_config" \
      --checkpoint-root "$model_checkpoint" \
      --run-id "$run_id" \
      --block 13 \
      --device cuda:0 \
      --max-wall-seconds 13500
done

for model in q3 q2; do
  if [[ "$model" == q3 ]]; then
    model_config="$q3_config"
    model_checkpoint="$q3_checkpoint"
  else
    model_config="$q2_config"
    model_checkpoint="$q2_checkpoint"
  fi
  run_id="20260718_kuaisearch_mech_d5_${model}_rope_b13_v1"
  run_resumable "$run_id" \
    "$python_bin" scripts/score_deep_dive_rope.py \
      --standardized-dir "$standardized" \
      --config "$model_config" \
      --checkpoint-root "$model_checkpoint" \
      --run-id "$run_id" \
      --block 13 \
      --device cuda:0 \
      --max-wall-seconds 13500
done
