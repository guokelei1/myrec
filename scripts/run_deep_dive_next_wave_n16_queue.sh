#!/usr/bin/env bash
set -euo pipefail

# N16 RMSNorm queue. It follows all N15 operator evaluations and then runs
# two Q2/Q3 scope pairs at a time across the four physical GPUs.
python_bin="/home/gkl/miniconda3/envs/pps-kuaisearch/bin/python"
standardized="data/standardized/kuaisearch/full_confirm_preceding40k_v11"
q2_config="configs/methods/kuaisearch_motivation_v12_q2_recranker_generalqwen.yaml"
q3_config="configs/methods/kuaisearch_motivation_v12_q3_tallrec_generalqwen.yaml"
q2_checkpoint="artifacts/motivation_v1_2/checkpoints/q2_recranker_generalqwen_seed20260714"
q3_checkpoint="artifacts/motivation_v1_2/checkpoints/q3_tallrec_generalqwen_seed20260714"

wait_completed() {
  local path="$1"
  while true; do
    local status="missing"
    if [[ -f "$path" ]]; then status="$(jq -r '.status // "missing"' "$path")"; fi
    case "$status" in
      completed) return 0 ;;
      missing|initializing|running|wall_time_exhausted) sleep 30 ;;
      *) echo "N16 upstream terminal status=$status path=$path" >&2; return 3 ;;
    esac
  done
}

wait_gpus_free() {
  while true; do
    active="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | awk 'NF {count++} END {print count+0}')"
    [[ "$active" == 0 ]] && return 0
    sleep 30
  done
}

run_one() {
  local config="$1" checkpoint="$2" scope="$3" block="$4" gpu="$5" run_id="$6"
  local block_arg=()
  if [[ "$scope" != "final" ]]; then block_arg=(--block "$block"); fi
  env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src \
    scripts/run_deep_dive_resume_loop.sh \
      "runs/$run_id/metadata.json" "tmp/${run_id}_resume_loop.log" -- \
      "$python_bin" scripts/score_deep_dive_operator_stage.py \
        --standardized-dir "$standardized" --config "$config" \
        --checkpoint-root "$checkpoint" --run-id "$run_id" \
        --kind rmsnorm --scope "$scope" "${block_arg[@]}" \
        --device cuda:0 --max-wall-seconds 13500
}

run_scope_pair() {
  local scope="$1" block="$2" gpu_q2="$3" gpu_q3="$4"
  local suffix="${scope}_b${block}"
  if [[ "$scope" == "final" ]]; then suffix="final"; fi
  run_one "$q2_config" "$q2_checkpoint" "$scope" "$block" "$gpu_q2" "20260720_kuaisearch_mech_n16_q2_${suffix}_v1" & p0=$!
  run_one "$q3_config" "$q3_checkpoint" "$scope" "$block" "$gpu_q3" "20260720_kuaisearch_mech_n16_q3_${suffix}_v1" & p1=$!
  wait "$p0" "$p1"
}

echo "N16 queue waiting for all N15 evaluations"
for branch in attention mlp; do
  for block in 13 20 27; do
    wait_completed "runs/20260720_kuaisearch_mech_n15_${branch}_b${block}_eval_v1/metrics.json"
  done
done
wait_gpus_free

# Two scope pairs per wave use all four cards; every physical card has one
# model process only. The final scope is a single remaining pair.
run_scope_pair input 13 0 1 & p0=$!
run_scope_pair input 20 2 3 & p1=$!
wait "$p0" "$p1"
run_scope_pair input 27 0 1 & p0=$!
run_scope_pair post_attention 13 2 3 & p1=$!
wait "$p0" "$p1"
run_scope_pair post_attention 20 0 1 & p0=$!
run_scope_pair post_attention 27 2 3 & p1=$!
wait "$p0" "$p1"
run_scope_pair final 0 1

for spec in input_b13 input_b20 input_b27 post_attention_b13 post_attention_b20 post_attention_b27 final; do
  if [[ "$spec" == final ]]; then scope=final; block_arg=(); else scope="${spec%_b*}"; block="${spec##*_b}"; block_arg=(--block "$block"); fi
  env PYTHONPATH=src "$python_bin" scripts/evaluate_deep_dive_operator_stage.py \
    --standardized-dir "$standardized" --kind n16_rmsnorm \
    --q2-bundle "runs/20260720_kuaisearch_mech_n16_q2_${spec}_v1" \
    --q3-bundle "runs/20260720_kuaisearch_mech_n16_q3_${spec}_v1" \
    --output-dir "runs/20260720_kuaisearch_mech_n16_${spec}_eval_v1" \
    --analysis-run-id "20260720_kuaisearch_mech_n16_${spec}_eval_v1"
done
