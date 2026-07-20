#!/usr/bin/env bash
set -euo pipefail

# N17/N18 are fixed-boundary diagnostic waves. They start only after the
# N16 evaluator has closed, then use one model process per physical GPU and
# independent resumable output directories. No layer, component, or KV group
# is selected from observed effects.
python_bin="/home/gkl/miniconda3/envs/pps-kuaisearch/bin/python"
standardized="data/standardized/kuaisearch/full_confirm_preceding40k_v11"
q2_config="configs/methods/kuaisearch_motivation_v12_q2_recranker_generalqwen.yaml"
q3_config="configs/methods/kuaisearch_motivation_v12_q3_tallrec_generalqwen.yaml"
q2_checkpoint="artifacts/motivation_v1_2/checkpoints/q2_recranker_generalqwen_seed20260714"
q3_checkpoint="artifacts/motivation_v1_2/checkpoints/q3_tallrec_generalqwen_seed20260714"
n16_gate="runs/20260720_kuaisearch_mech_n16_final_eval_v1/metrics.json"

wait_completed() {
  local path="$1"
  while true; do
    local status="missing"
    if [[ -f "$path" ]]; then status="$(jq -r '.status // "missing"' "$path")"; fi
    case "$status" in
      completed) return 0 ;;
      missing|initializing|running|wall_time_exhausted) sleep 30 ;;
      *) echo "N17/N18 upstream terminal status=$status path=$path" >&2; return 3 ;;
    esac
  done
}

wait_gpus_free() {
  while true; do
    local active
    active="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | awk 'NF {count++} END {print count+0}')"
    [[ "$active" == 0 ]] && return 0
    sleep 30
  done
}

run_one() {
  local config="$1" checkpoint="$2" kind="$3" block="$4" component="$5" gpu="$6" run_id="$7"
  local component_arg=()
  if [[ "$kind" == "head_norm" ]]; then component_arg=(--component "$component"); fi
  env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src \
    scripts/run_deep_dive_resume_loop.sh \
      "runs/$run_id/metadata.json" "tmp/${run_id}_resume_loop.log" -- \
      "$python_bin" scripts/score_deep_dive_routing_boundary.py \
        --standardized-dir "$standardized" --config "$config" \
        --checkpoint-root "$checkpoint" --run-id "$run_id" \
        --kind "$kind" --block "$block" "${component_arg[@]}" \
        --device cuda:0 --max-wall-seconds 13500
}

run_n17_block() {
  local block="$1"
  run_one "$q2_config" "$q2_checkpoint" head_norm "$block" q 0 "20260720_kuaisearch_mech_n17_q2_q_b${block}_v1" & p0=$!
  run_one "$q3_config" "$q3_checkpoint" head_norm "$block" q 1 "20260720_kuaisearch_mech_n17_q3_q_b${block}_v1" & p1=$!
  run_one "$q2_config" "$q2_checkpoint" head_norm "$block" k 2 "20260720_kuaisearch_mech_n17_q2_k_b${block}_v1" & p2=$!
  run_one "$q3_config" "$q3_checkpoint" head_norm "$block" k 3 "20260720_kuaisearch_mech_n17_q3_k_b${block}_v1" & p3=$!
  wait "$p0" "$p1" "$p2" "$p3"
}

run_n18_pair() {
  local first="$1" second="$2"
  run_one "$q2_config" "$q2_checkpoint" gqa "$first" none 0 "20260720_kuaisearch_mech_n18_q2_b${first}_v1" & p0=$!
  run_one "$q3_config" "$q3_checkpoint" gqa "$first" none 1 "20260720_kuaisearch_mech_n18_q3_b${first}_v1" & p1=$!
  run_one "$q2_config" "$q2_checkpoint" gqa "$second" none 2 "20260720_kuaisearch_mech_n18_q2_b${second}_v1" & p2=$!
  run_one "$q3_config" "$q3_checkpoint" gqa "$second" none 3 "20260720_kuaisearch_mech_n18_q3_b${second}_v1" & p3=$!
  wait "$p0" "$p1" "$p2" "$p3"
}

echo "N17/N18 queue waiting for N16 final evaluator: $n16_gate"
wait_completed "$n16_gate"
wait_gpus_free

for block in 13 20 27; do run_n17_block "$block"; done
run_n18_pair 13 20
run_one "$q2_config" "$q2_checkpoint" gqa 27 none 0 "20260720_kuaisearch_mech_n18_q2_b27_v1" & p0=$!
run_one "$q3_config" "$q3_checkpoint" gqa 27 none 1 "20260720_kuaisearch_mech_n18_q3_b27_v1" & p1=$!
wait "$p0" "$p1"

for component in q k; do
  for block in 13 20 27; do
    env PYTHONPATH=src "$python_bin" scripts/evaluate_deep_dive_operator_stage.py \
      --standardized-dir "$standardized" --kind n17_head_norm \
      --q2-bundle "runs/20260720_kuaisearch_mech_n17_q2_${component}_b${block}_v1" \
      --q3-bundle "runs/20260720_kuaisearch_mech_n17_q3_${component}_b${block}_v1" \
      --output-dir "runs/20260720_kuaisearch_mech_n17_${component}_b${block}_eval_v1" \
      --analysis-run-id "20260720_kuaisearch_mech_n17_${component}_b${block}_eval_v1"
  done
done
for block in 13 20 27; do
  env PYTHONPATH=src "$python_bin" scripts/evaluate_deep_dive_operator_stage.py \
    --standardized-dir "$standardized" --kind n18_gqa_grouping \
    --q2-bundle "runs/20260720_kuaisearch_mech_n18_q2_b${block}_v1" \
    --q3-bundle "runs/20260720_kuaisearch_mech_n18_q3_b${block}_v1" \
    --output-dir "runs/20260720_kuaisearch_mech_n18_b${block}_eval_v1" \
    --analysis-run-id "20260720_kuaisearch_mech_n18_b${block}_eval_v1"
done
