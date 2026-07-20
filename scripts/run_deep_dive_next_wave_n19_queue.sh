#!/usr/bin/env bash
set -euo pipefail

# N19 covers every registered Q3 q/v adapter path (28 blocks x 2 projections).
# It waits for all fixed N17/N18 evaluator outputs, then consumes four cards in
# disjoint two-block waves. No block or projection is selected by an effect.
python_bin="/home/gkl/miniconda3/envs/pps-kuaisearch/bin/python"
standardized="data/standardized/kuaisearch/full_confirm_preceding40k_v11"
q3_config="configs/methods/kuaisearch_motivation_v12_q3_tallrec_generalqwen.yaml"
q3_checkpoint="artifacts/motivation_v1_2/checkpoints/q3_tallrec_generalqwen_seed20260714"

wait_completed() {
  local path="$1"
  while true; do
    local status="missing"
    if [[ -f "$path" ]]; then status="$(jq -r '.status // "missing"' "$path")"; fi
    case "$status" in
      completed) return 0 ;;
      missing|initializing|running|wall_time_exhausted) sleep 30 ;;
      *) echo "N19 upstream terminal status=$status path=$path" >&2; return 3 ;;
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
  local component="$1" block="$2" gpu="$3" run_id="$4"
  env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src \
    scripts/run_deep_dive_resume_loop.sh \
      "runs/$run_id/metadata.json" "tmp/${run_id}_resume_loop.log" -- \
      "$python_bin" scripts/score_deep_dive_routing_boundary.py \
        --standardized-dir "$standardized" --config "$q3_config" \
        --checkpoint-root "$q3_checkpoint" --run-id "$run_id" \
        --kind lora_branch --block "$block" --component "$component" \
        --device cuda:0 --max-wall-seconds 13500
}

run_block_pair() {
  local first="$1" second="$2"
  run_one q "$first" 0 "20260720_kuaisearch_mech_n19_q3_q_b${first}_v1" & p0=$!
  run_one v "$first" 1 "20260720_kuaisearch_mech_n19_q3_v_b${first}_v1" & p1=$!
  run_one q "$second" 2 "20260720_kuaisearch_mech_n19_q3_q_b${second}_v1" & p2=$!
  run_one v "$second" 3 "20260720_kuaisearch_mech_n19_q3_v_b${second}_v1" & p3=$!
  wait "$p0" "$p1" "$p2" "$p3"
}

echo "N19 queue waiting for N17/N18 evaluator closeout"
for component in q k; do
  for block in 13 20 27; do
    wait_completed "runs/20260720_kuaisearch_mech_n17_${component}_b${block}_eval_v1/metrics.json"
  done
done
for block in 13 20 27; do
  wait_completed "runs/20260720_kuaisearch_mech_n18_b${block}_eval_v1/metrics.json"
done
wait_gpus_free

for first in $(seq 0 2 26); do
  second=$((first + 1))
  run_block_pair "$first" "$second"
done

for component in q v; do
  for block in $(seq 0 27); do
    bundle="runs/20260720_kuaisearch_mech_n19_q3_${component}_b${block}_v1"
    env PYTHONPATH=src "$python_bin" scripts/evaluate_deep_dive_q3_lora_branch.py \
      --standardized-dir "$standardized" --bundle "$bundle" \
      --output-dir "${bundle}_eval_v1" \
      --analysis-run-id "20260720_kuaisearch_mech_n19_q3_${component}_b${block}_eval_v1"
  done
done
