#!/usr/bin/env bash
set -euo pipefail

# N25 covers all four SwiGLU formation operators at blocks 13/20/27 for both
# Q2 and Q3. One model x one block occupies a lane (the scorer records all
# four operators in the same bundle), so two blocks x two models fill all four
# physical GPUs without duplicating identical operator passes.
python_bin="/home/gkl/miniconda3/envs/pps-kuaisearch/bin/python"
standardized="data/standardized/kuaisearch/full_confirm_preceding40k_v11"
q2_config="configs/methods/kuaisearch_motivation_v12_q2_recranker_generalqwen.yaml"
q3_config="configs/methods/kuaisearch_motivation_v12_q3_tallrec_generalqwen.yaml"
q2_checkpoint="artifacts/motivation_v1_2/checkpoints/q2_recranker_generalqwen_seed20260714"
q3_checkpoint="artifacts/motivation_v1_2/checkpoints/q3_tallrec_generalqwen_seed20260714"
gate="runs/20260720_kuaisearch_mech_n20_q1_cache_phase_v1_eval_v1/metrics.json"

wait_completed() {
  local path="$1"
  while true; do
    local status="missing"
    if [[ -f "$path" ]]; then status="$(jq -r '.status // "missing"' "$path")"; fi
    case "$status" in
      completed) return 0 ;;
      missing|initializing|running|wall_time_exhausted) sleep 30 ;;
      *) echo "N25 upstream terminal status=$status path=$path" >&2; return 3 ;;
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
  local config="$1" checkpoint="$2" block="$3" operator="$4" gpu="$5" run_id="$6"
  env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src \
    scripts/run_deep_dive_resume_loop.sh \
      "runs/$run_id/metadata.json" "tmp/${run_id}_resume_loop.log" -- \
      "$python_bin" scripts/score_deep_dive_swiglu_formation.py \
        --standardized-dir "$standardized" --config "$config" \
        --checkpoint-root "$checkpoint" --run-id "$run_id" \
        --block "$block" --operator "$operator" --device cuda:0 \
        --max-wall-seconds 13500
}

run_block_pair() {
  local first="$1" second="$2"
  run_one "$q2_config" "$q2_checkpoint" "$first" all 0 "20260720_kuaisearch_mech_n25_q2_all_b${first}_v1" & p0=$!
  run_one "$q3_config" "$q3_checkpoint" "$first" all 1 "20260720_kuaisearch_mech_n25_q3_all_b${first}_v1" & p1=$!
  run_one "$q2_config" "$q2_checkpoint" "$second" all 2 "20260720_kuaisearch_mech_n25_q2_all_b${second}_v1" & p2=$!
  run_one "$q3_config" "$q3_checkpoint" "$second" all 3 "20260720_kuaisearch_mech_n25_q3_all_b${second}_v1" & p3=$!
  wait "$p0" "$p1" "$p2" "$p3"
}

echo "N25 queue waiting for N20 evaluator: $gate"
wait_completed "$gate"
wait_gpus_free
run_block_pair 13 20
run_one "$q2_config" "$q2_checkpoint" 27 all 0 "20260720_kuaisearch_mech_n25_q2_all_b27_v1" & p0=$!
run_one "$q3_config" "$q3_checkpoint" 27 all 1 "20260720_kuaisearch_mech_n25_q3_all_b27_v1" & p1=$!
wait "$p0" "$p1"

for block in 13 20 27; do
  env PYTHONPATH=src "$python_bin" scripts/evaluate_deep_dive_operator_stage.py \
    --standardized-dir "$standardized" --kind n25_swiglu_formation \
    --q2-bundle "runs/20260720_kuaisearch_mech_n25_q2_all_b${block}_v1" \
    --q3-bundle "runs/20260720_kuaisearch_mech_n25_q3_all_b${block}_v1" \
    --output-dir "runs/20260720_kuaisearch_mech_n25_b${block}_eval_v1" \
    --analysis-run-id "20260720_kuaisearch_mech_n25_b${block}_eval_v1"
done
