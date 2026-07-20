#!/usr/bin/env bash
set -euo pipefail

# N15 residual-composition queue.  It is CPU-only until N14 has a valid shared
# evaluator and every physical GPU is free.  Each four-card wave has one model
# process per card; the second block/branch wave never overlaps the first.
python_bin="/home/gkl/miniconda3/envs/pps-kuaisearch/bin/python"
standardized="data/standardized/kuaisearch/full_confirm_preceding40k_v11"
q2_config="configs/methods/kuaisearch_motivation_v12_q2_recranker_generalqwen.yaml"
q3_config="configs/methods/kuaisearch_motivation_v12_q3_tallrec_generalqwen.yaml"
q2_checkpoint="artifacts/motivation_v1_2/checkpoints/q2_recranker_generalqwen_seed20260714"
q3_checkpoint="artifacts/motivation_v1_2/checkpoints/q3_tallrec_generalqwen_seed20260714"
n14_eval="runs/20260720_kuaisearch_mech_n14_embedding_stage_eval_v1/metrics.json"

wait_completed() {
  local path="$1"
  while true; do
    local status="missing"
    if [[ -f "$path" ]]; then status="$(jq -r '.status // "missing"' "$path")"; fi
    case "$status" in
      completed) return 0 ;;
      missing|initializing|running|wall_time_exhausted) sleep 30 ;;
      *) echo "N15 upstream terminal status=$status path=$path" >&2; return 3 ;;
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
  local config="$1" checkpoint="$2" block="$3" branch="$4" gpu="$5" run_id="$6"
  env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src \
    scripts/run_deep_dive_resume_loop.sh \
      "runs/$run_id/metadata.json" "tmp/${run_id}_resume_loop.log" -- \
      "$python_bin" scripts/score_deep_dive_operator_stage.py \
        --standardized-dir "$standardized" --config "$config" \
        --checkpoint-root "$checkpoint" --run-id "$run_id" \
        --kind residual --block "$block" --branch "$branch" \
        --device cuda:0 --max-wall-seconds 13500
}

echo "N15 queue waiting for N14 evaluator: $n14_eval"
wait_completed "$n14_eval"
wait_gpus_free

# Two disjoint waves per branch cover all registered blocks.  The first wave
# uses all four cards; the final b27 pair reuses only two cards because no
# other registered N15 cell is available without duplicating a bundle.
run_pair_wave() {
  local first="$1" second="$2" branch="$3"
  run_one "$q2_config" "$q2_checkpoint" "$first" "$branch" 0 "20260720_kuaisearch_mech_n15_q2_${branch}_b${first}_v1" & p0=$!
  run_one "$q2_config" "$q2_checkpoint" "$second" "$branch" 1 "20260720_kuaisearch_mech_n15_q2_${branch}_b${second}_v1" & p1=$!
  run_one "$q3_config" "$q3_checkpoint" "$first" "$branch" 2 "20260720_kuaisearch_mech_n15_q3_${branch}_b${first}_v1" & p2=$!
  run_one "$q3_config" "$q3_checkpoint" "$second" "$branch" 3 "20260720_kuaisearch_mech_n15_q3_${branch}_b${second}_v1" & p3=$!
  wait "$p0" "$p1" "$p2" "$p3"
}

for branch in attention mlp; do
  run_pair_wave 13 20 "$branch"
  run_one "$q2_config" "$q2_checkpoint" 27 "$branch" 0 "20260720_kuaisearch_mech_n15_q2_${branch}_b27_v1" & p0=$!
  run_one "$q3_config" "$q3_checkpoint" 27 "$branch" 1 "20260720_kuaisearch_mech_n15_q3_${branch}_b27_v1" & p1=$!
  wait "$p0" "$p1"
done

for branch in attention mlp; do
  for block in 13 20 27; do
    env PYTHONPATH=src "$python_bin" scripts/evaluate_deep_dive_operator_stage.py \
      --standardized-dir "$standardized" --kind n15_residual_composition \
      --q2-bundle "runs/20260720_kuaisearch_mech_n15_q2_${branch}_b${block}_v1" \
      --q3-bundle "runs/20260720_kuaisearch_mech_n15_q3_${branch}_b${block}_v1" \
      --output-dir "runs/20260720_kuaisearch_mech_n15_${branch}_b${block}_eval_v1" \
      --analysis-run-id "20260720_kuaisearch_mech_n15_${branch}_b${block}_eval_v1"
  done
done
