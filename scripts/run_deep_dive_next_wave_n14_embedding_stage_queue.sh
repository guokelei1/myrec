#!/usr/bin/env bash
set -euo pipefail

# N14 isolates the model input embedding stage after the Q/K/V projection wave.
python_bin="/home/gkl/miniconda3/envs/pps-kuaisearch/bin/python"
standardized="data/standardized/kuaisearch/full_confirm_preceding40k_v11"
q2_config="configs/methods/kuaisearch_motivation_v12_q2_recranker_generalqwen.yaml"
q3_config="configs/methods/kuaisearch_motivation_v12_q3_tallrec_generalqwen.yaml"
q2_checkpoint="artifacts/motivation_v1_2/checkpoints/q2_recranker_generalqwen_seed20260714"
q3_checkpoint="artifacts/motivation_v1_2/checkpoints/q3_tallrec_generalqwen_seed20260714"
n13_eval="runs/20260720_kuaisearch_mech_n13_qkv_projection_eval_v1/metrics.json"

wait_completed() {
  local path="$1"
  while true; do
    local status="missing"
    if [[ -f "$path" ]]; then status="$(jq -r '.status // "missing"' "$path")"; fi
    case "$status" in
      completed) return 0 ;;
      missing|initializing|running|wall_time_exhausted) sleep 30 ;;
      *) echo "N14 upstream terminal status=$status path=$path" >&2; return 3 ;;
    esac
  done
}

run_one() {
  local config="$1" checkpoint="$2" gpu="$3" run_id="$4"
  env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src \
    scripts/run_deep_dive_resume_loop.sh \
      "runs/$run_id/metadata.json" "tmp/${run_id}_resume_loop.log" -- \
      "$python_bin" scripts/score_deep_dive_embedding_stage.py \
        --standardized-dir "$standardized" --config "$config" \
        --checkpoint-root "$checkpoint" --run-id "$run_id" \
        --device cuda:0 --max-wall-seconds 13500
}

echo "N14 embedding-stage queue waiting for N13 evaluator: $n13_eval"
wait_completed "$n13_eval"
run_one "$q2_config" "$q2_checkpoint" 0 20260720_kuaisearch_mech_n14_q2_embedding_stage_v1 & p0=$!
run_one "$q3_config" "$q3_checkpoint" 1 20260720_kuaisearch_mech_n14_q3_embedding_stage_v1 & p1=$!
wait "$p0" "$p1"

exec env PYTHONPATH=src "$python_bin" scripts/evaluate_deep_dive_embedding_stage.py \
  --standardized-dir "$standardized" \
  --q2-bundle runs/20260720_kuaisearch_mech_n14_q2_embedding_stage_v1 \
  --q3-bundle runs/20260720_kuaisearch_mech_n14_q3_embedding_stage_v1 \
  --output-dir runs/20260720_kuaisearch_mech_n14_embedding_stage_eval_v1 \
  --analysis-run-id 20260720_kuaisearch_mech_n14_embedding_stage_eval_v1
