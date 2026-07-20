#!/usr/bin/env bash
set -euo pipefail

# N10 rank-path follow-up.  It starts only after both N8 and N9 have passed
# their shared evaluators; it never preempts the four-card N8/N9 wave.
python_bin="/home/gkl/miniconda3/envs/pps-kuaisearch/bin/python"
standardized="data/standardized/kuaisearch/full_confirm_preceding40k_v11"
qrels_split="artifacts/motivation_transformer_deep_dive/frozen_controls/dev_qrels_folds_v1"
config="configs/methods/kuaisearch_motivation_v12_q3_tallrec_generalqwen.yaml"
checkpoint="artifacts/motivation_v1_2/checkpoints/q3_tallrec_generalqwen_seed20260714"
n8_eval="runs/20260720_kuaisearch_mech_n8_composition_eval_v1/metrics.json"
n9_eval="runs/20260720_kuaisearch_mech_n9_history_path_eval_v1/metrics.json"

echo "N10 rank queue waiting for N8/N9 evaluators: $n8_eval ; $n9_eval"

wait_completed() {
  local path="$1"
  while true; do
    local status="missing"
    if [[ -f "$path" ]]; then status="$(jq -r '.status // "missing"' "$path")"; fi
    case "$status" in
      completed) return 0 ;;
      missing|initializing|running|wall_time_exhausted) sleep 30 ;;
      *) echo "N10 rank upstream terminal status=$status path=$path" >&2; return 3 ;;
    esac
  done
}

wait_completed "$n8_eval"
wait_completed "$n9_eval"
run_id="20260720_kuaisearch_mech_n10_q3_lora_rank_paths_v1"
env CUDA_VISIBLE_DEVICES=0 \
  scripts/run_deep_dive_resume_loop.sh \
    "runs/$run_id/metadata.json" \
    "tmp/${run_id}_resume_loop.log" \
    -- \
    "$python_bin" scripts/score_deep_dive_q3_lora_rank_paths.py \
      --standardized-dir "$standardized" \
      --config "$config" \
      --checkpoint-root "$checkpoint" \
      --run-id "$run_id" \
      --device cuda:0 \
      --max-wall-seconds 13500

exec env PYTHONPATH=src "$python_bin" scripts/evaluate_deep_dive_q3_lora_rank_paths.py \
  --standardized-dir "$standardized" \
  --qrels-split-dir "$qrels_split" \
  --bundle "runs/$run_id" \
  --output-dir runs/20260720_kuaisearch_mech_n10_q3_lora_rank_eval_v1 \
  --analysis-run-id 20260720_kuaisearch_mech_n10_q3_lora_rank_eval_v1
