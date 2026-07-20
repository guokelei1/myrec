#!/usr/bin/env bash
set -euo pipefail

# N20 is a Q1-only cache-phase diagnostic.  It waits for the fixed N17--N19
# routing/adapter closeout and for all physical GPUs to become idle.  The
# bundle itself remains qrels-blind and resumable; evaluation is a separate
# process after complete finite coverage and identity checks.
python_bin="/home/gkl/miniconda3/envs/pps-kuaisearch/bin/python"
standardized="data/standardized/kuaisearch/full_confirm_preceding40k_v11"
config="configs/methods/kuaisearch_motivation_v12_q1_instructrec_generalqwen.yaml"
checkpoint="artifacts/motivation_v1_2/checkpoints/q1_instructrec_generalqwen_seed20260714"
run_id="20260720_kuaisearch_mech_n20_q1_cache_phase_v1"
bundle="runs/$run_id"
gate="runs/20260720_kuaisearch_mech_n19_q3_v_b27_eval_v1/metrics.json"

wait_completed() {
  local path="$1"
  while true; do
    local status="missing"
    if [[ -f "$path" ]]; then status="$(jq -r '.status // "missing"' "$path")"; fi
    case "$status" in
      completed) return 0 ;;
      missing|initializing|running|wall_time_exhausted) sleep 30 ;;
      *) echo "N20 upstream terminal status=$status path=$path" >&2; return 3 ;;
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

echo "N20 queue waiting for N19 closeout: $gate"
wait_completed "$gate"
wait_gpus_free
env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
  scripts/run_deep_dive_resume_loop.sh \
    "${bundle}/metadata.json" "tmp/${run_id}_resume_loop.log" -- \
    "$python_bin" scripts/score_deep_dive_q1_cache_phase.py \
      --standardized-dir "$standardized" \
      --config "$config" \
      --checkpoint-root "$checkpoint" \
      --run-id "$run_id" \
      --device cuda:0 \
      --max-wall-seconds 13500

env PYTHONPATH=src "$python_bin" scripts/evaluate_deep_dive_q1_cache_phase.py \
  --standardized-dir "$standardized" \
  --bundle "$bundle" \
  --output-dir "${bundle}_eval_v1" \
  --analysis-run-id "20260720_kuaisearch_mech_n20_q1_cache_phase_eval_v1"

