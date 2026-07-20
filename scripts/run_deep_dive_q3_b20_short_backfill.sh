#!/usr/bin/env bash
set -euo pipefail

# Deterministic qrels-blind utilization backfill for physical GPU 3.  Lane 1
# finishes Q3 fold-0 block 26 before lane 0 can finish block 27.  Use only that
# waiting window for three already-registered short breadth jobs.  The main
# queue safely re-enters completed run IDs later.  Long edge/RoPE jobs are
# deliberately excluded so D2 fold-1 remains the scheduling priority.

python_bin="/home/gkl/miniconda3/envs/pps-kuaisearch/bin/python"
standardized="data/standardized/kuaisearch/full_confirm_preceding40k_v11"
config="configs/methods/kuaisearch_motivation_v12_q3_tallrec_generalqwen.yaml"
checkpoint="artifacts/motivation_v1_2/checkpoints/q3_tallrec_generalqwen_seed20260714"
b26_metadata="runs/20260718_kuaisearch_mech_d2_q3_postblock_b26_fold0_v1/metadata.json"
b27_metadata="runs/20260718_kuaisearch_mech_d2_q3_postblock_b27_fold0_v1/metadata.json"
b27_progress="runs/20260718_kuaisearch_mech_d2_q3_postblock_b27_fold0_v1/progress.json"

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

# Return success only when the registered b27 job has enough conservative
# remaining work for the next short task.  This reads mechanical progress and
# run-contract size only; it never opens scores, metrics, qrels, or effects.
b27_fraction_below() {
  local threshold="$1"
  if [[ ! -f "$b27_metadata" ]]; then
    return 0
  fi
  local status
  status="$(jq -r '.status // "missing"' "$b27_metadata")"
  case "$status" in
    completed) return 1 ;;
    missing|initializing) return 0 ;;
    running|wall_time_exhausted) ;;
    *) return 1 ;;
  esac
  if [[ ! -f "$b27_progress" ]]; then
    return 1
  fi
  local completed target
  completed="$(jq -r '.completed_requests // -1' "$b27_progress")"
  target="$(jq -r '.run_contract.target_requests // -1' "$b27_metadata")"
  awk -v completed="$completed" -v target="$target" -v threshold="$threshold" \
    'BEGIN { exit !(completed >= 0 && target > 0 && completed / target < threshold) }'
}

run_attention_heads() {
  local run_id="20260718_kuaisearch_mech_d3_q3_attention_heads_b20_v1"
  scripts/run_deep_dive_resume_loop.sh \
    "runs/$run_id/metadata.json" \
    "tmp/${run_id}_resume_loop.log" \
    -- \
    "$python_bin" scripts/observe_deep_dive_attention_heads.py \
      --standardized-dir "$standardized" \
      --config "$config" \
      --checkpoint-root "$checkpoint" \
      --run-id "$run_id" \
      --block 20 \
      --device cuda:0 \
      --max-wall-seconds 13500
}

run_attention_groups() {
  local run_id="20260718_kuaisearch_mech_d3_q3_attention_groups_b20_v1"
  scripts/run_deep_dive_resume_loop.sh \
    "runs/$run_id/metadata.json" \
    "tmp/${run_id}_resume_loop.log" \
    -- \
    "$python_bin" scripts/score_deep_dive_attention_groups.py \
      --standardized-dir "$standardized" \
      --config "$config" \
      --checkpoint-root "$checkpoint" \
      --run-id "$run_id" \
      --block 20 \
      --device cuda:0 \
      --max-wall-seconds 13500
}

run_mlp_groups() {
  local run_id="20260718_kuaisearch_mech_d4_q3_mlp_groups_b20_v1"
  scripts/run_deep_dive_resume_loop.sh \
    "runs/$run_id/metadata.json" \
    "tmp/${run_id}_resume_loop.log" \
    -- \
    "$python_bin" scripts/score_deep_dive_mlp_groups.py \
      --standardized-dir "$standardized" \
      --config "$config" \
      --checkpoint-root "$checkpoint" \
      --run-id "$run_id" \
      --block 20 \
      --device cuda:0 \
      --max-wall-seconds 13500
}

for smoke in \
  runs/20260718_kuaisearch_mech_d3_q3_attention_heads_b13_smoke_gpu_v2/metadata.json \
  runs/20260718_kuaisearch_mech_d3_q3_attention_groups_b13_smoke_gpu_v2/metadata.json \
  runs/20260718_kuaisearch_mech_d4_q3_mlp_groups_b13_smoke_gpu_v2/metadata.json
do
  [[ "$(jq -r '.status // "missing"' "$smoke")" == completed ]]
done

wait_completed "$b26_metadata"

# At the observed stable runtimes, these thresholds leave approximately
# 20+ minutes of extra margin before lane 1 must begin fold 1.
if b27_fraction_below 0.90; then
  run_attention_heads
fi
if b27_fraction_below 0.25; then
  run_attention_groups
fi
if b27_fraction_below 0.75; then
  run_mlp_groups
fi
