#!/usr/bin/env bash
set -euo pipefail

# Qrels-blind critical-path backfill for physical GPU 3.  After the fixed Q3
# block-20 MLP-group job, lane 1 waits for Q3 fold-0 block 27 on physical GPU 1.
# If that upstream bundle still has a conservative time margin, advance the
# already-registered Q2 fold-1 block-27 bundle for one bounded 2,100-second
# attempt.  The canonical resume lock is shared with the main queue; the main
# Q2 lane later resumes the same atomic bundle and never reveals it early.

python_bin="/home/gkl/miniconda3/envs/pps-kuaisearch/bin/python"
standardized="data/standardized/kuaisearch/full_confirm_preceding40k_v11"
q2_config="configs/methods/kuaisearch_motivation_v12_q2_recranker_generalqwen.yaml"
q2_checkpoint="artifacts/motivation_v1_2/checkpoints/q2_recranker_generalqwen_seed20260714"
q3_config="configs/methods/kuaisearch_motivation_v12_q3_tallrec_generalqwen.yaml"
q3_checkpoint="artifacts/motivation_v1_2/checkpoints/q3_tallrec_generalqwen_seed20260714"
selection="runs/20260718_kuaisearch_mech_d2_q2_postblock_fold0_selection_v1/selection.json"
b20_mlp_metadata="runs/20260718_kuaisearch_mech_d4_q3_mlp_groups_b20_v1/metadata.json"
q3_b27_metadata="runs/20260718_kuaisearch_mech_d2_q3_postblock_b27_fold0_v1/metadata.json"
q3_b27_progress="runs/20260718_kuaisearch_mech_d2_q3_postblock_b27_fold0_v1/progress.json"
q2_run_id="20260718_kuaisearch_mech_d2_q2_postblock_b27_fold1_v1"
q2_metadata="runs/$q2_run_id/metadata.json"

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

# Read only mechanical progress and the frozen request count.  Any completed
# or terminal Q3 b27 state disables backfill and yields immediately to D2.
q3_b27_fraction_below() {
  local threshold="$1"
  if [[ ! -f "$q3_b27_metadata" || ! -f "$q3_b27_progress" ]]; then
    return 1
  fi
  local status
  status="$(jq -r '.status // "missing"' "$q3_b27_metadata")"
  case "$status" in
    running|wall_time_exhausted) ;;
    *) return 1 ;;
  esac
  local completed target
  completed="$(jq -r '.completed_requests // -1' "$q3_b27_progress")"
  target="$(jq -r '.run_contract.target_requests // -1' "$q3_b27_metadata")"
  awk -v completed="$completed" -v target="$target" -v threshold="$threshold" \
    'BEGIN { exit !(completed >= 0 && target > 0 && completed / target < threshold) }'
}

run_q2_b27_one_attempt() {
  local resume_args=()
  if [[ -f "$q2_metadata" ]]; then
    local status
    status="$(jq -r '.status // "missing"' "$q2_metadata")"
    case "$status" in
      completed) return 0 ;;
      wall_time_exhausted) resume_args+=(--resume) ;;
      *) echo "Q2 b27 is not one-shot resumable: $status" >&2; return 3 ;;
    esac
  fi

  local canonical_metadata lock_key lock_root lock_path lock_fd
  canonical_metadata="$(realpath -m "$q2_metadata")"
  lock_key="$(printf '%s' "$canonical_metadata" | sha256sum | cut -d' ' -f1)"
  lock_root="${MYREC_DEEP_DIVE_RESUME_LOCK_DIR:-$(realpath -m tmp/deep_dive_resume_locks)}"
  mkdir -p "$lock_root"
  lock_path="$lock_root/$lock_key.lock"
  lock_fd=""
  exec {lock_fd}> "$lock_path"
  if ! flock -n "$lock_fd"; then
    echo "Q2 b27 resume lock is busy: $canonical_metadata" >&2
    return 7
  fi

  "$python_bin" scripts/score_deep_dive_postblock_sweep.py \
    --standardized-dir "$standardized" \
    --config "$q2_config" \
    --checkpoint-root "$q2_checkpoint" \
    --run-id "$q2_run_id" \
    --block 27 \
    --fold 1 \
    --device cuda:0 \
    --max-wall-seconds 2100 \
    --fold0-selection "$selection" \
    "${resume_args[@]}" \
    >> "tmp/${q2_run_id}_one_attempt.log" 2>&1

  local final_status
  final_status="$(jq -r '.status // "missing"' "$q2_metadata")"
  case "$final_status" in
    completed|wall_time_exhausted) ;;
    *) echo "Q2 b27 one-shot terminal status=$final_status" >&2; return 6 ;;
  esac
}

run_q3_attention_heads() {
  local run_id="20260718_kuaisearch_mech_d3_q3_attention_heads_b27_v1"
  scripts/run_deep_dive_resume_loop.sh \
    "runs/$run_id/metadata.json" \
    "tmp/${run_id}_resume_loop.log" \
    -- \
    "$python_bin" scripts/observe_deep_dive_attention_heads.py \
      --standardized-dir "$standardized" \
      --config "$q3_config" \
      --checkpoint-root "$q3_checkpoint" \
      --run-id "$run_id" \
      --block 27 \
      --device cuda:0 \
      --max-wall-seconds 13500
}

for smoke in \
  runs/20260718_kuaisearch_mech_d2_q2_postblock_b13_reuse_smoke_v1/metadata.json \
  runs/20260718_kuaisearch_mech_d3_q3_attention_heads_b13_smoke_gpu_v2/metadata.json
do
  [[ "$(jq -r '.status // "missing"' "$smoke")" == completed ]]
done
[[ "$(jq -r '.status // "missing"' "$selection")" == completed ]]

wait_completed "$b20_mlp_metadata"

# Completed Q2 fold-1 bundles take about 4,525 seconds.  A bounded 2,100-second
# attempt admitted below 0.65 leaves more than twice the observed head runtime
# plus a substantial Q3 D2 handoff margin; it cannot loop into a second attempt.
if q3_b27_fraction_below 0.65; then
  run_q2_b27_one_attempt
fi
if q3_b27_fraction_below 0.90; then
  run_q3_attention_heads
fi
