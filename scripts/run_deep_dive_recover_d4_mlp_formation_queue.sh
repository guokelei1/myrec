#!/usr/bin/env bash
set -euo pipefail

# The historical D4 watchers were chained to stale D5--D7 sentinels and never
# reached the registered MLP-formation runs.  This recovery queue does not
# preempt the active four-card wave: it waits for all four current bundles to
# complete, then fills the missing D4 lanes with the frozen observer.  The
# run IDs are the original registered IDs, so downstream N8--N16 watchers can
# consume them without a new outcome-dependent branch.
python_bin="/home/gkl/miniconda3/envs/pps-kuaisearch/bin/python"
standardized="data/standardized/kuaisearch/full_confirm_preceding40k_v11"

current_runs=(
  "runs/20260718_kuaisearch_mech_d5_q2_rope_b20_v1/metadata.json"
  "runs/20260718_kuaisearch_mech_d5_q2_rope_b27_v1/metadata.json"
  "runs/20260718_kuaisearch_mech_d2_q3_selected_branch_fold1_shard0of2_v1/metadata.json"
  "runs/20260718_kuaisearch_mech_d2_q3_selected_branch_fold1_shard1of2_v1/metadata.json"
)

wait_completed() {
  local path="$1"
  while true; do
    local status="missing"
    if [[ -f "$path" ]]; then status="$(jq -r '.status // "missing"' "$path")"; fi
    case "$status" in
      completed) return 0 ;;
      missing|initializing|running|wall_time_exhausted) sleep 30 ;;
      *) echo "D4 recovery upstream terminal status=$status path=$path" >&2; return 3 ;;
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

for path in "${current_runs[@]}"; do wait_completed "$path"; done
# Other already-registered D3--D7 handoff queues may release at the same
# upstream boundary.  Do not start the four D4 lanes until every physical card
# is genuinely idle; this preserves one-writer-per-GPU ownership.
wait_gpus_free

run_lane() {
  local lane="$1" gpu="$2"
  env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src \
    scripts/run_deep_dive_mlp_formation_lane.sh "$lane"
}

run_lane 0 0 & p0=$!
run_lane 1 1 & p1=$!
run_lane 2 2 & p2=$!
run_lane 3 3 & p3=$!
wait "$p0" "$p1" "$p2" "$p3"
