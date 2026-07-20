#!/usr/bin/env bash
set -euo pipefail

# Recovery scheduler for the registered D4 MLP-formation deliverable.  The
# original lane watchers were chained to obsolete predecessor IDs; this queue
# waits for the currently active four-card wave, then materializes the exact
# D4 run IDs consumed by the component-necessity evaluator.  It is CPU-only
# while waiting and uses all four GPUs only after they are released.
current_runs=(
  runs/20260718_kuaisearch_mech_d3_q2_attention_edges_b27_v1/metadata.json
  runs/20260718_kuaisearch_mech_d2_q3_postblock_b26_fold1_v1/metadata.json
  runs/20260718_kuaisearch_mech_d2_q3_postblock_b27_fold1_v1/metadata.json
  runs/20260718_kuaisearch_mech_d5_q2_rope_b20_v1/metadata.json
)

echo "D4 recovery queue waiting for current four-card wave to complete"
while true; do
  all_done=true
  for path in "${current_runs[@]}"; do
    status="missing"
    if [[ -f "$path" ]]; then status="$(jq -r '.status // "missing"' "$path")"; fi
    case "$status" in
      completed) ;;
      missing|initializing|running|wall_time_exhausted)
        all_done=false ;;
      *)
        echo "D4 recovery upstream terminal failure status=$status path=$path" >&2
        exit 3
        ;;
    esac
  done
  if [[ "$all_done" == true ]]; then break; fi
  sleep 30
done

echo "D4 recovery waiting for all physical GPUs to be free"
while true; do
  active="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | awk 'NF {count++} END {print count+0}')"
  if [[ "$active" == 0 ]]; then break; fi
  sleep 30
done

echo "D4 recovery starting four lanes"
for lane_gpu in 0 1 2 3; do
  env CUDA_VISIBLE_DEVICES="$lane_gpu" PYTHONPATH=src \
    scripts/run_deep_dive_mlp_formation_lane.sh "$lane_gpu" \
    > "tmp/20260720_d4_recovery_lane${lane_gpu}.log" 2>&1 &
  pids[$lane_gpu]=$!
done
for lane_gpu in 0 1 2 3; do wait "${pids[$lane_gpu]}"; done
echo "D4 recovery lanes completed; existing D4 evaluator watcher may now proceed"
