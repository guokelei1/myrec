#!/usr/bin/env bash
set -euo pipefail

python_bin="/home/gkl/miniconda3/envs/pps-kuaisearch/bin/python"
necessity="runs/20260719_kuaisearch_mech_component_necessity_eval_v1/metrics.json"
selected="runs/20260718_kuaisearch_mech_d2_selected_branch_synthesis_v1/metrics.json"
analysis_run="20260719_kuaisearch_mech_component_design_synthesis_v1"

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

wait_completed "$necessity"
wait_completed "$selected"
exec "$python_bin" scripts/synthesize_deep_dive_component_design.py \
  --necessity-metrics "$necessity" \
  --selected-synthesis "$selected" \
  --output-dir "runs/$analysis_run" \
  --analysis-run-id "$analysis_run"
