#!/usr/bin/env bash
set -euo pipefail

python_bin="/home/gkl/miniconda3/envs/pps-kuaisearch/bin/python"
gate="runs/20260718_kuaisearch_mech_d2_q3_native_gate_eval_v1/metrics.json"
q2_selection="runs/20260718_kuaisearch_mech_d2_q2_postblock_fold0_selection_v1/selection.json"
q2_confirmation="runs/20260718_kuaisearch_mech_d2_q2_postblock_fold1_confirmation_v1/metrics.json"
q2_contract="runs/20260718_kuaisearch_mech_d2_q2_selected_branch_contract_v1/contract.json"
q2_selected="runs/20260718_kuaisearch_mech_d2_q2_selected_branch_eval_v1/metrics.json"
q3_selection="runs/20260718_kuaisearch_mech_d2_q3_postblock_fold0_selection_v1/selection.json"
q3_confirmation="runs/20260718_kuaisearch_mech_d2_q3_postblock_fold1_confirmation_v1/metrics.json"
q3_contract="runs/20260718_kuaisearch_mech_d2_q3_selected_branch_contract_v1/contract.json"
q3_selected="runs/20260718_kuaisearch_mech_d2_q3_selected_branch_eval_v1/metrics.json"

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

wait_contract() {
  local path="$1"
  while [[ ! -f "$path" ]]; do sleep 30; done
  if [[ "$(jq -r '.status // "missing"' "$path")" != completed ]]; then
    echo "selected branch contract is not completed: $path" >&2
    return 3
  fi
}

wait_completed "$gate"
wait_completed "$q2_selection"
wait_completed "$q2_confirmation"

postblock_args=(
  --q2-fold0-selection "$q2_selection"
  --q2-fold1-confirmation "$q2_confirmation"
)
q3_admitted="$(jq -r '.q3_sweep_admitted' "$gate")"
if [[ "$q3_admitted" == true ]]; then
  wait_completed "$q3_selection"
  wait_completed "$q3_confirmation"
  postblock_args+=(
    --q3-fold0-selection "$q3_selection"
    --q3-fold1-confirmation "$q3_confirmation"
  )
fi

postblock_run="20260718_kuaisearch_mech_d2_postblock_synthesis_v1"
"$python_bin" scripts/synthesize_deep_dive_postblock.py \
  "${postblock_args[@]}" \
  --output-dir "runs/$postblock_run" \
  --analysis-run-id "$postblock_run"

wait_contract "$q2_contract"
selected_args=()
if [[ "$(jq -r '.branch_scoring_eligible' "$q2_contract")" == true ]]; then
  wait_completed "$q2_selected"
  selected_args+=(--metrics q2_recranker_generalqwen "$q2_selected")
fi
if [[ "$q3_admitted" == true ]]; then
  wait_contract "$q3_contract"
  if [[ "$(jq -r '.branch_scoring_eligible' "$q3_contract")" == true ]]; then
    wait_completed "$q3_selected"
    selected_args+=(--metrics q3_tallrec_generalqwen "$q3_selected")
  fi
fi

selected_run="20260718_kuaisearch_mech_d2_selected_branch_synthesis_v1"
"$python_bin" scripts/synthesize_deep_dive_selected_branches.py \
  "${selected_args[@]}" \
  --output-dir "runs/$selected_run" \
  --analysis-run-id "$selected_run"
