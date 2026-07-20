#!/usr/bin/env bash
set -euo pipefail

python_bin="/home/gkl/miniconda3/envs/pps-kuaisearch/bin/python"
standardized="data/standardized/kuaisearch/full_confirm_preceding40k_v11"
config="configs/methods/kuaisearch_motivation_v12_q2_recranker_generalqwen.yaml"
checkpoint="artifacts/motivation_v1_2/checkpoints/q2_recranker_generalqwen_seed20260714"
contract_run="20260718_kuaisearch_mech_d2_q2_selected_branch_contract_v1"
selected_run="20260718_kuaisearch_mech_d2_q2_selected_branch_fold1_shard1of2_v1"

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

selected_eligible="$(jq -r '.branch_scoring_eligible' "runs/$contract_run/contract.json")"
if [[ "$selected_eligible" == true ]]; then
  selected_block="$(jq -r '.selected_block' "runs/$contract_run/contract.json")"
  if [[ ! "$selected_block" =~ ^(1[4-9]|2[0-7])$ ]]; then
    echo "invalid Q2 selected block: $selected_block" >&2
    exit 3
  fi
  selected_smoke="runs/20260718_kuaisearch_mech_d2_q2_selected_branch_b${selected_block}_smoke_gpu_v1/metadata.json"
  wait_completed "$selected_smoke"

  scripts/run_deep_dive_resume_loop.sh \
    "runs/$selected_run/metadata.json" \
    "tmp/${selected_run}_resume_loop.log" \
    -- \
    "$python_bin" scripts/score_deep_dive_selected_branches.py \
      --standardized-dir "$standardized" \
      --config "$config" \
      --checkpoint-root "$checkpoint" \
      --run-id "$selected_run" \
      --device cuda:0 \
      --branch-contract "runs/$contract_run/contract.json" \
      --request-shard-index 1 \
      --request-shard-count 2 \
      --max-wall-seconds 13500
elif [[ "$selected_eligible" != false ]]; then
  echo "invalid Q2 selected-branch eligibility: $selected_eligible" >&2
  exit 3
fi

exec scripts/run_deep_dive_q3_after_gate_queue.sh 3
