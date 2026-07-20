#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 0 ]]; then
  echo "usage: $0" >&2
  exit 2
fi
contract="runs/20260718_kuaisearch_mech_d2_q2_selected_branch_contract_v1/contract.json"
selected_shard0="runs/20260718_kuaisearch_mech_d2_q2_selected_branch_fold1_shard0of2_v1/metadata.json"

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

[[ "$(jq -r '.status // "missing"' "$contract")" == completed ]]
eligible="$(jq -r '.branch_scoring_eligible' "$contract")"
if [[ "$eligible" == true ]]; then
  # shard0 is the only scorer owned by physical GPU0.  Once its immutable
  # metadata is terminal, lane2 may use disjoint preregistered breadth run IDs
  # while shard1/merge/evaluation continue elsewhere.
  wait_completed "$selected_shard0"
elif [[ "$eligible" != false ]]; then
  echo "invalid Q2 selected-branch eligibility: $eligible" >&2
  exit 3
fi

exec scripts/run_deep_dive_q3_after_gate_queue.sh 2
