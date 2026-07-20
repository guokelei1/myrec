#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^(q2|q3)$ ]]; then
  echo "usage: $0 q2|q3" >&2
  exit 2
fi

short="$1"
python_bin="/home/gkl/miniconda3/envs/pps-kuaisearch/bin/python"
standardized="data/standardized/kuaisearch/full_confirm_preceding40k_v11"
if [[ "$short" == q2 ]]; then
  config="configs/methods/kuaisearch_motivation_v12_q2_recranker_generalqwen.yaml"
  checkpoint="artifacts/motivation_v1_2/checkpoints/q2_recranker_generalqwen_seed20260714"
  contract="runs/20260718_kuaisearch_mech_d2_q2_selected_branch_contract_v1/contract.json"
  parent="runs/20260718_kuaisearch_mech_d2_q2_selected_branch_fold1_v1"
  dependencies=(
    "runs/20260719_kuaisearch_mech_d4_q2_mlp_formation_b13_v1/metadata.json"
    "runs/20260719_kuaisearch_mech_d4_q2_mlp_formation_b20_v1/metadata.json"
  )
else
  config="configs/methods/kuaisearch_motivation_v12_q3_tallrec_generalqwen.yaml"
  checkpoint="artifacts/motivation_v1_2/checkpoints/q3_tallrec_generalqwen_seed20260714"
  contract="runs/20260718_kuaisearch_mech_d2_q3_selected_branch_contract_v1/contract.json"
  parent="runs/20260718_kuaisearch_mech_d2_q3_selected_branch_fold1_v1"
  dependencies=(
    "runs/20260719_kuaisearch_mech_d4_q3_mlp_formation_b13_v1/metadata.json"
    "runs/20260719_kuaisearch_mech_d4_q3_mlp_formation_b20_v1/metadata.json"
  )
fi
smoke_run="20260719_kuaisearch_mech_component_necessity_${short}_smoke_v1"
formal_run="20260719_kuaisearch_mech_component_necessity_${short}_v1"

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
      *) echo "required upstream terminal status=$status path=$path" >&2; return 3 ;;
    esac
  done
}

wait_lane_released() {
  local path="$1"
  while true; do
    local status="missing"
    if [[ -f "$path" ]]; then
      status="$(jq -r '.status // "missing"' "$path")"
    fi
    case "$status" in
      completed|mechanical_failure) return 0 ;;
      missing|initializing|running|wall_time_exhausted) sleep 30 ;;
      *) echo "unexpected lane predecessor status=$status path=$path" >&2; return 3 ;;
    esac
  done
}

while [[ ! -f "$contract" ]]; do sleep 30; done
if [[ "$(jq -r '.status // "missing"' "$contract")" != completed ]]; then
  echo "component-necessity contract is not completed: $contract" >&2
  exit 3
fi
if [[ "$(jq -r '.fold1_negative_transition_reproduced // false' "$contract")" != true ]]; then
  echo "component-necessity gate stopped for $short" >&2
  exit 0
fi
if [[ "$(jq -r '.evidence_role // "missing"' "$contract")" != \
  registered_confirmatory_branch_localization ]]; then
  echo "component-necessity contract role is not confirmatory: $contract" >&2
  exit 3
fi

wait_completed "$parent/metadata.json"
for dependency in "${dependencies[@]}"; do
  wait_lane_released "$dependency"
done

if [[ ! -f "runs/$smoke_run/metadata.json" ]]; then
  "$python_bin" scripts/score_deep_dive_component_necessity.py \
    --standardized-dir "$standardized" \
    --config "$config" \
    --checkpoint-root "$checkpoint" \
    --run-id "$smoke_run" \
    --device cuda:0 \
    --branch-contract "$contract" \
    --parent-selected-branch "$parent" \
    --max-requests 8 \
    --max-wall-seconds 13500
fi
wait_completed "runs/$smoke_run/metadata.json"

exec scripts/run_deep_dive_resume_loop.sh \
  "runs/$formal_run/metadata.json" \
  "tmp/${formal_run}_resume_loop.log" \
  -- \
  "$python_bin" scripts/score_deep_dive_component_necessity.py \
    --standardized-dir "$standardized" \
    --config "$config" \
    --checkpoint-root "$checkpoint" \
    --run-id "$formal_run" \
    --device cuda:0 \
    --branch-contract "$contract" \
    --parent-selected-branch "$parent" \
    --max-wall-seconds 13500
