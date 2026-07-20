#!/usr/bin/env bash
set -euo pipefail

python_bin="/home/gkl/miniconda3/envs/pps-kuaisearch/bin/python"
standardized="data/standardized/kuaisearch/full_confirm_preceding40k_v11"
qrels_split="artifacts/motivation_transformer_deep_dive/frozen_controls/dev_qrels_folds_v1"
config="configs/methods/kuaisearch_motivation_v12_q2_recranker_generalqwen.yaml"
checkpoint="artifacts/motivation_v1_2/checkpoints/q2_recranker_generalqwen_seed20260714"
selection_run="20260718_kuaisearch_mech_d2_q2_postblock_fold0_selection_v1"
confirmation_run="20260718_kuaisearch_mech_d2_q2_postblock_fold1_confirmation_v1"
contract_run="20260718_kuaisearch_mech_d2_q2_selected_branch_contract_v1"
selected_run="20260718_kuaisearch_mech_d2_q2_selected_branch_fold1_v1"
selected_shard0_run="20260718_kuaisearch_mech_d2_q2_selected_branch_fold1_shard0of2_v1"
selected_shard1_run="20260718_kuaisearch_mech_d2_q2_selected_branch_fold1_shard1of2_v1"
selected_eval_run="20260718_kuaisearch_mech_d2_q2_selected_branch_eval_v1"

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

run_block() {
  local fold="$1"
  local block="$2"
  local run_id="20260718_kuaisearch_mech_d2_q2_postblock_b${block}_fold${fold}_v1"
  local args=()
  # The strict frozen M2 reuse gate failed on cross-device BF16 score identity,
  # so every registered block, including 13 and 27, is recomputed live.
  if [[ "$fold" == 1 ]]; then
    args+=(--fold0-selection "runs/$selection_run/selection.json")
  fi
  scripts/run_deep_dive_resume_loop.sh \
    "runs/$run_id/metadata.json" \
    "tmp/${run_id}_resume_loop.log" \
    -- \
    "$python_bin" scripts/score_deep_dive_postblock_sweep.py \
      --standardized-dir "$standardized" \
      --config "$config" \
      --checkpoint-root "$checkpoint" \
      --run-id "$run_id" \
      --block "$block" \
      --fold "$fold" \
      --device cuda:0 \
      --max-wall-seconds 13500 \
      "${args[@]}"
}

for block in $(seq 13 27); do
  run_block 0 "$block"
done

selection_args=()
for block in $(seq 13 27); do
  selection_args+=(
    --bundle "$block" "runs/20260718_kuaisearch_mech_d2_q2_postblock_b${block}_fold0_v1"
  )
done
"$python_bin" scripts/select_deep_dive_postblock_fold0.py \
  --standardized-dir "$standardized" \
  --qrels-split-dir "$qrels_split" \
  --method-id q2_recranker_generalqwen \
  "${selection_args[@]}" \
  --output-dir "runs/$selection_run" \
  --analysis-run-id "$selection_run"

for block in $(seq 13 27); do
  run_block 1 "$block"
done

confirmation_args=()
for block in $(seq 13 27); do
  confirmation_args+=(
    --bundle "$block" "runs/20260718_kuaisearch_mech_d2_q2_postblock_b${block}_fold1_v1"
  )
done
"$python_bin" scripts/confirm_deep_dive_postblock_fold1.py \
  --standardized-dir "$standardized" \
  --qrels-split-dir "$qrels_split" \
  --selection "runs/$selection_run/selection.json" \
  "${confirmation_args[@]}" \
  --output-dir "runs/$confirmation_run" \
  --analysis-run-id "$confirmation_run"

"$python_bin" scripts/materialize_deep_dive_selected_branch_contract.py \
  --selection "runs/$selection_run/selection.json" \
  --confirmation "runs/$confirmation_run/metrics.json" \
  --output "runs/$contract_run/contract.json"

selected_eligible="$(jq -r '.branch_scoring_eligible' "runs/$contract_run/contract.json")"
if [[ "$selected_eligible" != true && "$selected_eligible" != false ]]; then
  echo "invalid Q2 selected-branch eligibility: $selected_eligible" >&2
  exit 3
fi
if [[ "$selected_eligible" == true ]]; then
  selected_block="$(jq -r '.selected_block' "runs/$contract_run/contract.json")"
  if [[ ! "$selected_block" =~ ^(1[4-9]|2[0-7])$ ]]; then
    echo "invalid Q2 selected block: $selected_block" >&2
    exit 3
  fi
  selected_smoke_run="20260718_kuaisearch_mech_d2_q2_selected_branch_b${selected_block}_smoke_gpu_v1"
  "$python_bin" scripts/score_deep_dive_selected_branches.py \
    --standardized-dir "$standardized" \
    --config "$config" \
    --checkpoint-root "$checkpoint" \
    --run-id "$selected_smoke_run" \
    --device cuda:0 \
    --branch-contract "runs/$contract_run/contract.json" \
    --max-requests 1
  scripts/run_deep_dive_resume_loop.sh \
    "runs/$selected_shard0_run/metadata.json" \
    "tmp/${selected_shard0_run}_resume_loop.log" \
    -- \
    "$python_bin" scripts/score_deep_dive_selected_branches.py \
      --standardized-dir "$standardized" \
      --config "$config" \
      --checkpoint-root "$checkpoint" \
      --run-id "$selected_shard0_run" \
      --device cuda:0 \
      --branch-contract "runs/$contract_run/contract.json" \
      --request-shard-index 0 \
      --request-shard-count 2 \
      --max-wall-seconds 13500
  wait_completed "runs/$selected_shard1_run/metadata.json"
  "$python_bin" scripts/merge_deep_dive_selected_branch_shards.py \
    --standardized-dir "$standardized" \
    --shard "runs/$selected_shard0_run" \
    --shard "runs/$selected_shard1_run" \
    --output-dir "runs/$selected_run" \
    --analysis-run-id "$selected_run"
  "$python_bin" scripts/evaluate_deep_dive_selected_branches.py \
    --standardized-dir "$standardized" \
    --qrels-split-dir "$qrels_split" \
    --bundle "runs/$selected_run" \
    --output-dir "runs/$selected_eval_run" \
    --analysis-run-id "$selected_eval_run"
fi
