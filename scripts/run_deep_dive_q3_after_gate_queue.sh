#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[0-3]$ ]]; then
  echo "usage: $0 LANE(0|1|2|3)" >&2
  exit 2
fi
lane="$1"
python_bin="/home/gkl/miniconda3/envs/pps-kuaisearch/bin/python"
standardized="data/standardized/kuaisearch/full_confirm_preceding40k_v11"
qrels_split="artifacts/motivation_transformer_deep_dive/frozen_controls/dev_qrels_folds_v1"
config="configs/methods/kuaisearch_motivation_v12_q3_tallrec_generalqwen.yaml"
checkpoint="artifacts/motivation_v1_2/checkpoints/q3_tallrec_generalqwen_seed20260714"
q2_config="configs/methods/kuaisearch_motivation_v12_q2_recranker_generalqwen.yaml"
q2_checkpoint="artifacts/motivation_v1_2/checkpoints/q2_recranker_generalqwen_seed20260714"
q0_config="configs/methods/kuaisearch_motivation_v12_q0_qwen3_reranker_06b.yaml"
q0_checkpoint="artifacts/motivation_v1_2/checkpoints/q0_qwen3_reranker_06b_seed20260714"
q1_config="configs/methods/kuaisearch_motivation_v12_q1_instructrec_generalqwen.yaml"
q1_checkpoint="artifacts/motivation_v1_2/checkpoints/q1_instructrec_generalqwen_seed20260714"
gate_metrics="runs/20260718_kuaisearch_mech_d2_q3_native_gate_eval_v1/metrics.json"
postblock_smoke="runs/20260718_kuaisearch_mech_d2_q3_postblock_b13_smoke_gpu_v1/metadata.json"
selection_run="20260718_kuaisearch_mech_d2_q3_postblock_fold0_selection_v1"
confirmation_run="20260718_kuaisearch_mech_d2_q3_postblock_fold1_confirmation_v1"
contract_run="20260718_kuaisearch_mech_d2_q3_selected_branch_contract_v1"
selected_run="20260718_kuaisearch_mech_d2_q3_selected_branch_fold1_v1"
selected_shard0_run="20260718_kuaisearch_mech_d2_q3_selected_branch_fold1_shard0of2_v1"
selected_shard1_run="20260718_kuaisearch_mech_d2_q3_selected_branch_fold1_shard1of2_v1"
selected_eval_run="20260718_kuaisearch_mech_d2_q3_selected_branch_eval_v1"

wait_completed() {
  local path="$1"
  while true; do
    local status="missing"
    if [[ -f "$path" ]]; then
      status="$(jq -r '.status // "missing"' "$path")"
    fi
    case "$status" in
      completed)
        return 0
        ;;
      missing|initializing|running|wall_time_exhausted)
        sleep 30
        ;;
      *)
        echo "upstream terminal status=$status path=$path" >&2
        return 3
        ;;
    esac
  done
}

wait_all_postblocks() {
  local fold="$1"
  for block in $(seq 13 27); do
    wait_completed "runs/20260718_kuaisearch_mech_d2_q3_postblock_b${block}_fold${fold}_v1/metadata.json"
  done
}

wait_selected_smoke() {
  local selected_block
  selected_block="$(jq -r '.selected_block' "runs/$contract_run/contract.json")"
  if [[ ! "$selected_block" =~ ^(1[4-9]|2[0-7])$ ]]; then
    echo "invalid Q3 selected block: $selected_block" >&2
    return 3
  fi
  wait_completed \
    "runs/20260718_kuaisearch_mech_d2_q3_selected_branch_b${selected_block}_smoke_gpu_v1/metadata.json"
}

run_selected_shard() {
  local shard_index="$1"
  local run_id="$selected_shard0_run"
  if [[ "$shard_index" == 1 ]]; then
    run_id="$selected_shard1_run"
  fi
  scripts/run_deep_dive_resume_loop.sh \
    "runs/$run_id/metadata.json" \
    "tmp/${run_id}_resume_loop.log" \
    -- \
    "$python_bin" scripts/score_deep_dive_selected_branches.py \
      --standardized-dir "$standardized" \
      --config "$config" \
      --checkpoint-root "$checkpoint" \
      --run-id "$run_id" \
      --device cuda:0 \
      --branch-contract "runs/$contract_run/contract.json" \
      --request-shard-index "$shard_index" \
      --request-shard-count 2 \
      --max-wall-seconds 13500
}

run_postblock() {
  local fold="$1"
  local block="$2"
  local run_id="20260718_kuaisearch_mech_d2_q3_postblock_b${block}_fold${fold}_v1"
  local selection_args=()
  if [[ "$fold" == 1 ]]; then
    selection_args+=(--fold0-selection "runs/$selection_run/selection.json")
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
      --q3-gate-metrics "$gate_metrics" \
      --max-wall-seconds 13500 \
      "${selection_args[@]}"
}

run_attention_breadth() {
  local block="$1"
  local run_id="20260718_kuaisearch_mech_d3_q3_attention_edges_b${block}_v1"
  scripts/run_deep_dive_resume_loop.sh \
    "runs/$run_id/metadata.json" \
    "tmp/q3_attention_b${block}_resume_loop.log" \
    -- \
    "$python_bin" scripts/score_deep_dive_attention_edges.py \
      --standardized-dir "$standardized" \
      --config "$config" \
      --checkpoint-root "$checkpoint" \
      --run-id "$run_id" \
      --block "$block" \
      --device cuda:0 \
      --max-wall-seconds 13500
}

run_q2_attention_breadth() {
  local block="$1"
  local run_id="20260718_kuaisearch_mech_d3_q2_attention_edges_b${block}_v1"
  scripts/run_deep_dive_resume_loop.sh \
    "runs/$run_id/metadata.json" \
    "tmp/q2_attention_b${block}_resume_loop.log" \
    -- \
    "$python_bin" scripts/score_deep_dive_attention_edges.py \
      --standardized-dir "$standardized" \
      --config "$q2_config" \
      --checkpoint-root "$q2_checkpoint" \
      --run-id "$run_id" \
      --block "$block" \
      --device cuda:0 \
      --max-wall-seconds 13500
}

run_attention_heads() {
  local model="$1"
  local block="$2"
  local model_config="$config"
  local model_checkpoint="$checkpoint"
  if [[ "$model" == q2 ]]; then
    model_config="$q2_config"
    model_checkpoint="$q2_checkpoint"
  fi
  local run_id="20260718_kuaisearch_mech_d3_${model}_attention_heads_b${block}_v1"
  scripts/run_deep_dive_resume_loop.sh \
    "runs/$run_id/metadata.json" \
    "tmp/${run_id}_resume_loop.log" \
    -- \
    "$python_bin" scripts/observe_deep_dive_attention_heads.py \
      --standardized-dir "$standardized" \
      --config "$model_config" \
      --checkpoint-root "$model_checkpoint" \
      --run-id "$run_id" \
      --block "$block" \
      --device cuda:0 \
      --max-wall-seconds 13500
}

run_attention_groups() {
  local model="$1"
  local block="$2"
  local model_config="$config"
  local model_checkpoint="$checkpoint"
  if [[ "$model" == q2 ]]; then
    model_config="$q2_config"
    model_checkpoint="$q2_checkpoint"
  fi
  local run_id="20260718_kuaisearch_mech_d3_${model}_attention_groups_b${block}_v1"
  scripts/run_deep_dive_resume_loop.sh \
    "runs/$run_id/metadata.json" \
    "tmp/${run_id}_resume_loop.log" \
    -- \
    "$python_bin" scripts/score_deep_dive_attention_groups.py \
      --standardized-dir "$standardized" \
      --config "$model_config" \
      --checkpoint-root "$model_checkpoint" \
      --run-id "$run_id" \
      --block "$block" \
      --device cuda:0 \
      --max-wall-seconds 13500
}

run_mlp_groups() {
  local model="$1"
  local block="$2"
  local model_config="$config"
  local model_checkpoint="$checkpoint"
  if [[ "$model" == q2 ]]; then
    model_config="$q2_config"
    model_checkpoint="$q2_checkpoint"
  fi
  local run_id="20260718_kuaisearch_mech_d4_${model}_mlp_groups_b${block}_v1"
  scripts/run_deep_dive_resume_loop.sh \
    "runs/$run_id/metadata.json" \
    "tmp/${run_id}_resume_loop.log" \
    -- \
    "$python_bin" scripts/score_deep_dive_mlp_groups.py \
      --standardized-dir "$standardized" \
      --config "$model_config" \
      --checkpoint-root "$model_checkpoint" \
      --run-id "$run_id" \
      --block "$block" \
      --device cuda:0 \
      --max-wall-seconds 13500
}

run_rope() {
  local model="$1"
  local block="$2"
  local model_config="$config"
  local model_checkpoint="$checkpoint"
  if [[ "$model" == q2 ]]; then
    model_config="$q2_config"
    model_checkpoint="$q2_checkpoint"
  fi
  local run_id="20260718_kuaisearch_mech_d5_${model}_rope_b${block}_v1"
  scripts/run_deep_dive_resume_loop.sh \
    "runs/$run_id/metadata.json" \
    "tmp/${run_id}_resume_loop.log" \
    -- \
    "$python_bin" scripts/score_deep_dive_rope.py \
      --standardized-dir "$standardized" \
      --config "$model_config" \
      --checkpoint-root "$model_checkpoint" \
      --run-id "$run_id" \
      --block "$block" \
      --device cuda:0 \
      --max-wall-seconds 13500
}

run_context() {
  local model="$1"
  local model_config="$config"
  local model_checkpoint="$checkpoint"
  if [[ "$model" == q2 ]]; then
    model_config="$q2_config"
    model_checkpoint="$q2_checkpoint"
  fi
  local run_id="20260718_kuaisearch_mech_d5_${model}_context_v1"
  scripts/run_deep_dive_resume_loop.sh \
    "runs/$run_id/metadata.json" \
    "tmp/${run_id}_resume_loop.log" \
    -- \
    "$python_bin" scripts/score_deep_dive_contextual_controls.py \
      --standardized-dir "$standardized" \
      --config "$model_config" \
      --checkpoint-root "$model_checkpoint" \
      --run-id "$run_id" \
      --device cuda:0 \
      --max-wall-seconds 13500
}

run_q3_native_readout() {
  wait_completed "runs/20260718_kuaisearch_mech_d6_q3_native_readout_smoke_gpu_v1/metadata.json"
  local run_id="20260718_kuaisearch_mech_d6_q3_native_readout_v1"
  scripts/run_deep_dive_resume_loop.sh \
    "runs/$run_id/metadata.json" \
    "tmp/${run_id}_resume_loop.log" \
    -- \
    "$python_bin" scripts/score_deep_dive_q3_native_readout.py \
      --standardized-dir "$standardized" \
      --config "$config" \
      --checkpoint-root "$checkpoint" \
      --run-id "$run_id" \
      --device cuda:0 \
      --max-wall-seconds 13500
}

run_q0_breadth() {
  for block in 13 20 27; do
    local branch_run="20260718_kuaisearch_mech_d6_q0_branch_b${block}_v1"
    scripts/run_deep_dive_resume_loop.sh \
      "runs/$branch_run/metadata.json" \
      "tmp/${branch_run}_resume_loop.log" \
      -- \
      "$python_bin" scripts/score_deep_dive_q0_branches.py \
        --standardized-dir "$standardized" \
        --config "$q0_config" \
        --checkpoint-root "$q0_checkpoint" \
        --run-id "$branch_run" \
        --block "$block" \
        --device cuda:0 \
        --max-wall-seconds 13500
  done
  for condition in full null; do
    local trajectory_run="20260718_kuaisearch_mech_d6_q0_${condition}_trajectory_v1"
    scripts/run_deep_dive_resume_loop.sh \
      "runs/$trajectory_run/metadata.json" \
      "tmp/${trajectory_run}_resume_loop.log" \
      -- \
      "$python_bin" scripts/extract_deep_dive_representations.py \
        --standardized-dir "$standardized" \
        --config "$q0_config" \
        --checkpoint-root "$q0_checkpoint" \
        --run-id "$trajectory_run" \
        --role dev_representation \
        --condition "$condition" \
        --device cuda:0 \
        --max-wall-seconds 13500
  done
  local readout_run="20260718_kuaisearch_mech_d6_q0_final_readout_v1"
  scripts/run_deep_dive_resume_loop.sh \
    "runs/$readout_run/metadata.json" \
    "tmp/${readout_run}_resume_loop.log" \
    -- \
    "$python_bin" scripts/score_deep_dive_breadth_readout.py \
      --standardized-dir "$standardized" \
      --config "$q0_config" \
      --checkpoint-root "$q0_checkpoint" \
      --run-id "$readout_run" \
      --device cuda:0 \
      --max-wall-seconds 13500
}

run_q1_breadth() {
  for block in 13 20 27; do
    local branch_run="20260718_kuaisearch_mech_d6_q1_branch_b${block}_v1"
    scripts/run_deep_dive_resume_loop.sh \
      "runs/$branch_run/metadata.json" \
      "tmp/${branch_run}_resume_loop.log" \
      -- \
      "$python_bin" scripts/score_deep_dive_q1_branches.py \
        --standardized-dir "$standardized" \
        --config "$q1_config" \
        --checkpoint-root "$q1_checkpoint" \
        --run-id "$branch_run" \
        --block "$block" \
        --device cuda:0 \
        --max-wall-seconds 13500
  done
  local trajectory_run="20260718_kuaisearch_mech_d6_q1_kv_trajectory_v1"
  scripts/run_deep_dive_resume_loop.sh \
    "runs/$trajectory_run/metadata.json" \
    "tmp/${trajectory_run}_resume_loop.log" \
    -- \
    "$python_bin" scripts/score_deep_dive_q1_trajectory.py \
      --standardized-dir "$standardized" \
      --config "$q1_config" \
      --checkpoint-root "$q1_checkpoint" \
      --run-id "$trajectory_run" \
      --device cuda:0 \
      --max-wall-seconds 13500
  local readout_run="20260718_kuaisearch_mech_d6_q1_final_readout_v1"
  scripts/run_deep_dive_resume_loop.sh \
    "runs/$readout_run/metadata.json" \
    "tmp/${readout_run}_resume_loop.log" \
    -- \
    "$python_bin" scripts/score_deep_dive_breadth_readout.py \
      --standardized-dir "$standardized" \
      --config "$q1_config" \
      --checkpoint-root "$q1_checkpoint" \
      --run-id "$readout_run" \
      --device cuda:0 \
      --max-wall-seconds 13500
}

run_optimizer_replay() {
  local model="$1"
  local model_config="$config"
  local runner="scripts/run_deep_dive_q3_optimizer_replay.py"
  if [[ "$model" == q2 ]]; then
    model_config="$q2_config"
    runner="scripts/run_deep_dive_q2_optimizer_replay.py"
  fi
  local run_id="20260718_kuaisearch_mech_d7_${model}_step501_replay_v1"
  scripts/run_deep_dive_resume_loop.sh \
    "runs/$run_id/metadata.json" \
    "tmp/${run_id}_resume_loop.log" \
    -- \
    "$python_bin" "$runner" \
      --standardized-dir "$standardized" \
      --config "$model_config" \
      --run-id "$run_id" \
      --device cuda:0 \
      --max-wall-seconds 13500
}

run_block_breadth_followup_model() {
  local model="$1"
  local block="$2"
  run_attention_heads "$model" "$block"
  run_attention_groups "$model" "$block"
  run_mlp_groups "$model" "$block"
  run_rope "$model" "$block"
}

# Lanes 2/3 are breadth-only continuations launched after Q2 selected-branch
# ownership is released on physical GPUs 0/2.  Their run IDs are disjoint from
# lanes 0/1, so no outcome-dependent work stealing or concurrent writes occur.
if [[ "$lane" == 2 ]]; then
  run_q2_attention_breadth 20
  run_block_breadth_followup_model q2 20
  run_q0_breadth
  run_optimizer_replay q2
  exit 0
elif [[ "$lane" == 3 ]]; then
  run_q2_attention_breadth 27
  run_block_breadth_followup_model q2 27
  run_q1_breadth
  run_optimizer_replay q3
  exit 0
fi

if [[ "$(jq -r '.q3_sweep_admitted' "$gate_metrics")" != true ]]; then
  if [[ "$lane" == 0 ]]; then
    run_attention_breadth 20
    run_block_breadth_followup_model q3 20
    run_context q3
  else
    run_attention_breadth 27
    run_block_breadth_followup_model q3 27
    run_context q2
    run_q3_native_readout
  fi
  exit 0
fi

wait_completed "$postblock_smoke"
for block in $(seq $((13 + lane)) 2 27); do
  run_postblock 0 "$block"
done
wait_all_postblocks 0

if [[ "$lane" == 0 ]]; then
  selection_args=()
  for block in $(seq 13 27); do
    selection_args+=(
      --bundle "$block" "runs/20260718_kuaisearch_mech_d2_q3_postblock_b${block}_fold0_v1"
    )
  done
  "$python_bin" scripts/select_deep_dive_postblock_fold0.py \
    --standardized-dir "$standardized" \
    --qrels-split-dir "$qrels_split" \
    --method-id q3_tallrec_generalqwen \
    "${selection_args[@]}" \
    --output-dir "runs/$selection_run" \
    --analysis-run-id "$selection_run"
else
  wait_completed "runs/$selection_run/selection.json"
fi

for block in $(seq $((13 + lane)) 2 27); do
  run_postblock 1 "$block"
done
wait_all_postblocks 1

if [[ "$lane" == 0 ]]; then
  confirmation_args=()
  for block in $(seq 13 27); do
    confirmation_args+=(
      --bundle "$block" "runs/20260718_kuaisearch_mech_d2_q3_postblock_b${block}_fold1_v1"
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
    echo "invalid Q3 selected-branch eligibility: $selected_eligible" >&2
    exit 3
  fi
  if [[ "$selected_eligible" == true ]]; then
    selected_block="$(jq -r '.selected_block' "runs/$contract_run/contract.json")"
    selected_smoke_run="20260718_kuaisearch_mech_d2_q3_selected_branch_b${selected_block}_smoke_gpu_v1"
    "$python_bin" scripts/score_deep_dive_selected_branches.py \
      --standardized-dir "$standardized" \
      --config "$config" \
      --checkpoint-root "$checkpoint" \
      --run-id "$selected_smoke_run" \
      --device cuda:0 \
      --branch-contract "runs/$contract_run/contract.json" \
      --max-requests 1
    run_selected_shard 0
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
  run_attention_breadth 20
  run_block_breadth_followup_model q3 20
  run_context q3
else
  wait_completed "runs/$confirmation_run/metrics.json"
  wait_completed "runs/$contract_run/contract.json"
  selected_eligible="$(jq -r '.branch_scoring_eligible' "runs/$contract_run/contract.json")"
  if [[ "$selected_eligible" != true && "$selected_eligible" != false ]]; then
    echo "invalid Q3 selected-branch eligibility: $selected_eligible" >&2
    exit 3
  fi
  if [[ "$selected_eligible" == true ]]; then
    wait_selected_smoke
    run_selected_shard 1
  fi
  run_attention_breadth 27
  run_block_breadth_followup_model q3 27
  run_context q2
  run_q3_native_readout
fi
