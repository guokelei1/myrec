#!/usr/bin/env bash
set -euo pipefail

# N8 is deliberately chained after the existing D0--D7 and component-design
# wave.  It never competes with an active registered worker for a GPU.
python_bin="/home/gkl/miniconda3/envs/pps-kuaisearch/bin/python"
standardized="data/standardized/kuaisearch/full_confirm_preceding40k_v11"
qrels_split="artifacts/motivation_transformer_deep_dive/frozen_controls/dev_qrels_folds_v1"
q2_config="configs/methods/kuaisearch_motivation_v12_q2_recranker_generalqwen.yaml"
q3_config="configs/methods/kuaisearch_motivation_v12_q3_tallrec_generalqwen.yaml"
q2_checkpoint="artifacts/motivation_v1_2/checkpoints/q2_recranker_generalqwen_seed20260714"
q3_checkpoint="artifacts/motivation_v1_2/checkpoints/q3_tallrec_generalqwen_seed20260714"
q2_contract="runs/20260718_kuaisearch_mech_d2_q2_selected_branch_contract_v1/contract.json"
q3_contract="runs/20260718_kuaisearch_mech_d2_q3_selected_branch_contract_v1/contract.json"
q2_parent="runs/20260718_kuaisearch_mech_d2_q2_selected_branch_fold1_v1"
q3_parent="runs/20260718_kuaisearch_mech_d2_q3_selected_branch_fold1_v1"
terminal_sentinel="runs/20260719_kuaisearch_mech_component_design_synthesis_v1/metrics.json"

echo "N8 queue waiting for terminal sentinel: $terminal_sentinel"

wait_completed() {
  local path="$1"
  while true; do
    local status="missing"
    if [[ -f "$path" ]]; then
      status="$(jq -r '.status // "missing"' "$path")"
    fi
    case "$status" in
      completed) return 0 ;;
      missing|initializing|running|selection_finalized|wall_time_exhausted) sleep 30 ;;
      *) echo "N8 upstream terminal status=$status path=$path" >&2; return 3 ;;
    esac
  done
}

wait_completed "$terminal_sentinel"
wait_completed "runs/20260718_kuaisearch_mech_d2_q2_selected_branch_eval_v1/metrics.json"
wait_completed "runs/20260718_kuaisearch_mech_d2_q3_selected_branch_eval_v1/metrics.json"
wait_completed "$q2_contract"
wait_completed "$q3_contract"
wait_completed "$q2_parent/metadata.json"
wait_completed "$q3_parent/metadata.json"

q2_run="20260720_kuaisearch_mech_n8_q2_joint_composition_v1"
q3_run="20260720_kuaisearch_mech_n8_q3_joint_composition_v1"

run_one() {
  local method="$1"
  local config="$2"
  local checkpoint="$3"
  local contract="$4"
  local parent="$5"
  local run_id="$6"
  local physical_gpu="$7"
  env CUDA_VISIBLE_DEVICES="$physical_gpu" \
    scripts/run_deep_dive_resume_loop.sh \
      "runs/$run_id/metadata.json" \
      "tmp/${run_id}_resume_loop.log" \
      -- \
      "$python_bin" scripts/score_deep_dive_component_composition.py \
        --standardized-dir "$standardized" \
        --config "$config" \
        --checkpoint-root "$checkpoint" \
        --run-id "$run_id" \
        --device cuda:0 \
        --branch-contract "$contract" \
        --parent-selected-branch "$parent" \
        --max-wall-seconds 13500
}

run_one q2 "$q2_config" "$q2_checkpoint" "$q2_contract" "$q2_parent" "$q2_run" 0 &
q2_pid=$!
run_one q3 "$q3_config" "$q3_checkpoint" "$q3_contract" "$q3_parent" "$q3_run" 1 &
q3_pid=$!
wait "$q2_pid"
wait "$q3_pid"

exec env PYTHONPATH=src "$python_bin" scripts/evaluate_deep_dive_component_composition.py \
  --standardized-dir "$standardized" \
  --qrels-split-dir "$qrels_split" \
  --q2-bundle "runs/$q2_run" \
  --q3-bundle "runs/$q3_run" \
  --output-dir runs/20260720_kuaisearch_mech_n8_composition_eval_v1 \
  --analysis-run-id 20260720_kuaisearch_mech_n8_composition_eval_v1
