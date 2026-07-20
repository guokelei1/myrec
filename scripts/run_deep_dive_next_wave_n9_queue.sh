#!/usr/bin/env bash
set -euo pipefail

# N9 occupies GPU2/GPU3 in parallel with N8 on GPU0/GPU1.  The queue is a
# CPU-only watcher until the current registered wave reaches its terminal
# sentinel, so it cannot preempt or duplicate an active worker.
python_bin="/home/gkl/miniconda3/envs/pps-kuaisearch/bin/python"
standardized="data/standardized/kuaisearch/full_confirm_preceding40k_v11"
qrels_split="artifacts/motivation_transformer_deep_dive/frozen_controls/dev_qrels_folds_v1"
q2_config="configs/methods/kuaisearch_motivation_v12_q2_recranker_generalqwen.yaml"
q3_config="configs/methods/kuaisearch_motivation_v12_q3_tallrec_generalqwen.yaml"
q2_checkpoint="artifacts/motivation_v1_2/checkpoints/q2_recranker_generalqwen_seed20260714"
q3_checkpoint="artifacts/motivation_v1_2/checkpoints/q3_tallrec_generalqwen_seed20260714"
terminal_sentinel="runs/20260719_kuaisearch_mech_component_design_synthesis_v1/metrics.json"

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
      *) echo "N9 upstream terminal status=$status path=$path" >&2; return 3 ;;
    esac
  done
}

echo "N9 queue waiting for terminal sentinel: $terminal_sentinel"
wait_completed "$terminal_sentinel"
wait_completed "runs/20260718_kuaisearch_mech_d2_q2_selected_branch_eval_v1/metrics.json"
wait_completed "runs/20260718_kuaisearch_mech_d2_q3_selected_branch_eval_v1/metrics.json"
wait_completed "runs/20260718_kuaisearch_mech_d2_q2_selected_branch_contract_v1/contract.json"
wait_completed "runs/20260718_kuaisearch_mech_d2_q3_selected_branch_contract_v1/contract.json"

run_model() {
  local method="$1"
  local config="$2"
  local checkpoint="$3"
  local gpu="$4"
  local block
  for block in 13 20 27; do
    local run_id="20260720_kuaisearch_mech_n9_${method}_b${block}_v1"
    env CUDA_VISIBLE_DEVICES="$gpu" \
      scripts/run_deep_dive_resume_loop.sh \
        "runs/$run_id/metadata.json" \
        "tmp/${run_id}_resume_loop.log" \
        -- \
        "$python_bin" scripts/score_deep_dive_history_path.py \
          --standardized-dir "$standardized" \
          --config "$config" \
          --checkpoint-root "$checkpoint" \
          --run-id "$run_id" \
          --block "$block" \
          --device cuda:0 \
          --max-wall-seconds 13500
  done
}

run_model q2_recranker_generalqwen "$q2_config" "$q2_checkpoint" 2 &
q2_pid=$!
run_model q3_tallrec_generalqwen "$q3_config" "$q3_checkpoint" 3 &
q3_pid=$!
wait "$q2_pid"
wait "$q3_pid"

exec env PYTHONPATH=src "$python_bin" scripts/evaluate_deep_dive_history_path.py \
  --standardized-dir "$standardized" \
  --qrels-split-dir "$qrels_split" \
  --q2-b13 runs/20260720_kuaisearch_mech_n9_q2_recranker_generalqwen_b13_v1 \
  --q2-b20 runs/20260720_kuaisearch_mech_n9_q2_recranker_generalqwen_b20_v1 \
  --q2-b27 runs/20260720_kuaisearch_mech_n9_q2_recranker_generalqwen_b27_v1 \
  --q3-b13 runs/20260720_kuaisearch_mech_n9_q3_tallrec_generalqwen_b13_v1 \
  --q3-b20 runs/20260720_kuaisearch_mech_n9_q3_tallrec_generalqwen_b20_v1 \
  --q3-b27 runs/20260720_kuaisearch_mech_n9_q3_tallrec_generalqwen_b27_v1 \
  --output-dir runs/20260720_kuaisearch_mech_n9_history_path_eval_v1 \
  --analysis-run-id 20260720_kuaisearch_mech_n9_history_path_eval_v1

