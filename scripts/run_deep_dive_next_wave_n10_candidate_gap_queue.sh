#!/usr/bin/env bash
set -euo pipefail

# N10 candidate-gap queue.  It shares the N8/N9 dependency gate with the rank
# queue, then uses the remaining three physical GPUs: GPU1 runs Q0 then Q1,
# GPU2 runs Q2, and GPU3 runs Q3.  The scorer and evaluator never open qrels.
python_bin="/home/gkl/miniconda3/envs/pps-kuaisearch/bin/python"
standardized="data/standardized/kuaisearch/full_confirm_preceding40k_v11"
n8_eval="runs/20260720_kuaisearch_mech_n8_composition_eval_v1/metrics.json"
n9_eval="runs/20260720_kuaisearch_mech_n9_history_path_eval_v1/metrics.json"

q0_config="configs/methods/kuaisearch_motivation_v12_q0_qwen3_reranker_06b.yaml"
q1_config="configs/methods/kuaisearch_motivation_v12_q1_instructrec_generalqwen.yaml"
q2_config="configs/methods/kuaisearch_motivation_v12_q2_recranker_generalqwen.yaml"
q3_config="configs/methods/kuaisearch_motivation_v12_q3_tallrec_generalqwen.yaml"
q0_checkpoint="artifacts/motivation_v1_2/checkpoints/q0_qwen3_reranker_06b_seed20260714"
q1_checkpoint="artifacts/motivation_v1_2/checkpoints/q1_instructrec_generalqwen_seed20260714"
q2_checkpoint="artifacts/motivation_v1_2/checkpoints/q2_recranker_generalqwen_seed20260714"
q3_checkpoint="artifacts/motivation_v1_2/checkpoints/q3_tallrec_generalqwen_seed20260714"

wait_completed() {
  local path="$1"
  while true; do
    local status="missing"
    if [[ -f "$path" ]]; then status="$(jq -r '.status // "missing"' "$path")"; fi
    case "$status" in
      completed) return 0 ;;
      missing|initializing|running|wall_time_exhausted) sleep 30 ;;
      *) echo "N10 candidate-gap upstream terminal status=$status path=$path" >&2; return 3 ;;
    esac
  done
}

run_one() {
  local method="$1" config="$2" checkpoint="$3" gpu="$4" run_id="$5"
  env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src \
    scripts/run_deep_dive_resume_loop.sh \
      "runs/$run_id/metadata.json" "tmp/${run_id}_resume_loop.log" -- \
      "$python_bin" scripts/score_deep_dive_candidate_gap.py \
        --standardized-dir "$standardized" --config "$config" \
        --checkpoint-root "$checkpoint" --run-id "$run_id" \
        --device cuda:0 --max-wall-seconds 13500
}

echo "N10 candidate-gap queue waiting for N8/N9 evaluators: $n8_eval ; $n9_eval"
wait_completed "$n8_eval"
wait_completed "$n9_eval"

run_one q0 "$q0_config" "$q0_checkpoint" 1 20260720_kuaisearch_mech_n10_candidate_gap_q0_v1 &
q0_pid=$!
run_one q2 "$q2_config" "$q2_checkpoint" 2 20260720_kuaisearch_mech_n10_candidate_gap_q2_v1 &
q2_pid=$!
run_one q3 "$q3_config" "$q3_checkpoint" 3 20260720_kuaisearch_mech_n10_candidate_gap_q3_v1 &
q3_pid=$!
wait "$q0_pid" "$q2_pid" "$q3_pid"
# GPU1 is deliberately reused only after Q0 has closed and passed its
# resume-loop boundary; Q1 has the more expensive listwise continuation path.
run_one q1 "$q1_config" "$q1_checkpoint" 1 20260720_kuaisearch_mech_n10_candidate_gap_q1_v1

for method in q0 q1 q2 q3; do
  env PYTHONPATH=src "$python_bin" scripts/evaluate_deep_dive_candidate_gap.py \
    --standardized-dir "$standardized" \
    --bundle "runs/20260720_kuaisearch_mech_n10_candidate_gap_${method}_v1" \
    --output-dir "runs/20260720_kuaisearch_mech_n10_candidate_gap_${method}_eval_v1" \
    --analysis-run-id "20260720_kuaisearch_mech_n10_candidate_gap_${method}_eval_v1"
done
