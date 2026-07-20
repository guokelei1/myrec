#!/usr/bin/env bash
set -euo pipefail

# N11 is intentionally downstream of N8/N9/N10.  It isolates the pre-softmax
# scaled-QK-logit operator, rather than adding another history-edge mask.  The
# first wave uses all four physical GPUs; the second wave reuses only two after
# the first four bundles have closed.
python_bin="/home/gkl/miniconda3/envs/pps-kuaisearch/bin/python"
standardized="data/standardized/kuaisearch/full_confirm_preceding40k_v11"
q2_config="configs/methods/kuaisearch_motivation_v12_q2_recranker_generalqwen.yaml"
q3_config="configs/methods/kuaisearch_motivation_v12_q3_tallrec_generalqwen.yaml"
q2_checkpoint="artifacts/motivation_v1_2/checkpoints/q2_recranker_generalqwen_seed20260714"
q3_checkpoint="artifacts/motivation_v1_2/checkpoints/q3_tallrec_generalqwen_seed20260714"
n8_eval="runs/20260720_kuaisearch_mech_n8_composition_eval_v1/metrics.json"
n9_eval="runs/20260720_kuaisearch_mech_n9_history_path_eval_v1/metrics.json"
n10_rank_eval="runs/20260720_kuaisearch_mech_n10_q3_lora_rank_eval_v1/metrics.json"
n10_candidate_q0_eval="runs/20260720_kuaisearch_mech_n10_candidate_gap_q0_eval_v1/metrics.json"
n10_candidate_q1_eval="runs/20260720_kuaisearch_mech_n10_candidate_gap_q1_eval_v1/metrics.json"
n10_candidate_q2_eval="runs/20260720_kuaisearch_mech_n10_candidate_gap_q2_eval_v1/metrics.json"
n10_candidate_q3_eval="runs/20260720_kuaisearch_mech_n10_candidate_gap_q3_eval_v1/metrics.json"

wait_completed() {
  local path="$1"
  while true; do
    local status="missing"
    if [[ -f "$path" ]]; then status="$(jq -r '.status // "missing"' "$path")"; fi
    case "$status" in
      completed) return 0 ;;
      missing|initializing|running|wall_time_exhausted) sleep 30 ;;
      *) echo "N11 upstream terminal status=$status path=$path" >&2; return 3 ;;
    esac
  done
}

run_one() {
  local method="$1" config="$2" checkpoint="$3" block="$4" gpu="$5" run_id="$6"
  env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src \
    scripts/run_deep_dive_resume_loop.sh \
      "runs/$run_id/metadata.json" "tmp/${run_id}_resume_loop.log" -- \
      "$python_bin" scripts/score_deep_dive_attention_logits.py \
        --standardized-dir "$standardized" --config "$config" \
        --checkpoint-root "$checkpoint" --run-id "$run_id" \
        --block "$block" --device cuda:0 --max-wall-seconds 13500
}

echo "N11 attention-logit queue waiting for N8/N9/N10 evaluators"
wait_completed "$n8_eval"
wait_completed "$n9_eval"
wait_completed "$n10_rank_eval"
wait_completed "$n10_candidate_q0_eval"
wait_completed "$n10_candidate_q1_eval"
wait_completed "$n10_candidate_q2_eval"
wait_completed "$n10_candidate_q3_eval"

# First four fixed model/block jobs occupy the four physical cards.
run_one q2 "$q2_config" "$q2_checkpoint" 13 0 20260720_kuaisearch_mech_n11_q2_attention_logits_b13_v1 & p0=$!
run_one q2 "$q2_config" "$q2_checkpoint" 20 1 20260720_kuaisearch_mech_n11_q2_attention_logits_b20_v1 & p1=$!
run_one q3 "$q3_config" "$q3_checkpoint" 13 2 20260720_kuaisearch_mech_n11_q3_attention_logits_b13_v1 & p2=$!
run_one q3 "$q3_config" "$q3_checkpoint" 20 3 20260720_kuaisearch_mech_n11_q3_attention_logits_b20_v1 & p3=$!
wait "$p0" "$p1" "$p2" "$p3"

# The remaining block is independent and uses disjoint run IDs.
run_one q2 "$q2_config" "$q2_checkpoint" 27 0 20260720_kuaisearch_mech_n11_q2_attention_logits_b27_v1 & p4=$!
run_one q3 "$q3_config" "$q3_checkpoint" 27 1 20260720_kuaisearch_mech_n11_q3_attention_logits_b27_v1 & p5=$!
wait "$p4" "$p5"

exec env PYTHONPATH=src "$python_bin" scripts/evaluate_deep_dive_attention_logits.py \
  --standardized-dir "$standardized" \
  --q2-b13 runs/20260720_kuaisearch_mech_n11_q2_attention_logits_b13_v1 \
  --q2-b20 runs/20260720_kuaisearch_mech_n11_q2_attention_logits_b20_v1 \
  --q2-b27 runs/20260720_kuaisearch_mech_n11_q2_attention_logits_b27_v1 \
  --q3-b13 runs/20260720_kuaisearch_mech_n11_q3_attention_logits_b13_v1 \
  --q3-b20 runs/20260720_kuaisearch_mech_n11_q3_attention_logits_b20_v1 \
  --q3-b27 runs/20260720_kuaisearch_mech_n11_q3_attention_logits_b27_v1 \
  --output-dir runs/20260720_kuaisearch_mech_n11_attention_logit_eval_v1 \
  --analysis-run-id 20260720_kuaisearch_mech_n11_attention_logit_eval_v1
