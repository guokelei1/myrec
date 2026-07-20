#!/usr/bin/env bash
set -euo pipefail

# N13 separates Q formation, K formation and V transport after the N12 MLP
# stage wave. It is a diagnostic-only queue and never starts until the prior
# evaluator is terminal, so a released GPU cannot be double-booked.
python_bin="/home/gkl/miniconda3/envs/pps-kuaisearch/bin/python"
standardized="data/standardized/kuaisearch/full_confirm_preceding40k_v11"
qrels_split="artifacts/motivation_transformer_deep_dive/frozen_controls/dev_qrels_folds_v1"
q2_config="configs/methods/kuaisearch_motivation_v12_q2_recranker_generalqwen.yaml"
q3_config="configs/methods/kuaisearch_motivation_v12_q3_tallrec_generalqwen.yaml"
q2_checkpoint="artifacts/motivation_v1_2/checkpoints/q2_recranker_generalqwen_seed20260714"
q3_checkpoint="artifacts/motivation_v1_2/checkpoints/q3_tallrec_generalqwen_seed20260714"
n12_eval="runs/20260720_kuaisearch_mech_n12_mlp_stage_eval_v1/metrics.json"

wait_completed() {
  local path="$1"
  while true; do
    local status="missing"
    if [[ -f "$path" ]]; then status="$(jq -r '.status // "missing"' "$path")"; fi
    case "$status" in
      completed) return 0 ;;
      missing|initializing|running|wall_time_exhausted) sleep 30 ;;
      *) echo "N13 upstream terminal status=$status path=$path" >&2; return 3 ;;
    esac
  done
}

run_one() {
  local config="$1" checkpoint="$2" block="$3" gpu="$4" run_id="$5"
  env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src \
    scripts/run_deep_dive_resume_loop.sh \
      "runs/$run_id/metadata.json" "tmp/${run_id}_resume_loop.log" -- \
      "$python_bin" scripts/score_deep_dive_qkv_projection.py \
        --standardized-dir "$standardized" --config "$config" \
        --checkpoint-root "$checkpoint" --run-id "$run_id" \
        --block "$block" --device cuda:0 --max-wall-seconds 13500
}

echo "N13 QKV-projection queue waiting for N12 evaluator: $n12_eval"
wait_completed "$n12_eval"

run_one "$q2_config" "$q2_checkpoint" 13 0 20260720_kuaisearch_mech_n13_q2_qkv_b13_v1 & p0=$!
run_one "$q2_config" "$q2_checkpoint" 20 1 20260720_kuaisearch_mech_n13_q2_qkv_b20_v1 & p1=$!
run_one "$q3_config" "$q3_checkpoint" 13 2 20260720_kuaisearch_mech_n13_q3_qkv_b13_v1 & p2=$!
run_one "$q3_config" "$q3_checkpoint" 20 3 20260720_kuaisearch_mech_n13_q3_qkv_b20_v1 & p3=$!
wait "$p0" "$p1" "$p2" "$p3"

run_one "$q2_config" "$q2_checkpoint" 27 0 20260720_kuaisearch_mech_n13_q2_qkv_b27_v1 & p4=$!
run_one "$q3_config" "$q3_checkpoint" 27 1 20260720_kuaisearch_mech_n13_q3_qkv_b27_v1 & p5=$!
wait "$p4" "$p5"

exec env PYTHONPATH=src "$python_bin" scripts/evaluate_deep_dive_qkv_projection.py \
  --standardized-dir "$standardized" \
  --q2-b13 runs/20260720_kuaisearch_mech_n13_q2_qkv_b13_v1 \
  --q2-b20 runs/20260720_kuaisearch_mech_n13_q2_qkv_b20_v1 \
  --q2-b27 runs/20260720_kuaisearch_mech_n13_q2_qkv_b27_v1 \
  --q3-b13 runs/20260720_kuaisearch_mech_n13_q3_qkv_b13_v1 \
  --q3-b20 runs/20260720_kuaisearch_mech_n13_q3_qkv_b20_v1 \
  --q3-b27 runs/20260720_kuaisearch_mech_n13_q3_qkv_b27_v1 \
  --output-dir runs/20260720_kuaisearch_mech_n13_qkv_projection_eval_v1 \
  --analysis-run-id 20260720_kuaisearch_mech_n13_qkv_projection_eval_v1
