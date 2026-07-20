#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[0-3]$ ]]; then
  echo "usage: $0 LANE(0|1|2|3)" >&2
  exit 2
fi

lane="$1"
python_bin="/home/gkl/miniconda3/envs/pps-kuaisearch/bin/python"
standardized="data/standardized/kuaisearch/full_confirm_preceding40k_v11"
q2_config="configs/methods/kuaisearch_motivation_v12_q2_recranker_generalqwen.yaml"
q2_checkpoint="artifacts/motivation_v1_2/checkpoints/q2_recranker_generalqwen_seed20260714"
q3_config="configs/methods/kuaisearch_motivation_v12_q3_tallrec_generalqwen.yaml"
q3_checkpoint="artifacts/motivation_v1_2/checkpoints/q3_tallrec_generalqwen_seed20260714"

model="q2"
config="$q2_config"
checkpoint="$q2_checkpoint"
blocks=(13 20)
case "$lane" in
  0) ;;
  1) blocks=(27) ;;
  2)
    model="q3"
    config="$q3_config"
    checkpoint="$q3_checkpoint"
    blocks=(13 20)
    ;;
  3)
    model="q3"
    config="$q3_config"
    checkpoint="$q3_checkpoint"
    blocks=(27)
    ;;
esac

first_block="${blocks[0]}"
smoke_id="20260719_kuaisearch_mech_d4_${model}_mlp_formation_b${first_block}_smoke_gpu_v1"
smoke_metadata="runs/$smoke_id/metadata.json"
if [[ ! -f "$smoke_metadata" ]]; then
  "$python_bin" scripts/observe_deep_dive_mlp_features.py \
    --standardized-dir "$standardized" \
    --config "$config" \
    --checkpoint-root "$checkpoint" \
    --run-id "$smoke_id" \
    --block "$first_block" \
    --device cuda:0 \
    --max-rows 1
fi

[[ "$(jq -r '.status // "missing"' "$smoke_metadata")" == completed ]]
[[ "$(jq -r '.result_eligible' "$smoke_metadata")" == false ]]
awk 'BEGIN {exit !(ARGV[1] <= 0.00001)}' \
  "$(jq -r '.maximum_score_identity_delta' "$smoke_metadata")"
awk 'BEGIN {exit !(ARGV[1] <= 1.0)}' \
  "$(jq -r '.maximum_product_recomposition_low_precision_ratio' "$smoke_metadata")"

for block in "${blocks[@]}"; do
  run_id="20260719_kuaisearch_mech_d4_${model}_mlp_formation_b${block}_v1"
  scripts/run_deep_dive_resume_loop.sh \
    "runs/$run_id/metadata.json" \
    "tmp/${run_id}_resume_loop.log" \
    -- \
    "$python_bin" scripts/observe_deep_dive_mlp_features.py \
      --standardized-dir "$standardized" \
      --config "$config" \
      --checkpoint-root "$checkpoint" \
      --run-id "$run_id" \
      --block "$block" \
      --device cuda:0 \
      --max-wall-seconds 13500
done
