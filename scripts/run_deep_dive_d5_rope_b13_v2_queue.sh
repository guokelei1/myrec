#!/usr/bin/env bash
set -euo pipefail

python_bin="/home/gkl/miniconda3/envs/pps-kuaisearch/bin/python"
standardized="data/standardized/kuaisearch/full_confirm_preceding40k_v11"
replacement_digest="fce73e1241cc67533bd4547a7e061f33c6df199a192e0c6ae2e17433f43db469"

run_replacement() {
  local model="$1"
  local config="configs/methods/kuaisearch_motivation_v12_${model}_recranker_generalqwen.yaml"
  local checkpoint="artifacts/motivation_v1_2/checkpoints/${model}_recranker_generalqwen_seed20260714"
  if [[ "$model" == q3 ]]; then
    config="configs/methods/kuaisearch_motivation_v12_q3_tallrec_generalqwen.yaml"
    checkpoint="artifacts/motivation_v1_2/checkpoints/q3_tallrec_generalqwen_seed20260714"
  fi
  local run_id="20260718_kuaisearch_mech_d5_${model}_rope_b13_v2"
  scripts/run_deep_dive_resume_loop.sh \
    "runs/$run_id/metadata.json" \
    "tmp/${run_id}_resume_loop.log" \
    -- \
    "$python_bin" scripts/score_deep_dive_rope.py \
      --standardized-dir "$standardized" \
      --config "$config" \
      --checkpoint-root "$checkpoint" \
      --run-id "$run_id" \
      --block 13 \
      --device cuda:0 \
      --max-wall-seconds 13500
}

for smoke in \
  runs/20260718_kuaisearch_mech_d5_q3_rope_b13_smoke_gpu_v4_cover32/metadata.json \
  runs/20260718_kuaisearch_mech_d5_q2_rope_b13_smoke_gpu_v3_cover32/metadata.json
do
  [[ "$(jq -r '.status' "$smoke")" == completed ]]
  [[ "$(jq -r '.identity_passed' "$smoke")" == true ]]
  [[ "$(jq -r '.maximum_identity_delta' "$smoke")" == 0.0 ]]
  [[ "$(jq -r '.implementation_identity.digest' "$smoke")" == "$replacement_digest" ]]
done

run_replacement q3
run_replacement q2
