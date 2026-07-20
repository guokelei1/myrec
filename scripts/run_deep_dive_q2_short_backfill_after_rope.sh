#!/usr/bin/env bash
set -euo pipefail

# Qrels-blind utilization backfill for physical GPU 2.  Once the fixed Q3->Q2
# block-13 RoPE queue finishes, use only the remaining Q2 fold-1 waiting window
# for already-registered short b20/b27 breadth jobs.  Stop before the Q2
# selected-branch contract can claim this card.  Long edge/RoPE jobs are
# excluded deliberately.

python_bin="/home/gkl/miniconda3/envs/pps-kuaisearch/bin/python"
standardized="data/standardized/kuaisearch/full_confirm_preceding40k_v11"
config="configs/methods/kuaisearch_motivation_v12_q2_recranker_generalqwen.yaml"
checkpoint="artifacts/motivation_v1_2/checkpoints/q2_recranker_generalqwen_seed20260714"
rope_metadata="runs/20260718_kuaisearch_mech_d5_q2_rope_b13_v2/metadata.json"
contract="runs/20260718_kuaisearch_mech_d2_q2_selected_branch_contract_v1/contract.json"

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

remaining_q2_fold1_bundles() {
  local completed=0 block metadata status
  for block in $(seq 13 27); do
    metadata="runs/20260718_kuaisearch_mech_d2_q2_postblock_b${block}_fold1_v1/metadata.json"
    status="missing"
    if [[ -f "$metadata" ]]; then
      status="$(jq -r '.status // "missing"' "$metadata")"
    fi
    case "$status" in
      completed) completed=$((completed + 1)) ;;
      missing|initializing|running|wall_time_exhausted) ;;
      *) echo "Q2 fold1 terminal status=$status path=$metadata" >&2; return 3 ;;
    esac
  done
  echo $((15 - completed))
}

safe_before_contract() {
  local minimum_remaining="$1"
  if [[ -f "$contract" ]]; then
    local contract_status
    contract_status="$(jq -r '.status // "missing"' "$contract")"
    if [[ "$contract_status" == completed ]]; then
      return 1
    fi
  fi
  local remaining
  remaining="$(remaining_q2_fold1_bundles)"
  [[ "$remaining" -ge "$minimum_remaining" ]]
}

run_attention_heads() {
  local block="$1"
  local run_id="20260718_kuaisearch_mech_d3_q2_attention_heads_b${block}_v1"
  scripts/run_deep_dive_resume_loop.sh \
    "runs/$run_id/metadata.json" \
    "tmp/${run_id}_resume_loop.log" \
    -- \
    "$python_bin" scripts/observe_deep_dive_attention_heads.py \
      --standardized-dir "$standardized" \
      --config "$config" \
      --checkpoint-root "$checkpoint" \
      --run-id "$run_id" \
      --block "$block" \
      --device cuda:0 \
      --max-wall-seconds 13500
}

run_attention_groups() {
  local block="$1"
  local run_id="20260718_kuaisearch_mech_d3_q2_attention_groups_b${block}_v1"
  scripts/run_deep_dive_resume_loop.sh \
    "runs/$run_id/metadata.json" \
    "tmp/${run_id}_resume_loop.log" \
    -- \
    "$python_bin" scripts/score_deep_dive_attention_groups.py \
      --standardized-dir "$standardized" \
      --config "$config" \
      --checkpoint-root "$checkpoint" \
      --run-id "$run_id" \
      --block "$block" \
      --device cuda:0 \
      --max-wall-seconds 13500
}

run_mlp_groups() {
  local block="$1"
  local run_id="20260718_kuaisearch_mech_d4_q2_mlp_groups_b${block}_v1"
  scripts/run_deep_dive_resume_loop.sh \
    "runs/$run_id/metadata.json" \
    "tmp/${run_id}_resume_loop.log" \
    -- \
    "$python_bin" scripts/score_deep_dive_mlp_groups.py \
      --standardized-dir "$standardized" \
      --config "$config" \
      --checkpoint-root "$checkpoint" \
      --run-id "$run_id" \
      --block "$block" \
      --device cuda:0 \
      --max-wall-seconds 13500
}

for smoke in \
  runs/20260718_kuaisearch_mech_d3_q2_attention_heads_b13_smoke_gpu_v2/metadata.json \
  runs/20260718_kuaisearch_mech_d3_q2_attention_groups_b13_smoke_gpu_v2/metadata.json \
  runs/20260718_kuaisearch_mech_d4_q2_mlp_groups_b13_smoke_gpu_v2/metadata.json
do
  [[ "$(jq -r '.status // "missing"' "$smoke")" == completed ]]
done

wait_completed "$rope_metadata"

for block in 20 27; do
  if safe_before_contract 2; then
    run_attention_heads "$block"
  fi
  if safe_before_contract 3; then
    run_attention_groups "$block"
  fi
  if safe_before_contract 2; then
    run_mlp_groups "$block"
  fi
done
