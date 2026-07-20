#!/usr/bin/env bash
set -euo pipefail

# Qrels-blind registered breadth backfill for physical GPU 3.  This lane runs
# only fixed Q2 blocks 20/27 that are already required by D3/D4.  Canonical
# resume locks prevent overlap with the slower post-RoPE Q2 backfill lane.

python_bin="/home/gkl/miniconda3/envs/pps-kuaisearch/bin/python"
standardized="data/standardized/kuaisearch/full_confirm_preceding40k_v11"
config="configs/methods/kuaisearch_motivation_v12_q2_recranker_generalqwen.yaml"
checkpoint="artifacts/motivation_v1_2/checkpoints/q2_recranker_generalqwen_seed20260714"
q3_fold0_selection="runs/20260718_kuaisearch_mech_d2_q3_postblock_fold0_selection_v1/selection.json"

# This script is an opportunistic pre-selection backfill only.  Once Q3 has
# frozen its fold-0 transition, physical GPU 3 belongs to the registered Q3
# fold-1/selected-branch lane.  Any partial Q2 bundle remains canonically
# resumable by the later main breadth lanes.
if [[ -f "$q3_fold0_selection" ]]; then
  echo "Q3 fold-0 selection exists; yielding GPU3 Q2 breadth to the Q3 fold-1 lane"
  exit 0
fi

run_resumable() {
  local run_id="$1"
  shift
  local metadata="runs/$run_id/metadata.json"
  if [[ -f "$metadata" ]]; then
    local status
    status="$(jq -r '.status // "missing"' "$metadata")"
    case "$status" in
      completed) return 0 ;;
      initializing|running|wall_time_exhausted) ;;
      *) echo "registered breadth run has terminal status=$status: $run_id" >&2; return 3 ;;
    esac
  fi
  scripts/run_deep_dive_resume_loop.sh \
    "$metadata" \
    "tmp/${run_id}_resume_loop.log" \
    -- \
    "$@"
}

for smoke in \
  runs/20260718_kuaisearch_mech_d3_q2_attention_heads_b13_smoke_gpu_v2/metadata.json \
  runs/20260718_kuaisearch_mech_d3_q2_attention_groups_b13_smoke_gpu_v2/metadata.json \
  runs/20260718_kuaisearch_mech_d4_q2_mlp_groups_b13_smoke_gpu_v2/metadata.json
do
  [[ "$(jq -r '.status // "missing"' "$smoke")" == completed ]]
done

for block in 20 27; do
  run_id="20260718_kuaisearch_mech_d3_q2_attention_heads_b${block}_v1"
  run_resumable "$run_id" \
    "$python_bin" scripts/observe_deep_dive_attention_heads.py \
      --standardized-dir "$standardized" \
      --config "$config" \
      --checkpoint-root "$checkpoint" \
      --run-id "$run_id" \
      --block "$block" \
      --device cuda:0 \
      --max-wall-seconds 13500
done

for block in 20 27; do
  run_id="20260718_kuaisearch_mech_d3_q2_attention_groups_b${block}_v1"
  run_resumable "$run_id" \
    "$python_bin" scripts/score_deep_dive_attention_groups.py \
      --standardized-dir "$standardized" \
      --config "$config" \
      --checkpoint-root "$checkpoint" \
      --run-id "$run_id" \
      --block "$block" \
      --device cuda:0 \
      --max-wall-seconds 13500
done

for block in 20 27; do
  run_id="20260718_kuaisearch_mech_d4_q2_mlp_groups_b${block}_v1"
  run_resumable "$run_id" \
    "$python_bin" scripts/score_deep_dive_mlp_groups.py \
      --standardized-dir "$standardized" \
      --config "$config" \
      --checkpoint-root "$checkpoint" \
      --run-id "$run_id" \
      --block "$block" \
      --device cuda:0 \
      --max-wall-seconds 13500
done
