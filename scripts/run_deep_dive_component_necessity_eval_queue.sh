#!/usr/bin/env bash
set -euo pipefail

python_bin="/home/gkl/miniconda3/envs/pps-kuaisearch/bin/python"
standardized="data/standardized/kuaisearch/full_confirm_preceding40k_v11"
qrels_split="artifacts/motivation_transformer_deep_dive/frozen_controls/dev_qrels_folds_v1"
q2_contract="runs/20260718_kuaisearch_mech_d2_q2_selected_branch_contract_v1/contract.json"
q3_contract="runs/20260718_kuaisearch_mech_d2_q3_selected_branch_contract_v1/contract.json"
q2_bundle="runs/20260719_kuaisearch_mech_component_necessity_q2_v1"
q3_bundle="runs/20260719_kuaisearch_mech_component_necessity_q3_v1"
analysis_run="20260719_kuaisearch_mech_component_necessity_eval_v1"

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

wait_completed "$q2_contract"
wait_completed "$q3_contract"
args=()
for short in q2 q3; do
  contract_var="${short}_contract"
  bundle_var="${short}_bundle"
  contract="${!contract_var}"
  bundle="${!bundle_var}"
  if [[ "$(jq -r '.fold1_negative_transition_reproduced // false' "$contract")" == true ]]; then
    wait_completed "$bundle/metadata.json"
    args+=("--${short}-bundle" "$bundle")
  else
    args+=("--${short}-gate-contract" "$contract")
  fi
done

exec "$python_bin" scripts/evaluate_deep_dive_component_necessity.py \
  --standardized-dir "$standardized" \
  --qrels-split-dir "$qrels_split" \
  "${args[@]}" \
  --output-dir "runs/$analysis_run" \
  --analysis-run-id "$analysis_run"
