# Motivation V1 experiment boundary

This directory now contains only the current V1 state, the frozen Qwen
confirmation protocol, and reusable score/data cards. Superseded exploration,
repair, cross-dataset, and representative-architecture plans were removed.

## Active files

- `../../doc/40_transformer_recurrence_transfer_motivation_v1_zh.md`: current
  human-readable conclusion and claim boundary;
- `../../reports/pps_three_transformer_history_surface_audit.json`: canonical
  three-model machine-readable evidence;
- `kuaisearch_confirmation_protocol.md`: frozen protocol used for the binding
  Qwen confirmation;
- `experiment_manifest.yaml`: current authorization and evidence manifest;
- `pipeline_state.yaml`: compact current state and next robustness boundary;
- `score_bundle_contract.md`: shared score/metadata contract;
- `_counterfactual_bundle_card.yaml`: reusable true/null/wrong lock template;
- `_dataset_admission_card.yaml`: reusable dataset-admission template.

Raw scores, checkpoints, logs, and standardized records remain under `runs/`,
`models/`, `artifacts/`, and `data/`. V1 authorizes neither test access nor a
new proposed architecture.
