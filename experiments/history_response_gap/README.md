# History-response direction-gap workspace

This is the clean active experiment area for
`doc/34_history_response_direction_gap_validation_plan.md`.

Expected tracked contents:

- E0 source/admission protocol and dataset cards;
- frozen field mappings, collision eligibility rules, and power/MDE rules;
- model-family and control boundary cards;
- concise run manifests and decision records.

Expected local-only contents:

- checkpoints in `models/`;
- raw outputs in `runs/`;
- generated materializations in `artifacts/`;
- standardized records in `data/standardized/`.

Current status: E0 and metric-instrumentation development ready. No E0
admission decision, model training, dev/confirmation label evaluation, or
proposed architecture is authorized yet.

## Active files

- `experiment_manifest.yaml`: machine-readable phase and authorization state;
- `e0_admission_protocol.md`: human-review draft for the first gate;
- `_dataset_admission_card.yaml`: one card per candidate dataset;
- `_counterfactual_bundle_card.yaml`: true/null/wrong execution lock template;
- `score_bundle_contract.md`: score and metadata boundary enforced by the
  shared evaluator;
- `archive_reuse_policy.md`: what may be selectively migrated from legacy;
- `workspace_status.md`: current local prerequisites and blockers.
