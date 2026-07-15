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

Current status: the exploratory controlled-history-composition motivation is
established across KuaiSearch, Amazon-C4, and JDsearch. The representative Lite
matrix is now executable end to end: Qwen/BGE evidence is available, HSTU and
matched SASRec have QC/FULL bundles, and the independent LLM-SRec mechanism has
a frozen teacher plus true/null/wrong results. The sequence-oriented outcomes
are supportive but not binding because their adequacy gates are still open.
Authorized next work is ordinary baseline adequacy, Amazon-C4 replication,
standard repair, recoverability, and later confirmation. Test and proposed
architecture remain locked.

## Active files

- `../../doc/35_controlled_history_composition_motivation.md`: current
  cross-dataset motivation decision and architecture boundary;
- `experiment_manifest.yaml`: machine-readable phase and authorization state;
- `exploration_protocol.md`: flexible exploration/confirmation boundary and
  observation discipline;
- `pipeline_state.yaml`: current question, evidence delta, correction, and next
  reversible probe;
- `representative_architecture_protocol.yaml`: frozen exploratory comparison of
  Qwen, HSTU, LLM-SRec, and the existing BGE encoder anchor;
- `e0_admission_protocol.md`: human-review draft for the first gate;
- `_dataset_admission_card.yaml`: one card per candidate dataset;
- `_counterfactual_bundle_card.yaml`: true/null/wrong execution lock template;
- `score_bundle_contract.md`: score and metadata boundary enforced by the
  shared evaluator;
- `archive_reuse_policy.md`: what may be selectively migrated from legacy;
- `workspace_status.md`: current local prerequisites and blockers.
