# Problem Discovery Artifacts

Tracked control plane for the active doc/31 pipeline. Raw runs, scores,
checkpoints, and sweeps remain under ignored `runs/`, `models/`, and
`artifacts/`.

Continuous execution follows `doc/32_autonomous_pipeline_controller.md` and
persists live state in `pipeline_state.yaml`, initialized from
`_pipeline_state_template.yaml`.

## Required Order

```text
R0 scope/observability
  -> R0-M Motivation Brief
  -> R0-C0 model-family adequacy
  -> R0-C1 strong-baseline tuning
  -> R0-D Motivation-aligned Failure Atlas
  -> Fxx failure card
  -> Hxx proposal and trial budget
  -> Hxx-Iyy implementations / Hxx-Tzzz dev trials
  -> Hxx-CONFyy lock
```

Do not create an Hxx proposal before its Fxx Failure Card passes review. Copy
the templates in this directory to stable IDs; do not edit a template in place
to represent a live experiment.

R0 does not require a full Failure Card. Use one five-field iteration record,
keep at most three active failure ideas, and probe at most the top two. Create
additional tracked documents only for new evidence, locks, or decisions.
Engineering repairs do not advance `current_r0_round` or `contribution_level`.
Configuration trials and evaluator invocations are separate budget counters.

## Templates

- `_r0_iteration_template.yaml` - five-field discovery record.
- `_motivation_brief_template.md` - one-page problem-value gate before R0-C/D.
- `_pipeline_state_template.yaml` - resumable autonomous controller state.
- `_failure_card_template.md` - full architecture-entry evidence for a survivor.
- `_trial_budget_template.yaml` - development feedback and change ledger.
- `_confirmation_lock_template.yaml` - no-feedback confirmation boundary.

Every dev evaluator invocation must also appear in
`reports/dev_eval_log.jsonl`. These manifests do not replace the shared
evaluator log or run metadata.
