# Problem Discovery Artifacts

Tracked control plane for the active doc/31 pipeline. Raw runs, scores,
checkpoints, and sweeps remain under ignored `runs/`, `models/`, and
`artifacts/`.

## Required Order

```text
R0 scope/observability/strong baseline
  -> Fxx failure card
  -> Hxx proposal and trial budget
  -> Hxx-Iyy implementations / Hxx-Tzzz dev trials
  -> Hxx-CONFyy lock
```

Do not create an Hxx proposal before its Fxx Failure Card passes review. Copy
the templates in this directory to stable IDs; do not edit a template in place
to represent a live experiment.

## Templates

- `_failure_card_template.md` - architecture-entry evidence.
- `_trial_budget_template.yaml` - development feedback and change ledger.
- `_confirmation_lock_template.yaml` - no-feedback confirmation boundary.

Every dev evaluator invocation must also appear in
`reports/dev_eval_log.jsonl`. These manifests do not replace the shared
evaluator log or run metadata.
