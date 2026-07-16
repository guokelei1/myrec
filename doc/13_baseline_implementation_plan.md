# Baseline implementation notes

Status: supporting fairness notes. The active methods, witness, model sizes,
seed staging, and budgets are defined only in
`experiments/motivation_v1_2/plan.md`.

## Durable fairness boundary

- All methods read the standardized interface and fixed candidate slate.
- Each method declares its visible fields and boundary card.
- Training labels may supervise a loss but may not become model input fields.
- All paper metrics come from the shared evaluator.
- A method must have non-degenerate query-candidate ranking behavior before its
  history-surface result is interpreted.
- Ordinary method-specific choices such as learning rate, epoch, truncation,
  sampling, and loss weight may be tuned only on train-only internal-dev and
  must be logged.
- Report all completed seeds; never promote only the best seed.
- A diagnostic witness establishes information availability or recoverability;
  it is not silently promoted into a main LLM method.

## V1.2 source boundary

Published source may be inspected, but active paper mechanisms are independently
and minimally reimplemented inside the shared project harness. Record upstream
URL, commit, license, migrated mechanism, omitted pieces, and local files in
`experiments/pps_baseline_cards.md`.

Old B0--B9 role names, doc 34 phases, matched-null repair rounds, and proposed-
system eligibility rules are historical and do not add work to V1.2.
