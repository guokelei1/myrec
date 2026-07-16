# Paper evidence constraints

Status: supporting evidence-hygiene rules. Current methods, datasets, budgets,
and execution order come from `experiments/motivation_v1_2/plan.md`.

1. Use one unified record contract with strictly prior history, a fixed
   candidate slate, and labels outside internal-dev/confirmation records.
2. Keep candidate identities, one shared evaluator, and candidate/request
   hashes comparable across methods.
3. Keep controls claim-specific: `full-null` measures incremental history
   contribution, wrong-user tests provenance, and shuffle is only for an order
   claim.
4. Make checkpoint and ordinary tuning decisions on train-only internal-dev.
   Freeze methods and analysis before final holdout outcomes.
5. Separate mechanical response, ranking quality, recurrence, strict transfer,
   provenance, uncertainty, and data sufficiency.
6. Preserve negative and contradictory results; do not rescue a claim with an
   outcome-selected seed, method, slice, or endpoint.

Former doc 34/E0--E8, C01--C80, R0, and Failure Card workflows are inactive and
are not part of Motivation V1.2.
