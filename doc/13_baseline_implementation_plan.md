# Baseline implementation plan for the new direction

Status: active. This plan covers controls for doc 34, not the old B0–B9
leaderboard as a paper claim.

## Required roles

1. deterministic popularity/source-order and BM25 controls;
2. an eligible query-candidate dense/cross-encoder control;
3. a query-only strong base (`E-QC`/`D-QC`);
4. an ordinary full-token history model (`E-FULL`/`D-FULL`);
5. one traditional query-aware personalized control;
6. one train-only cross-fitted signal witness.

The encoder and decoder families must use matched input boundaries, candidate
scoring, data split, and evaluation. `E-FULL`/`D-FULL` are ordinary controls,
not proposed architectures.

## Tuning and reporting

Use a small pre-registered budget per family. Tune only ordinary choices such
as objective, learning rate, history length, truncation, capacity, and
initialization. Record every dev call. Report frozen multi-seed results, not
the best seed. A family that cannot produce an adequate query-candidate base
is excluded from the shared-failure claim.

## Fairness boundary

All methods read the standardized interface and fixed candidate slate. A
baseline may use identity, text, action, and time only when those fields are
available in the same record and declared in its boundary card. A witness is
diagnostic only and cannot become the proposed method by renaming it.
