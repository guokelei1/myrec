# C42 frozen confirmation protocol

Status: pre-outcome; immutable after proposal lock.

## Execution order

1. trigger report and label-free C42 selection pass;
2. proposal lock binds design, cohort, source, all checkpoints, and evaluator;
3. label-free C42-A feature encoding only;
4. G0 confirms isolation/coverage and keeps A labels closed;
5. execution lock binds features before any C42 score;
6. three frozen checkpoint groups score true/wrong histories and all controls;
7. A0 is computed label-free; only a full pass opens A labels for A1.

No training occurs. C42 has no delayed-B and never opens dev/test.

## A0

- all checkpoint/source/config hashes match;
- zero optimizer steps and exact C41 checkpoint state hashes;
- finite, deterministic, candidate-permutation-equivariant scores;
- no-history, absent-query, and repeat corrections are exactly zero;
- primary true versus base and every control changes at least 2% of orders and
  0.5% of top-10 sets;
- primary true versus wrong changes at least 2% of orders and 0.5% of top-10;
- coupled loop assignments are identity and all semantic states are finite.

## A1

Seed-averaged primary must satisfy all:

1. versus base: mean `>=+0.002`, CI lower `>0`, every seed/fold positive;
2. versus C38: mean `>=+0.001`, CI lower `>0`, every seed/fold positive;
3. versus each C41 matched control: mean `>=+0.0005`, CI lower `>0`, every
   seed nonnegative;
4. true versus wrong: CI lower `>0`, every seed/fold positive;
5. clicked-minus-negative correction: CI lower `>0`.

Failure is terminal. No new seeds, retraining, threshold, rank/head,
temperature, scale, loss, cohort, or checkpoint rescue is allowed.
