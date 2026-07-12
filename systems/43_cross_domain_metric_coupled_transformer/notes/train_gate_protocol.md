# C43 frozen KuaiSearch transfer gate

Status: pre-outcome; immutable after proposal lock.

## Order

1. materialize label-free inherited selection;
2. freeze proposal, source, sources, architecture, and thresholds;
3. materialize fit/A features and fit labels only;
4. G0 verifies C37 role isolation and keeps A labels closed;
5. freeze execution inputs;
6. train and score all four modes for all three seeds before A labels;
7. A0 verifies structure, capacity, determinism, fallbacks, and activity;
8. only a complete A0 pass may open C43-A train-internal labels once.

## A0

- paired initialization, equal 65,536 parameters, active gradients and updates;
- exact mode loop assignments and finite normalized states;
- deterministic and candidate-permutation-equivariant scores;
- no-history/query-absent correction exactly zero;
- exact-repeat score exactly equals the frozen item-only fallback;
- primary changes at least 2% of orders and 0.5% of top-10 sets versus base and
  every control; true versus wrong satisfies the same activity floor.

## A1

Seed-averaged primary must satisfy all:

1. versus D2p: mean at least `+0.002`, CI lower `>0`, every seed/fold positive;
2. versus every equal-capacity mode: mean at least `+0.0005`, CI lower `>0`,
   every seed/fold positive;
3. versus fixed semantic attention: mean at least `+0.001`, CI lower `>0`,
   every seed/fold positive;
4. true versus wrong history: CI lower `>0`, every seed/fold positive;
5. clicked-minus-unclicked correction: CI lower `>0`.

Failure is terminal. No seed, cohort, rank/head, temperature, scale, loss,
epoch, width bridge, encoder, threshold, or checkpoint rescue is allowed.
