# C41 frozen Amazon train-internal boundary gate

Status: pre-outcome; immutable after the proposal lock.

## Cohorts and isolation

- Fit is exactly C38's 6,000 fit requests and may reuse only its fit labels.
- C41-A is exactly C38 delayed-B: 1,200 requests never previously feature-
  materialized, scored, or label-opened. It has zero overlap with C38-A,
  C39-A, and prior C38 feature indices.
- C41 delayed-B is C38 escrow, also never feature-materialized, scored, or
  label-opened. It remains closed regardless of C41-A outcome.
- Wrong histories are the frozen C38 same-length-bin distinct-user donors with
  100% coverage and zero same-user assignments.
- Upstream dev/test and all qrels remain closed.

Only fit and C41-A may be label-free encoded after proposal lock. G0 then opens
fit labels only. All four trainable modes and all functional controls must
produce A scores before A labels can open.

## Modes and capacity

The four trainable modes own 49,152 parameters, paired initialization, the same
fit data, optimizer, full candidate lists, steps, and one epoch:

1. `semantic_routing` primary;
2. `single_wide_routing`;
3. `asymmetric_routing`;
4. `coupled_content` (C40 primary).

Functional controls are parameter-free fixed semantic attention, uniform raw
history, and the three frozen C38 query-attended-unprojected checkpoints. C38
was trained on the exact same fit set and snapshot before C41-A existed as an
opened outcome.

## A0: label-free structure and activity

Before A labels open, every trainable mode must have finite training, both
factors active, changed parameters, equal capacity, and paired initialization.
All modes and controls must be deterministic, candidate-permutation equivariant,
and exact-base under no-history, absent-query, and repeat fallback.

For the primary:

- profile reproduction from attention-weighted raw LM events has max error
  `<=1e-6`; attention is nonnegative and sums to one within `1e-6`;
- primary versus base changes at least 5% of complete orders and 1% of top-10;
- primary versus each matched and functional control changes at least 2% of
  orders and 0.5% of top-10;
- true versus wrong history changes at least 2% of orders and 0.5% of top-10.

Any A0 failure closes C41 with A labels unopened.

## A1: utility and specificity

Only after all A0 checks pass may the shared evaluator open C41-A labels. The
seed-averaged primary must satisfy every condition:

1. over frozen BGE/base: mean at least `+0.002`, 95% paired-bootstrap lower
   bound positive, every seed and all three request-hash folds positive;
2. over C38 unprojected: mean at least `+0.001`, positive lower bound, every
   paired seed and fold positive;
3. over fixed semantic attention: the same `+0.001`/CI/seed/fold conditions;
4. over each trainable matched mode: mean at least `+0.0005`, positive lower
   bound, every seed nonnegative;
5. true over wrong-user history: positive 95% lower bound;
6. clicked-minus-negative primary correction: positive 95% lower bound.

These conjunctive gates deliberately ask whether learned routing pays rent over
the already-good semantic carrier. Failure is terminal: no rank/head/
temperature/scale/loss/epoch/seed/cohort rescue and no delayed-B/dev/test.

Passing A1 validates a strong architecture boundary, not novelty. It authorizes
a new design review for a non-reducible ranking primitive and a separately
frozen cross-dataset confirmation; it does not directly authorize dev/test.
