# C39 pre-implementation design gate

This gate is frozen before any C39 real fit label or ranking outcome. Passing
authorizes implementation of the train-internal runner; it is not evidence of
real ranking utility.

## D0 algebra and implementation

All checks are binding:

1. exact closed-form projection agrees with a hand-computed halfspace example;
2. every projected value satisfies the halfspace inequality within `1e-6`;
3. already feasible values are unchanged within `1e-7`;
4. the `{a,-a}` versus `{0,0}` witness has equal ordinary aggregates but
   different primary aggregates;
5. primary and all four controls have identical parameter count and paired
   initial state;
6. finite forward/backward and nonzero gradients reach `Q/K/V/W_O` and the
   shared FFN;
7. candidate permutation equivariance error is at most `1e-6`;
8. no-history, query-masked, and repeat-present corrections are exactly zero;
9. unsupported candidate/event/head edges are exactly zero;
10. primary, raw, post-pool, ray-only, and global-only produce finite and
    operator-distinct outputs on the non-repeat witness.

## D1 synthetic behavior

Three fixed seeds use a tiny candidate-ranking task with equal pooled history
means but different eventwise feasible components. The task is intentionally
only an operator falsifier. For each seed:

- the primary must improve NDCG@10 over the no-history base by at least `0.10`;
- the primary must beat `postpool_halfspace` and `global_only` by at least
  `0.05`;
- wrong-user history must reduce the primary gain by at least `0.05`;
- at least 5% of non-repeat candidate/event/head edges must be rejected and at
  least 5% of raw negative-readout edges must be changed by projection;
- no-history and repeat examples remain exactly base-equivalent.

The synthetic generator, seeds, thresholds, model dimensions, steps, and
controls must be committed before execution. A failure closes C39 without
real data. A pass authorizes only a new proposal lock and Amazon-C4
train-internal G0/A0/A1; dev/test and full implementation remain unauthorized.
