# C22 synthetic GPU gate protocol

Status: **pre-outcome draft; binding after source/config/tests are hash-locked
before the first optimizer step**.

## G0 structural gate

Tests must establish:

1. block masks and prefix RMSNorm satisfy the full zero-Jacobian contract;
2. recurrence-to-transfer Jacobians are nonzero on a constructed witness;
3. ordinary RMSNorm and dense mixing each violate the protected direction;
4. exact identity affects only the recurrence quotient at input;
5. no-history and query-absent rows return the base bitwise;
6. candidate permutation equivariance and candidate-centred transfer;
7. finite gradients reach every active prefix/transfer projection and readout;
8. all modes have identical parameter names, count and initialization;
9. zeroing the transfer quotient removes supported non-repeat gains but leaves
   recurrence writes; and
10. no repository data, evaluator, qrel or external model path exists.

## G1 one-shot learned gate

- three seeds on physical GPU 2;
- sign-symmetric candidate sets so query/candidate marginals do not identify the
  target without the appropriate history evidence;
- strata: no-history, exact recurrence and supported non-repeat;
- train-only semantic lures are independently resampled at evaluation so a
  protected recurrence path must coexist with transferable evidence rather than
  memorize one shared shortcut;
- modes: `filtration`, `dense`, `parallel`, `final_projection` with identical
  parameters, initialization, steps and batches;
- corruptions: wrong history, shuffled events, query mask, coarse-only relation
  removal and exact-identity removal;
- one attempt, no sweep/retry/early stopping, no repository data.

Every seed must pass all conditions:

1. no-history accuracy `>=0.98`, repeat accuracy `>=0.98`, supported non-repeat
   accuracy `>=0.85`;
2. supported gain over the filtration model's own history-blind base `>=0.60`
   with base supported accuracy `<=0.20`;
3. repeat and supported accuracy each exceed every control by `>=0.03`, and
   minimum-stratum accuracy exceeds every control by `>=0.05`;
4. primary corruption target-margin gain retention is at most `0.25` for wrong,
   shuffle, query-mask and coarse-only;
5. removing exact identity reduces repeat target margin by at least `0.50` but
   changes supported non-repeat margin by at most `0.05`;
6. protected Jacobian maxima are `<=1e-7`, recurrence-to-transfer Jacobian norm
   is positive, and dense/ordinary-norm counterexamples are positive;
7. at least 5% of history-present rows change order relative to base;
8. deterministic rescore, matched parameter/init, finite values/gradients,
   candidate permutation and bitwise no-history checks pass.

Failure closes C22.  Passage authorizes only a separately frozen real
train-internal gate, not dev/test or full implementation.
