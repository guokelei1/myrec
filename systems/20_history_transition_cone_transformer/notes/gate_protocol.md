# C20 synthetic gate protocol

Status: **pre-outcome draft; values become binding when source/config/tests are
hash-locked before the first optimizer step**.

## G0 structural gate

Tests must establish:

1. a hand-computed two-column nonorthogonal NNLS witness;
2. positive reconstruction and reverse rejection for a one-ray cone;
3. exact separation from span, one-step ReLU and simplex retrieval;
4. every projected iterate has nonnegative coefficients and non-increasing
   frozen quadratic objective within numerical tolerance;
5. no/single-event history returns base scores bitwise;
6. candidate permutation equivariance and common-translation invariance;
7. all modes have identical parameter names, count and initialization;
8. finite nonzero gradients reach lower Transformer, relation projections,
   transition dictionary and hidden-state write;
9. zeroing the hidden write removes every score effect despite nonzero solver
   coefficients; and
10. no repository-data/evaluator path exists.

## G1 one-shot synthetic GPU gate

- three predeclared seeds, physical GPU 0, deterministic CUDA;
- independent train/eval splits with no-history, exact-repeat and supported
  positive-composition strata;
- eight candidates arranged as four sign-symmetric displacement pairs, so the
  target is not identifiable from candidate/query features without history;
- correlated transition dictionaries, with the target a positive composition
  and a hard reverse candidate of equal marginal geometry;
- modes `cone`, `span`, `relu1`, `simplex`, `pooled_mlp` with identical
  parameters, initialization, optimizer steps and batch schedule;
- corruptions: wrong history, event shuffle, query mask, coarse relation
  removal and sequence reversal;
- one learned execution, no retry/sweep/early stopping;
- repository/standardized data, dev evaluator and test access: zero.

All three seeds must pass every frozen threshold:

1. repeat accuracy `>=0.98` and supported accuracy `>=0.85`;
2. cone supported accuracy exceeds its own history-blind base by `>=0.60` while
   base supported accuracy is `<=0.20`;
3. cone exceeds every structured control on supported accuracy by `>=0.05`
   and on the minimum of repeat/supported by `>=0.03`;
4. clean target-margin gain is positive and each primary corruption retains at
   most `0.25`;
5. reversing history retains at most `0.10` and changes the target-versus-
   reverse-candidate margin sign;
6. at least 50% of supported rows use two or more positive transition
   coefficients and have reconstruction-error reduction `>=0.25`;
7. no-history is bitwise base, deterministic rescore passes, all values and
   gradients are finite, and matched parameter/init audits pass;
8. CPU candidate permutation error is `<=1e-5`; the locked CUDA audit uses
   `<=2e-4`, chosen before C20 outcome to cover the deterministic reduction
   order observed in C19 without weakening algebraic equivariance.

Failure closes C20.  Passage authorizes only a separately frozen real
train-internal design, never direct dev/test use.
