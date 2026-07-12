# C77 data-free design-gate protocol

C77 reuses without modification the C76 generator and its train/validation
nuisance reversal.  Three seeds and five modes use identical initialization,
parameter count, batches, optimizer, 500 steps, and candidate surfaces.

## G0

- frozen anchor hash unchanged before/after optimization;
- unsupported token-to-personalized-score Jacobian exactly zero;
- admitted C-H and H-C edges exist and produce nonzero changes;
- no-history/base, query-mask/base, repeat/item-only, determinism, and
  candidate permutation pass within `2e-6`;
- interaction Transformer and output head receive finite nonzero gradients;
- all five modes have equal trainable parameters.

## D1

In every seed, the primary must:

- reach supported accuracy `>=0.75` and beat the base by `>=0.10`;
- keep repeat/no-history accuracy `>=0.95`;
- retain at most 30% of clean margin under wrong history or query mask;
- retain at least 80% under history-event permutation;
- change at least 5% of supported base orders;
- beat every registered graph reduction by `>=0.02` supported accuracy.

All mode losses must decrease and remain finite.  Failure is terminal before
repository data and consumes the first post-C76 architecture update.
