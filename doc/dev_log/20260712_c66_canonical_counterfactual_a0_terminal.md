# C66 canonical counterfactual state: A0 terminal

C66 was the sole mechanical continuation allowed by C65.  Stable SHA-256 item
keys forced all factual, NULL, and wrong-history branches into the same
candidate order and restored caller order afterward.  This reduced C65's
`3.64e-5` permutation error to bit-exact zero without changing any scientific
setting.

Three seeds then trained the hidden residual primary and three equal-parameter
controls on the exposed 4,800-request fit split.  All runs were finite,
gradient-active, deterministic, and exactly equivariant under full-candidate
fp32 scoring.  The primary was rank-active against the base, but matched wrong
history changed only `1/10/7` Top-10 sets out of 1,200.  All three seeds missed
the fixed 12-set minimum, and several final loss windows did not improve over
their initial windows.

The validation-label release gate failed and those labels remain unopened.
The architectural lesson is not that counterfactual computation is useless;
it is that output neutrality for wrong history is too weak to identify a
history-specific internal function.  A later candidate must change the
Transformer computation graph so that candidate ranking cannot be represented
without a history-derived operation.  It may not retune C66 or add another
post-hoc residual/gate around the same factual/NULL states.
