# C48 proposal — signed influence consensus

## Observation and primitive

C47 established that candidate self-support is the wrong fidelity object.  On
KuaiSearch, plain KRR correction had a positive clicked direction while
candidate support did not; multiplying them removed signal.  On Amazon,
candidate support was highly predictive but correlated strongly with the
query-base score, so it did not pay rent over ordinary semantic attention.

C48 instead decomposes the KRR correction into signed event influences:

`z_jc = <h_j,c> [(HH^T + I)^-1 Hq]_j`, with `e_c = sum_j z_jc`.

Its sole primitive is **signed influence consensus**:

`kappa_c = |sum_j z_jc| / (sum_j |z_jc| + eps)`,

`delta_c = kappa_c * e_c`.

The candidate write is unchanged when all event influences agree and contracts
only when the exact KRR decomposition cancels internally.  This tests
directional reliability rather than membership in the history span.  It has no
history-length, dataset, query-type, category, or outcome-conditioned branch.

If promoted later, this solve replaces a token-mixing head inside the
end-to-end Transformer ranking core; it is not a fixed-score router.  Exact
recurrence and no-history fallbacks remain separate output contracts.

## Current gate boundary

The first execution uses only the already-open C47 A cohorts.  It is a
formulation falsifier, not new evidence.  C48 stops before fresh selection or
training unless the fixed operator on both domains beats query base and plain
KRR with positive intervals, has every fold positive versus plain KRR, has
positive mean margins over fixed softmax and the signed-L1 nearest control, and
retains positive true/wrong and clicked-direction intervals.
