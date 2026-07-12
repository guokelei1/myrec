# C25 train-gate outcome

Status: **terminal failure at label-free A0**.

The locked run completed three seeds, four 182,209-parameter and
compute-matched modes, two epochs each, in 591.45 seconds.  All fits were
finite and had active gradients.  Determinism, candidate permutation,
query-absent D2p, no-history D2p and repeat-present item-only contracts passed.

The primary changed 95/1,200 request orders (7.92%) relative to D2p, but only
2/1,200 top-10 memberships (0.17%, required 1%).  Frozen wrong histories
changed every numerical correction but only 1.25%--1.83% of request orders and
zero top-10 memberships (required 5% order change in every seed).  The
Möbius/joint/pairwise/trilinear loss traces were nearly identical.

Two strict floating-point checks also failed: anchored cancellation reached
`6.35e-6` versus `1e-6`, and candidate-centering accumulation reached
`3.54e-5` versus `1e-5`.  These do not rescue the mechanism: even ignoring both
numeric tolerances, top-10 activity and wrong-history load-bearing still fail.
No tolerance repair or rerun is allowed.

Internal-A, delayed-B, escrow, dev and test labels remain unopened, so C25 has
no ranking-utility verdict.  It closes anchored pure-three-way interaction on
the pooled D2 representations.  The next candidate must change representation
granularity rather than apply another algebraic operator to the same pooled
states.

Raw report SHA-256:
`c0af86f9663ff953bed04b09b6153f2e72edc65002fabfc636c045aff4028fb3`.
