# Algebraic and constructive audit

Let `X=PR` be the candidate-centred hidden residual.

## Ordinary centred attention

Scalar RMS is exactly

```text
Y_rms = k(R) X,   k(R)>0.
```

It preserves every hidden direction, singular-value ratio, and the candidate
ordering induced by any fixed linear readout of the residual.  It can change
the final order only by changing residual magnitude relative to the base path.
For each request it is identical to choosing a scalar rescale after seeing the
raw residual norm.  It is not a new candidate-evidence interaction.

Whitening is different only when `rank(X)>=2` and singular values differ.  Let
`u_1=(1,-1,0)/sqrt(2)` and `u_2=(1,1,-2)/sqrt(6)` be candidate-contrast vectors,
and let feature directions `v_1,v_2` be orthogonal.  For

```text
X = 10 u_1 v_1^T + 1 u_2 v_2^T,
```

ordinary centred attention retains the 10:1 spectrum, while near-exact
whitening maps it toward 1:1.  Thus whitening is not algebraically ordinary
attention or a scalar rescale.

## Fixed scalar rescale

The preceding rank-two witness also separates whitening from every
`aX` with fixed scalar `a`: no scalar can simultaneously map singular values
`(10,1)` to approximately `(1,1)`.  But scalar set RMS is itself an adaptive
scalar rescale, and whitening reduces to a scalar in each of these cases:

- `rank(X)<=1`, including every two-candidate request after centring;
- all non-zero singular values are equal;
- `epsilon` dominates every `s_k^2/C`, yielding
  `Y_white approximately X/sqrt(epsilon)`;
- downstream write/readout observes only one singular mode.

## Post-head score RMS normalization

Post-head normalization forms centred scalar residual scores
`q=P(Ra)` and returns `q/sqrt(mean(q^2)+epsilon)`.  It cannot change the ordering
inside `q` because it applies one positive scalar after projection.

Hidden whitening can change that ordering before projection.  In the rank-two
witness above, choose readout coefficients `a^T v_1=1` and `a^T v_2=-1`.
The raw projected residual is `10u_1-u_2`; near-whitening gives `u_1-u_2`.
For the three candidates, the top residual changes from candidate 1 to
candidate 3.  A post-head scalar cannot reproduce this.  The distinction is
real, but it comes from generic spectral reconditioning rather than evidence
fidelity.

## Per-candidate LayerNorm

Per-candidate LayerNorm normalizes feature coordinates independently inside
each row.  It neither observes candidate-set covariance nor guarantees
candidate-common annihilation.  If `R_i=m` for every candidate, ordinary
per-row LayerNorm can return the same non-zero row for all candidates, whereas
`PR=0` and C13 returns exactly zero.

Even if a control applies `P LayerNorm(R)` afterward, it is not set whitening:
perturbing one candidate changes only that row before the final mean shift,
whereas changing one row of `X` changes the set covariance and therefore every
whitened row.  Conversely, per-row LayerNorm can remove row-magnitude evidence
that set whitening retains through its shared spectrum.

## Candidate permutation proof

For permutation matrix `Q`, `P` commutes with `Q`, so `X'=QX`.  The candidate
Gram matrix transforms as `G'=QGQ^T`; any spectral function satisfies
`f(G')=Qf(G)Q^T`.  Hence `Y'=QY`, recentering gives `PY'=QPY`, and the global
Frobenius bound is invariant.  The primitive is exactly permutation equivariant
apart from ordinary floating-point eigensolver tolerances.

## Why this is not evidence selection

Neither variant uses query/history agreement beyond what ordinary cross
attention already encoded in `R`.  Scalar RMS changes only amplitude.  Full
whitening changes the residual condition number but has no variable that says
which singular direction is correct, repeated, corrupted, or wrong-user.  It
can force order change, but cannot by algebra alone make that order change
ranking-aligned.
