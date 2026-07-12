# Matched controls and complexity

## Controls

All controls must consume the exact same raw cross-attention residual `R` from
the same checkpoint.  No control may retrain a different base path.

| control | transformation | purpose |
|---|---|---|
| ordinary centred attention | `PR` plus the same final global bound | tests whether whitening adds more than exact common-mode removal |
| affine-free per-candidate LayerNorm | `P LN_row(R)` plus bound | distinguishes row normalization from set coupling |
| post-head score RMS | normalize `P Head(R)` after projection | tests whether the benefit is only score-scale calibration |
| fixed scalar rescale | frozen `a PR`, `a` fixed before outcomes | nearest scale control |
| learned global scalar | `softplus(a) PR` with matched training | tests optimization/initialization explanation |
| scalar set RMS | request-adaptive scalar branch | mandatory ablation of spectral whitening |

Post-head score normalization is a control only because the proposed system may
not read or normalize base scores as a router.

## Parameters

Candidate centring, scalar RMS, and exact whitening add zero learned parameters
when `epsilon` and `rho` are protocol constants.  Affine-free LayerNorm and
fixed/post-head controls are also zero-parameter.  The learned-scalar control has
one parameter; a matched one-parameter primary may expose a post-whitening
global `softplus(gamma)` before the fixed bound, but that scale cannot be used as
an independent reliability gate and must be ablated.

No per-dataset epsilon, learned request gate, bias/fallback branch, or base-score
input is allowed.

## FLOPs and numerical cost

For `C` candidates and width `d`:

- centring/scalar RMS/fixed scale: `O(Cd)`;
- per-candidate LayerNorm: `O(Cd)`;
- post-head score RMS: `O(C)` after the shared projection;
- full feature covariance whitening: `O(Cd^2+d^3)`, unacceptable for large
  Transformer width;
- thin/candidate-Gram whitening when `C<d`: `O(C^2d+C^3)` plus backward;
- `K` fixed Newton-Schulz iterations on the candidate Gram matrix:
  `O(C^2d+KC^3)`.

The candidate-Gram route is the only plausible implementation, but repeated or
nearly equal eigenvalues can make eigendecomposition gradients unstable.
Fixed-iteration whitening changes the operator and adds an iteration
hyperparameter.  Wall time, peak memory, determinism, and FP32/FP64 parity would
be binding engineering gates.

For quality attribution, controls may execute the same Gram construction and
decomposition as detached dummy work, but actual end-to-end latency must also be
reported without dummy matching.  Parameter matching cannot hide whitening's
real serving overhead.

## Structural edge cases

- `C=1`: centring yields zero; personalization cannot change a one-item order.
- `C=2`: centred residual rank is at most one, so whitening is only scalar
  rescaling and has no spectral novelty.
- duplicate candidates: common and duplicate-difference null spaces must remain
  finite and permutation equivariant.
- all-zero/near-zero residual: exact zero must stay zero; a logarithmic scale
  ladder must test continuity and gradients around epsilon.
