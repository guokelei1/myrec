# C13 hypothetical proposal

Status: paper-only formulation; binding outcome is rejection.

## Motivation and narrow question

C05 showed that a history residual may move parameters and loss while becoming
candidate-common, which cannot change ranking.  Candidate centring correctly
removes that direction.  A later zero-sum attempt produced a hidden write around
`1.83e-5` with no candidate order change.  C13 asks whether normalizing a
centred history write over the **candidate set** can prevent both common-mode
and vanishing-amplitude collapse.

This observation establishes an amplitude problem, not that the tiny residual
is ranking-aligned.  C13 is therefore obliged to distinguish “recover aligned
small signal” from “force arbitrary small noise to move the ranking.”

## Shared Transformer information flow

The hypothetical ranker has a history-blind candidate state

```text
z_i^0 = Transformer_theta(q, c_i),
```

and a standard late history-to-candidate cross-attention residual

```text
R_i = CrossAttention_theta(z_i^0, H).
```

The proposed primitive acts only at this internal write interface.  It never
reads `base_score_i` and has no scorer router, dataset branch, query classifier,
or independent reliability gate.

Let `R` be the `C x d` residual matrix and
`P=I-11^T/C`.  First compute `X=PR`, so every candidate-common residual is
annihilated exactly.

Two possible meanings of “set-relative RMS/whitening” must be separated:

### Scalar set RMS

```text
sigma^2 = ||X||_F^2 / (C d)
Y_rms   = X / sqrt(sigma^2 + epsilon).
```

This is stable and cheap, but is only a request-dependent positive scalar times
ordinary centred attention.

### Candidate-set spectral whitening

For thin SVD `X=U diag(s_k) V^T`, define

```text
Y_white = U diag(s_k / sqrt(s_k^2/C + epsilon)) V^T.
```

Equivalently this is a regularized ZCA transform of the candidate-set
covariance, implemented through the smaller candidate Gram matrix when `C<d`.
It changes singular-mode ratios and is not generally a scalar.

Both variants receive one final request-global contraction

```text
W = rho * P Y / (rho + ||P Y||_F),
z_i = z_i^0 + W_i.
```

The contraction preserves zero sum and guarantees `||W||_F<rho`.

## Required structural contracts

- The base candidate mask has no history edge.
- With no valid history, final logits select the untouched base tensor bitwise;
  the normalizer is diagnostic-only and cannot perturb fallback.
- For candidate permutation matrix `Q`, `R'=QR` implies `W'=QW`; centring,
  singular values, Gram functional calculus, and the global bound are all
  permutation equivariant.
- `sum_i W_i=0` and any `R=1m^T` maps exactly to zero.
- Exact item-token recurrence occupies one coordinate of the same late evidence
  vector and is read by the same head with `softplus(alpha)>0`.  It is not an
  external item score.
- The architecture consumes generic token IDs and masks only.

These contracts are achievable.  They do not establish that normalization is
a novel or safe evidence-use primitive.
