# Jacobian and constructive-witness audit

## Direct value-path Jacobian

Holding allocation fixed, candidate-independent attention has

```text
partial o_i / partial h_j = p_ij W_V.
```

Bilinear C15 has

```text
partial o_i / partial h_j = p_ij M(z_i),
```

which is candidate-dependent but event-independent after the scalar allocation.
It factors into a candidate-only matrix and the event scalar, matching a
post-pooled candidate adapter.  Allocation derivatives do not change the value
path reduction because both primary and control share `p_ij`.

## Necessary non-factorization condition

A genuinely pair-specific value must make

```text
J(z,h) = partial Phi(z,h) / partial h
```

depend jointly on `z` and `h`, not admit the registered low-rank separation
`sum_k A_k(z) B_k(h)`.  For a scalar Jacobian, a minimal two-by-two witness is

```text
J(z1,h1) J(z2,h2) != J(z1,h2) J(z2,h1).
```

All rank-one factorizations make this determinant zero; rank-`r` versions require
the corresponding Jacobian block matrix to exceed rank `r`.

## Joint-nonlinear witness

The map

```text
Phi(z,h) = U sigma(Az + Bh)
```

is generically non-factorized because

```text
J(z,h) = U Diag(sigma'(Az+Bh)) B
```

depends on the pair before aggregation.  For uniform allocation, event sets
`{a,-a}` and `{0,0}` have the same ordinary mean, while

```text
[sigma(Az+Ba)+sigma(Az-Ba)]/2
```

generally differs from `sigma(Az)`.  This supplies the requested construction.

It does not rescue C15.  `Phi(z,h)` is now precisely a neural message function
conditioned on both endpoints/edge.  A small MLP is a dynamic edge-conditioned
filter; an unrestricted MLP is the full existing message-passing class.  The
witness establishes nonlinearity, not a new mechanism.

## Structural contracts do not alter the reduction

Zero-valued NULL/no-history handling, candidate centring, one global hidden
bound, a monotone exact coordinate, and small non-zero LayerScale apply after
the pair aggregation.  They are necessary safety contracts but preserve equality
between algebraically equal aggregates.  LayerScale also cannot carry novelty;
it only avoids exact zero-initialization gradients.
