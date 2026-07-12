# C19 mechanism fingerprint

## Identity

- candidate: `c19`
- primitive: diagonal-plus-skew temporal bilinear affinity operator;
- intervention: multiplicatively gated candidate-token residual inside the
  Transformer ranking core; no direct score addition;
- state: query trace `A`, per-candidate trace `B_i`, and a fixed causal shift;
- inference inputs: query, ordered prior history, candidate set, masks and exact
  identity relation;
- complexity: `O(CHd + CH)` after token encoding;
- no-history: exact structural no-op.

## Algebraic witnesses

Let `K=I+lambda(S-S^T)`.

1. `S-S^T` is skew-symmetric, so `x^T(S-S^T)x=0`: self/same-trace support comes
   only from the nonnegative diagonal overlap and cannot be spuriously created
   by temporal orientation.
2. Under sequence reversal matrix `R`,
   `R^T(S-S^T)R=-(S-S^T)`: the oriented score flips sign while the diagonal
   score is invariant.
3. Two histories with the same event multiset and identical diagonal overlap
   can have opposite oriented scores.  Any order-blind target attention or
   pooled adapter must tie them.
4. A forward induction term `A^T S B_i` cannot penalize a candidate that is a
   query-like event's predecessor; the cofactor subtracts exactly that reverse
   path.
5. If all candidates have the same `B_i`, centring removes the complete write;
   no candidate-common translation survives.
6. Zeroing the shared hidden-state write map makes personalized scores bitwise
   equal to base scores even when the temporal evidence scalar is nonzero; the
   operator has no score-side bypass.

## Degenerations

| mode | coefficients on `(D,F,R)` | interpretation |
|---|---|---|
| `oriented` | `(+, +lambda, -lambda)` | proposed cofactor |
| `diagonal` | `(+, 0, 0)` | same-event overlap / target-attention-like control |
| `forward` | `(+, +lambda, 0)` | ordinary successor induction |
| `symmetric` | `(+, +lambda/2, +lambda/2)` | order-insensitive adjacency |
| `free_signed` | `(+, lambda_f, lambda_r)` | unrestricted two-lag signed control |

All modes instantiate the same parameters, including two raw lag scalars.  The
structured modes tie or mask those scalars only at the operator.  If
`free_signed` matches OLT without worse corruption behavior, the performance
advantage is not attributable to the skew restriction even if its algebraic
witness remains true.
