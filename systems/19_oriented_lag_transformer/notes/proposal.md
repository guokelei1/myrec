# C19 proposal

## Observation

C18 proved that a final order constraint can be exact, active and load-bearing
while non-repeat transfer remains completely base-equivalent.  The missing
object is therefore not another safety gate: it is a candidate-aligned transfer
direction.  Real standardized histories contain ordered item/event sequences
but no past query field, so the cheapest untested structural hypothesis is
whether **oriented item transitions** provide evidence beyond same-event
similarity.

## Single primitive

The shared Transformer produces positive normalized affinity traces over valid
history positions:

```text
A_j  = softmax_j(sim(q,h_j))
B_ij = softmax_j(sim(c_i,h_j) + identity_bias * exact_ij).
```

For the one-step shift matrix `S[j,j+1]=1`, OLT uses

```text
D_i = A^T B_i
F_i = A^T S B_i
R_i = A^T S^T B_i
E_i = D_i + lambda (F_i - R_i).
```

Equivalently, every adjacent pair contributes the determinant

```text
A_j B_i,j+1 - A_j+1 B_i,j.
```

`D` is a nonnegative same-position overlap.  `F-R` is an oriented temporal
cofactor: it rewards candidate-like evidence after query-like evidence and
penalizes the reverse order.  Candidate centring occurs before a bounded,
multiplicative residual is written into each candidate token state:

```text
c'_i = c_i + alpha * tanh(center_candidates(E_i)) * W(c_i).
```

The same learned rank head reads `c_i` and `c'_i`; there is no direct evidence
bonus at the score interface.  Empty history skips the entire operator and
returns the paired query/candidate Transformer score bitwise.

## Falsification

The oriented restriction is useful only if it:

1. preserves exact repeat through the diagonal term;
2. learns a supported non-repeat successor relation;
3. beats diagonal, forward-only and symmetric-lag controls;
4. is not worse than a free signed-lag control and is more corruption-stable;
5. flips the oriented component under sequence reversal;
6. vanishes or loses utility under wrong, shuffled, query-masked and
   coarse-semantic evidence; and
7. remains permutation-equivariant over candidates and bit-exact without
   history.

Passing only establishes a learnable structured operator.  Real-data utility
and global novelty require separate locks.
