# C52 reduction and witness audit

## What reduces

If query-token aggregation is uniform or linear, the tensor
`e_ik=<c*_ik,P_H q_k>` may be averaged after token matching.  That is only a
token-granular version of C47 plain KRR.  `linearized_token_krr` binds the
stronger first-order reduction `sum_k softmax(b_i)_k e_ik`.

If `P_H q_k` is replaced by a nonnegative normalized history mean, the module
is ordinary two-stage attention.  `token_softmax` binds that reduction.  If
history states become values rather than logits, the module returns to C26,
C39, PEAR/TEM, or generic edge-conditioned message passing; that branch is not
part of C52.

## Nonlinear allocation witness

For two query concepts with equal base logits, take candidates whose evidence
vectors are `e_1=(a,-a)` and `e_2=(0,0)`.  Both have the same uniform and
first-order mean evidence, zero.  Nevertheless,

```text
tau log[(exp(a/tau)+exp(-a/tau))/2] > 0
```

for `a != 0`, while the second candidate remains zero.  Thus the primary cannot
be moved after query-concept allocation as a single linear pooled KRR score.
The same witness changes the factual semantic carrier mixture `aH` even when
the no-history carrier average is identical.

This is not a universal-function novelty claim: a sufficiently large generic
Transformer can approximate the map.  The falsifiable architectural claim is
the parameter-free KRR bias placement and its semantic-value invariant.  It
must pay rent over both the exact reductions and generic attention before any
fresh role is touched.
