# C20 reduction audit

Decision: **conditional pass for one synthetic falsifier; high novelty risk**.

## What is already known

Differentiable optimization layers are established by
[OptNet](https://proceedings.mlr.press/v70/amos17a.html) and general
[differentiable convex optimization layers](https://papers.nips.cc/paper_files/paper/2019/hash/9ce3c52fc54362e22053399d3181c638-Abstract.html).
Fixed-depth learned/unrolled sparse inference is established by
[LISTA](https://icml.cc/Conferences/2010/papers/449.pdf).  Nonnegative
factorization and convex-cone representation are also established; C20 cannot
claim any of those ingredients.

## Exact reductions

For orthogonal unit transition columns and zero ridge,

```text
alpha_j* = relu(d_j^T r).
```

Thus HTCT reduces exactly to unnormalized ReLU retrieval in that special case.
For a one-step projected-gradient update it is also exactly a ReLU similarity
write.  Both are mandatory controls.

If the nonnegativity projection is removed, the converged solution is ridge
projection onto `span(D)`.  It reconstructs a direction and its negative
equally well and is the mandatory sign control.

If coefficients are constrained to the probability simplex, the layer becomes
a standard positive weighted sum/attention retrieval over transition values.
That is the mandatory attention control.

## Surviving boundary

For a nonorthogonal dictionary, converged NNLS is a piecewise-linear map whose
active coefficients are jointly determined by the Gram matrix `D^T D`.  It is
not pointwise equal to independent ReLU weights, simplex attention, or
unconstrained projection.  A two-column constructive test must exhibit all
three separations before training.

This does not prove function-class novelty: a sufficiently deep Transformer,
generic recurrent optimizer or differentiable QP layer can emulate the fixed
computation.  The bounded claim is only that **positive compositional
reconstructibility of a candidate displacement from request-local history
transitions** is a distinct PPS inductive bias.  If matched controls explain
the synthetic result, C20 stops.
