# Algebraic reduction audit

Let `z_i` be the query/candidate state, `h_j` a history-event state, and `p_ij`
ordinary candidate-to-event allocation.  C15 proposes

```text
o_i = sum_j p_ij Phi(z_i,h_j).
```

## General vector-valued bilinear map

Any bilinear map `Phi: R^d x R^d -> R^m` has a tensor `T` and can be written

```text
Phi(z,h) = M(z) h,
M(z)_mb = sum_a T_mab z_a.
```

Because `M(z_i)` is independent of event index after fixing the candidate,

```text
o_i = M(z_i) [sum_j p_ij h_j].
```

Thus bilinear candidate/event values are exactly a candidate-conditioned linear
map applied **after** ordinary attention pooling.  They do not retain additional
event structure through aggregation.

## Low-rank CP/Hadamard form

For the natural rank-`r` implementation

```text
Phi(z,h) = U [(A z) elementwise-multiply (B h)],
```

the aggregate is

```text
o_i = U diag(A z_i) [sum_j p_ij B h_j].
```

This is feature-wise candidate modulation of an ordinary pooled value: FiLM,
diagonal hyperadapter, or a low-rank candidate-generated linear layer.  Adding
query and candidate as separate trilinear factors only changes the conditioning
vector used to generate that post-pooling map.

If a standard value term is retained,

```text
sum_j p_ij [W_V h_j + Phi(z_i,h_j)],
```

the result is ordinary attention plus the same reducible adapter.

## Impossibility of the requested multi-event witness

Take two event sets with the same allocation-weighted history aggregate.  For
example, under uniform allocation, `{a,-a}` and `{0,0}` both average to zero.
Every bilinear `Phi` gives aggregate zero for both:

```text
M(z)(a-a)/2 = M(z)(0+0)/2 = 0.
```

Therefore the constrained bilinear family cannot supply the required witness
“same ordinary aggregate, different aggregated pair values.”  Individual pair
values may differ, but their sum contains no information beyond the pooled
history vector.

## Relationship to post-SDPA gating

If `M(z)=Diag(g(z)) W_V`, the reduction is exactly an elementwise gate/FiLM
applied to the SDPA output before `W_O`.  More general low-rank `M(z)` is a
candidate hypernetwork generating a post-attention adapter.  Moving the notation
inside the event sum does not change the function.
