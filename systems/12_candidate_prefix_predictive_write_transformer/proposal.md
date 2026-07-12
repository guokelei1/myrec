# C12 paper proposal: candidate-prefix eventwise likelihood-ratio write

Status: paper-only hypothetical architecture; see `preimplementation_decision.md`
for the binding rejection.

## Question

C10 pooled history before candidate interaction.  C11 retained events but used
a candidate-independent decoder state, so candidate centring removed the shared
log-partition difference and left ordinary embedding/hidden similarity.  C12
asks whether making the **prediction context itself candidate-specific** avoids
that reduction.

## Single primitive

Let candidate `i` have an ordered token sequence `c_i,1:T`.  For every strictly
prior history event `e` and causal candidate position `t`, the same shared LM
computes

```text
h^H_i,e,t = F_theta(q, h_e,    c_i,<t)
h^0_i,t   = F_theta(q, NULL_e, c_i,<t)
g_i,e,t   = log p_theta(c_i,t | h^H_i,e,t)
            - log p_theta(c_i,t | h^0_i,t).
```

`h^0` is cacheable across events.  `NULL_e` has the same token count, positions,
and visibility mask as `h_e`, preventing a length/position shortcut.  A strict
causal mask forbids `c_i,t` and later candidate tokens from entering either
predictor.

The hypothetical late write is

```text
g'_i,e,t = g_i,e,t - mean_i g_i,e,t
r_i,e    = mean_t tanh(g'_i,e,t) E[c_i,t]
u_i      = EventIntegrator({r_i,e + position_e})
w_i      = rho * P_C(W u)_i / (rho + max_j ||P_C(W u)_j||)
score_i  = Head(z_i^base + w_i, log(1 + exact_count_i)).
```

The event and token axes survive until the late hidden write.  There is no
fixed-score router or dataset/query-type branch.

## Transformer contracts

- `z_i^base=F_theta(q,c_i)` is the end-to-end history-blind ranking path.
- The predictive and base paths share LM token embeddings and Transformer
  weights; the LM is load-bearing rather than an offline feature extractor.
- All candidates use identical weights.  Candidate centring and independent
  event integration make the result candidate-permutation equivariant.
- `w` is exactly zero-sum over candidates and each norm is strictly below `rho`.
- With no valid history, pointwise selection returns the untouched base tensor
  bitwise; diagnostic NULL computations cannot perturb it.
- Exact recurrence is a canonical item-token equality coordinate inside the same
  late evidence vector.  The same head reads it with `softplus(alpha)>0`; it is
  not a post-head item score.
- Token IDs, positions, and masks are generic.  No synthetic category/attribute,
  dataset, or query class appears in the architecture.

## Why candidate prefix matters

At `t=1`, the predictor has no candidate prefix and ordinarily falls back to the
C11 reduction.  Therefore at least one `t>=2` position must be load-bearing and
must pass a target-token leakage audit.  The primitive is not justified by an
item identifier alone; it depends on causal cross-token prediction conditioned
on event and candidate prefix.

## Intended claim boundary

Even if implemented and successful, the first claim would be only that
candidate-specific normalized token prediction provides a learnable eventwise
signal beyond hidden similarity.  It would not establish semantic transfer or
real ranking gain without a separately locked real train-internal gate.
