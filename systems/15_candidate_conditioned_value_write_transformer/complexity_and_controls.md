# Complexity and matched controls

## Parameters and compute

For hidden width `d`, pair rank `r`, `C` candidates, and `H` events, the
low-rank bilinear or additive joint map uses approximately

```text
A: r x d, B: r x d, U: d_v x r,
```

or `r(2d+d_v)` parameters per independently parameterized head/group.

Bilinear factorization can be evaluated after pooling and costs roughly ordinary
attention over width `r` plus `O(C r d_v)`.  This computational shortcut is
another manifestation of its post-pooling reduction.

Joint nonlinear `U sigma(Az_i+Bh_j)` must instantiate `O(CHr)` pair activations,
then costs `O(CHr+Crd_v)` after precomputing endpoint projections.  It is
feasible only for small `r`, but still adds pairwise work and memory versus
candidate-independent values.  A full pair MLP costs up to `O(CHd^2)` and is not
eligible as a lightweight primitive.

## Mandatory controls

Every control must share base states, allocation, NULL, candidate centring,
bound, LayerScale, objective, and initialization:

1. ordinary candidate-independent `W_V h_j`;
2. exact post-pooling low-rank map `M(z_i) sum_j p_ij h_j`;
3. candidate FiLM/diagonal hyperadapter on the pooled value;
4. Qiu-style post-SDPA elementwise gate;
5. matched edge-conditioned message `U sigma(Az_i+Bh_j)`;
6. same-parameter candidate-independent value MLP;
7. allocation-only DIN/ZAM target attention.

The exact post-pooling control must match bilinear C15 pointwise and in gradients.
The edge-conditioned control must match the nonlinear survivor.  Either equality
is a pre-implementation novelty failure.

Parameter counts alone are insufficient: report active-gradient parameters,
pair activations, peak memory, wall time, and maximum `C x H`.  Dummy parameters
cannot disguise an unrestricted pair MLP's extra capacity.
