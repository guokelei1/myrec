# C41 semantic-carrier routing boundary proposal

## Evidence boundary

C39's untied learned `V/O/FFN` erased true-user specificity. Post-terminal
diagnostics on already-open C39-A restored large true-over-wrong margins as
soon as history values and candidate readout returned to raw BGE coordinates.
C40 then tested full metric coupling. Although its coupled primary recovered a
planted teacher, the simpler `selection_only` reduction beat it in all three
seeds by `+0.032690` to `+0.043914` and retained clean-minus-wrong margins near
`0.49`.

This supports one minimal boundary: learn where to read, not what an admitted
history event means.

## Operator

For normalized frozen-LM states and routing head `r`:

```text
R_r(x) = normalize(x + U_r V_r x)
a_jr = softmax_j(<R_r(q), R_r(h_j)> / tau)
p_r = sum_j a_jr h_j                 # raw LM value, not R_r(h_j)
q'_r = normalize(q + p_r)            # raw LM query
d_ir = <c_i,q'_r> - <c_i,q>          # raw LM candidate readout
d_i = mean_r d_ir.
```

The trainable router may reallocate nonnegative attention mass but cannot
rotate, scale, or rewrite any event value. No downstream learned FFN, `W_V`,
`W_O`, candidate head, or value projection exists. No-history, absent-query,
and exact-repeat corrections are exactly zero.

## Matched controls

All trainable modes own identical factor tensors and paired initialization:

1. `semantic_routing` — four routing heads, raw semantic carrier;
2. `single_wide_routing` — one rank-64 router assembled from the same factors;
3. `asymmetric_routing` — query head `r` matches history head `r+1`;
4. `coupled_content` — C40 primary, which also rewrites values/readout.

Parameter-free fixed semantic attention and uniform history are functional
controls. The already-trained C38 query-attended unprojected model is the
external strong control; it used the same fit set and BGE snapshot.

## Interpretation boundary

Fixing values/projections is directly covered by *Simplifying Transformer
Blocks*. QKV projection sharing, attention's routing/content split, residual
adapters, and target-aware history attention are known. C41 therefore has
`novelty_status=boundary_only` before real outcomes. It can become a validated
backbone only if it beats fixed semantic attention, all equal-parameter modes,
and C38 on untouched data while true history beats matched wrong history.

If it passes, a separately reviewed successor must identify a
ranking-specific, non-reducible primitive. If it fails, semantic fixed routing
or C38 remains the architecture boundary and C41 closes without rescue.
