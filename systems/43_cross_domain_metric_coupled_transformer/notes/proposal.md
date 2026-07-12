# C43 metric-coupled cross-domain proposal

## Falsifiable primitive

For head `r`, one residual low-rank semantic metric `R_r` must be used for all
four operations: query/history selection, history value transport, transported
query state, and candidate readout:

```text
R_r(x) = normalize(x + U_r V_r x)
a_rj   = softmax_j(<R_r(q), R_r(h_j)> / tau)
p_r    = sum_j a_rj R_r(h_j)
q'_r   = normalize(R_r(q) + p_r)
d_ri   = <R_r(c_i), q'_r> - <R_r(c_i), R_r(q)>
d_i    = mean_r d_ri
```

The hypothesis is not merely that history attention helps. It is that the
same head metric must close the evidence loop; otherwise selected evidence is
interpreted or read in a mismatched coordinate system.

## Equal-capacity controls

All modes own the same `[4,16,512]` down and `[4,512,16]` up tensors and paired
initialization:

1. `multihead_coupled` — primary, identity head loop;
2. `selection_only` — learned Q/K routing with immutable LM values/readout;
3. `shifted_loop` — selection head `r` is consumed by content head `r+1`;
4. `single_wide_coupled` — all factors collapse into one rank-64 metric.

Parameter-free fixed semantic attention and uniform history are functional
controls. D2p is the non-personalized base. Exact-repeat requests return the
already frozen item-only ranking and receive no learned correction.

## Interpretation boundary

C41/C42 established Amazon utility and specificity but not unique structural
rent. C43 passes only if the coupled primary beats D2p, every equal-capacity
control, and fixed semantic attention with positive paired intervals and every
seed/fold sign, while true history beats matched wrong history. Any failure is
terminal.

Metric/QKV sharing and identity projections have direct prior art. Even a C43
pass validates a cross-domain foundation only; paper-level innovation still
requires a separately frozen recommendation-specific primitive.
