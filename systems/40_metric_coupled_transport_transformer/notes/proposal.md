# C40 metric-coupled transport proposal

## Observation and diagnosis

C38 showed on Amazon-C4 that query-attended unprojected LM history transport
has large ranking value and that true history beats same-length wrong-user
history. C39 replaced the shared semantic map with learned multi-head
`Q/K/V/O/FFN`. Its common trainable backbone strongly beat frozen BGE, but true
and wrong histories tied and every halfspace-specific control matched or won.

Post-terminal diagnostics used only the already-open C39-A cohort. The frozen
C38 unprojected checkpoint retained a `+0.0344` true-minus-wrong NDCG@10 margin
on that same cohort, ruling out cohort signal absence. With C39 weights,
semantic attention plus learned `V/O/FFN` still had no specificity. Raw BGE
identity values with spherical dot-product transport restored a `+0.1310`
margin, while feeding the same values through the trained FFN reversed the
margin to `-0.0410`. Learned Q/K with identity values retained `+0.1087` but
reduced clean utility. No C39 reserve, dev, or test input was opened.

The failure locus is coordinate decoupling: an event can be selected in one
learned space, rewritten in a second, and scored in a third. A constraint after
rewriting cannot restore the lost semantic identity.

## Single primitive: closed evidence metric loop

For frozen normalized LM states `x` and head `r`, learn one residual map:

```text
T_r(x) = normalize(x + U_r V_r x).
```

The same `T_r` is used at every point in that head's evidence loop:

```text
q_r = T_r(q),  h_jr = T_r(h_j),  c_ir = T_r(c_i)
a_jr = softmax_j(<q_r,h_jr>/tau)
p_r = sum_j a_jr h_jr
q'_r = normalize(q_r + p_r)
d_ir = <c_ir,q'_r> - <c_ir,q_r>
d_i = mean_r d_ir.
```

The standard base plus `d_i` is the final score. Selection, content, transport,
and readout share one learned metric per head. No independent value/output map
or FFN can reverse an event after attention. Both low-rank factors use nonzero
small initialization so every factor receives gradients from the first step.

No history, absent query, or exact-repeat fallback gives an exact zero
correction. Candidate permutation commutes with the operator. There is no
dataset/category/query-type branch, user-ID scoring edge, pair MLP, candidate
scalar gate, tangent projection, or halfspace projection.

## Equal-parameter reductions

Every mode owns the same `(heads, rank, dim)` factor tensors and starts from the
same state hash.

1. `multihead_coupled` (primary): each head closes its own metric loop.
2. `single_wide_coupled`: all factors form one wide coupled metric; tests
   whether multi-head nonlinear selection pays rent.
3. `selection_only`: learned maps choose events, while raw LM coordinates carry
   values and readout; tests whether coupling beyond semantic values matters.
4. `shifted_loop`: head `r` selects events while head `r+1` carries values and
   readout; identical maps and compute, but the loop is deliberately broken.

C38 one-head unprojected transport remains the external strong control in any
real-data gate. C39 global-only is an untied-family reference, not a target that
may be beaten merely through capacity.

## Novelty and authorization

Generic Q/K/V sharing is not the claim. Projection sharing, metric learning,
tied attention, Hopfield retrieval, residual adapters, and multi-head attention
all have direct precedents. C40's narrow falsifiable object is closure of the
entire history-evidence loop through the ranker's candidate readout, tested
against a same-parameter loop permutation. Novelty remains uncertain until the
loop beats both exact reductions and C38.

Only the data-free design gate is authorized. A pass permits freezing a new
train-internal protocol; it does not open C39 reserve, dev/test, or qrels.
