# C35 proposal: candidate-relative tangent surplus

Status: pre-outcome draft; immutable after proposal lock.

## Observation → consequence → falsification

**Observation.** C34 was mechanically active and candidate-specific, but
99.62%--99.78% of candidate/event tangent cosines were positive.  With multiple
events, absolute `cos>0` almost always admitted something and only 2--5 of
15,424 candidate rows abstained.  This was observed without A labels.

**Consequence.** Evidence must be relative to the alternatives already in the
query-conditioned candidate set.  For adapted unit query `q`, candidates
`c_i`, and authenticated events `h_j`:

```text
v_i = (I-qq^T)c_i
u_j = (I-qq^T)h_j
k_ij = cos(v_i,u_j)
b_j = mean_i k_ij
g_ij = ReLU(k_ij-b_j)
t_i = sum_j g_ij u_j / (1 + sum_j g_ij)
q_i^H = normalize(q+t_i)
d_i = 2[<c_i,q_i^H>-<c_i,q>].
```

Subtracting `b_j` is the one new operator.  It is candidate-permutation
equivariant, removes event-common compatibility, and gives exact zero write to
candidates that no event supports above the candidate-set mean.  Zero is not a
tuned cosine cutoff.  A rank-16 shared residual adapter is the only trained
representation change; there is no scalar head, pair MLP, category/query type,
or dataset branch.  Repeat returns item-only; missing query/history/authenticated
evidence returns D2p.

**Falsification.** Before labels, primary must retain at least 10% exact-zero
candidate rows and mixed admitted/rejected candidates in at least 50% of active
requests on untouched C35-A.  These conservative bounds were frozen after a
label-free C34-A formulation diagnostic found about 24.4%--24.8% and >93%,
respectively.  It must also differ materially from every control, respond to
wrong history, and preserve all fallbacks.  A1 requires positive-CI NDCG@10
over D2p and all controls, with every seed/fold positive.

## Matched controls and isolation

All four modes share tensors, initialization, fit, request order, loss, and
optimizer:

- `candidate_relative_surplus` — primary;
- `absolute_tangent_cone` — exact C34 admission without `b_j`;
- `candidate_axis_softmax` — generic Slot-Attention-like competition with no
  exact-zero surplus/null law;
- `global_tangent_transport` — candidate-shared transport reduction.

C35 reuses C34 fit because no further untouched 10k strict-nonrepeat fit pool
exists; this is declared, not hidden.  C34-A is excluded.  C35-A is C34
delayed-B, never feature/score/label-open; C35 delayed-B is C34 escrow.  Failure
closes the relative-surplus primitive without changing mean baseline, ReLU,
temperature, history length, or thresholds.  Only A1 passage can authorize a
new delayed-B gate; dev/test stay closed.
