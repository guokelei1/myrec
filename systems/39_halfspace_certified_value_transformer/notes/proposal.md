# C39 halfspace-certified value proposal

## Observation

C38 changed the evidence boundary rather than merely adding another negative
result. On an independent English dataset and BGE encoder, query-attended true
history beat frozen BGE by `+0.074314` NDCG@10 and matched wrong-user history
by `+0.041688`, both with strictly positive confidence intervals. Query
attention beat uniform history by `+0.010269`. But removing tangent projection
improved NDCG@10 by `+0.011381`. Thus the transferable object is a
query-conditioned history value, while hand-constraining its pooled geometry
is harmful.

Earlier C15 showed that a linear/bilinear candidate-dependent value pulls
through aggregation into FiLM/hyperadapter form; an unrestricted nonlinear
pair value is only generic edge-conditioned message passing. C28 further
showed that a free candidate comparator has an unidentified direction/scale.
The unresolved location is therefore not another attention weight or output
gate, but a structured, direction-identifiable value law before aggregation.

## Single primitive

For head `r`, let normalized LM states be query `q`, history event `h_j`, and
candidate `c_i`. Shared projections produce

```text
q_r = Q_r q
k_jr = K_r h_j
k_ir = K_r c_i
v_jr = V_r h_j
alpha_jr = softmax_j(<q_r,k_jr>/tau)
g_r = sum_j alpha_jr v_jr
e_jr = v_jr - g_r
```

The standard unprojected history write `g = W_O concat_r(g_r)` is common to
every mode. Candidate-relative admission is continuous and set-equivariant:

```text
s_ijr = ReLU(<k_ir - mean_l k_lr, k_jr>/sqrt(d_r))
```

Let `W_Or` be the head slice of the output projection. The local gradient of
the dot-product candidate readout with respect to a head value is

```text
n_ir = W_Or^T c_i.
```

C39's defining eventwise operator is the Euclidean projection onto the closed
score-compatible halfspace:

```text
P_ijr = argmin_u ||u-e_jr||^2  subject to <n_ir,u> >= 0
       = e_jr + ReLU(-<n_ir,e_jr>) n_ir / (||n_ir||^2 + eps).

beta_ijr = alpha_jr s_ijr / (1 + sum_t alpha_tr s_itr)
r_ir = sum_j beta_ijr P_ijr
```

The fixed `1` is a null budget, not a learned gate. If no event has positive
candidate-relative support, `r_ir` is exactly zero. Every admitted head/event
write satisfies `<c_i, W_Or P_ijr> >= 0` up to numerical tolerance before
aggregation. A candidate-local query token receives

```text
z_i = q + g + W_O concat_r(r_ir),
```

then passes through the same shared Transformer FFN/readout as every control.
The personalized logit is the difference between this factual internal state
and the same block with no history, added to the common frozen base. Therefore
empty history is an algebraic zero correction rather than a learned fallback.

This projection is nonlinear before event summation. With halfspace normal
`n`, event sets `{a,-a}` and `{0,0}` have the same ordinary pooled value, but
for `<n,a> > 0`, `P_n(a)+P_n(-a)` is nonzero whereas the zero set remains zero.
It therefore cannot be moved after ordinary pooling as a linear FiLM or
hyperadapter. Unlike a generic pair MLP, the map has no pair-specific learned
parameters: it is the unique KKT solution of the stated fidelity constraint.

## Exact contracts

- No history or masked query: correction is bitwise zero and ranking equals
  the common base.
- Exact candidate recurrence in true history: all cross-item writes are
  suppressed for that request and the common exact-item base is unchanged.
- Unsupported candidate/event/head: `s_ijr=0` gives an exact zero edge.
- Candidate permutation: candidate-local states permute exactly and all logits
  restore to caller order.
- The operator never uses dataset, category, query type, candidate position,
  or target-label attributes as a branch.

## Matched controls

All modes instantiate identical `Q/K/V/W_O`, FFN, parameter count,
initialization, optimizer, data, and request/candidate order.

1. `eventwise_halfspace` — primary, projection before event aggregation.
2. `eventwise_raw` — removes only the halfspace projection.
3. `postpool_halfspace` — pools admitted innovations first, then projects once;
   this is the aggregation-reducible nearest degeneration.
4. `ray_only` — keeps the same immediate nonnegative readout component but
   discards the score-neutral part of the projected vector; this tests whether
   an internal vector representation pays rent over a scalar-equivalent boost.
5. `global_only` — removes candidate-local values and retains the strong
   unprojected query-attended Transformer write.

## Data boundary after a design-gate pass

The initial real gate is Amazon-C4 train-internal only. It may reuse C38's
6,000 fit requests because those are training data. Its 1,200 A requests must
come exclusively from C38's 1,599 never-featured, never-scored, never-labeled
unused reserve; C38 internal-A, delayed-B, and escrow are excluded. Selection
uses request hashes and history length only. Upstream dev/test records and all
dev/test qrels remain forbidden.

A pass would still be only an Amazon train-internal mechanism result. It must
be followed by a newly frozen KuaiSearch transfer gate before C39 can be called
a proposed architecture. A failure closes this value law; no threshold,
projection strength, head count, scale, cohort, or loss rescue is allowed.
