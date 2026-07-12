# C11: Eventwise Predictive-Write Transformer

Status: **pre-lock independent review pending**.  Design, minimal
implementation, controls, generator audit, and an execution-disabled runner
exist.  No GPU outcome or real/dev/test/qrels access is authorized.

## Observation → architecture consequence → falsification

Observation: C10 compared one history-pooled vocabulary distribution with a
query-only distribution.  Its training loss moved, but its non-repeat ordering
was indistinguishable from pooled/single-pass controls, while candidate/event
centred attention remained learnable.  The pooling step can discard which event
made which candidate token more predictable.

Architecture consequence: preserve the complete candidate-token × history-event
conditional predictive-gain matrix until a late candidate-specific event
Transformer.  No event pooling may occur before candidate/event interaction in
the primary path.

Falsification: on a role-balanced generator with no candidate-local shortcut,
the primary must beat a pooled-C10 control, ordinary centred attention, scalar
logit delta, and a same-capacity eventwise hidden-similarity control; protect
exact repeats; change candidate order; and lose clean gain under wrong-user,
order, and evidence-query corruptions.

## Single primitive

The same LM `F_theta` encodes the history-blind base candidate and every
query/event context:

```text
z_i^0       = F_theta([q, c_i])
p_0         = softmax(E F_theta([q]))
p_e         = softmax(E F_theta([q, h_e]))
G_i,e,t     = log p_e(x_i,t) - log p_0(x_i,t)
G'_i,e,t    = G_i,e,t - mean_i G_i,e,t
r_i,e       = mean_t tanh(G'_i,e,t) E[x_i,t]
u_i         = LateEventTransformer_phi({r_i,e + position_e})
w_i         = BoundedZeroSum_i(W u_i)
score_i     = MonotoneHead(z_i^0 + w_i, log(1 + exact_count_i))
```

`G` remains `[batch,candidate,event,token]` through `r`; history is not reduced
to a request vector first.  The late Transformer learns chronology over the
candidate-specific evidence-event sequence.  This is the only proposed
primitive, not a mixture of history scorers.

## Structural information contracts

- The base path input is exactly `[q,c_i]`; history cannot reach it.
- An all-false history mask selects the untouched base score tensor pointwise,
  providing bitwise fallback even though diagnostic branches are evaluated.
- Every candidate is processed with shared weights; candidate centring and
  per-candidate late integration commute with candidate permutation.
- The hidden write sums to zero over candidates and is norm bounded by one
  shared per-request contraction.
- Exact item-token recurrence creates one coordinate in the same late evidence
  vector.  The same ranking head reads it with `softplus(alpha)>0`, so the final
  logit is strictly increasing in that coordinate holding other coordinates
  fixed.  There is no post-head item score.
- The architecture sees only undifferentiated token IDs, masks, and positions.
  Category/attribute roles exist only inside the synthetic generator; the model
  has no token-range, dataset, category, or query-type condition.
- Token embeddings, shared LM, tied decoder, late event Transformer, write, and
  ranking head are optimized jointly by the ranking objective.

## Named components

1. weight-shared token LM;
2. eventwise predictive-gain late integrator;
3. monotone evidence ranking head.

Removing or replacing component 2 is exactly what the registered controls do.
