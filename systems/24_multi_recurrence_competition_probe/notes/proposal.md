# C24 proposal — multi-recurrence competition signal gate

Status: pre-outcome design; no C24 delayed label opened.

## Observation → consequence → falsification

**Observation.** 14,179/25,122 repeat requests (56.44%) expose at least two
exact-recurrence candidates.  C23 changed rankings but ignored query, order and
suffix, indicating that independent anchor/count calibration is an easy
shortcut.  Registered item-only likewise computes each candidate's recurrence
strength independently and combines the resulting vector with D2p by z-score.

**Consequence to test.** Encode the query and all exact-recurrence candidate
tokens in one permutation-equivariant Transformer.  Cross-candidate attention
may condition the boost for candidate `c_i` on the competing recurrence states
`{c_j,r_j}` within the same query-conditioned candidate pool.  This is a signal
probe for listwise competition, not yet the paper primitive.

**Falsification.** On a new hash-frozen multi-repeat delayed role, the
candidate-set Transformer must beat both static item-only and an
equal-parameter diagonal-attention model.  Removing only candidate-candidate
edges from the trained primary must remove its gain.  Query masking, requests
with fewer than two repeated candidates, non-repeat and no-history must be
exact no-ops.

## Minimal model

For repeated candidate `i`, construct

```text
x_i = W_c c_i + W_q q + MLP[D2p_i, recurrence_mass_i,
                             repeat_count_i, last_recency_i, purchase_share_i].
```

A position-free Transformer encodes `[QUERY, x_1, ..., x_R]`.  Only repeated
candidate readouts receive a bounded, request-centered correction.  The score
anchor is registered item-only when `R>=2`; otherwise the output is exactly the
registered fallback.

Modes share every parameter and initialization:

- `set_attention`: candidates attend query and each other (primary);
- `independent`: each candidate attends only itself and query;
- `query_independent`: full candidate attention with query projections zeroed.

## Stage boundary

Even a pass does not establish novelty: context-aware self-attention for
learning-to-rank is known.  A pass establishes only that multi-recurrence
competition is a real information object worth a separate architecture design.
No dev/test/full training is authorized here.
