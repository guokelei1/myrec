# C78 proposal — Query-Filtered Set-Interaction Transformer

Status: pre-outcome; architecture update 2/3 after C76.  No C78 outcome or
repository input has been observed.

## Observation → architecture consequence → falsification

**Observation.** C77 showed that frozen query-side candidate-token admission
is sufficient to block a candidate nuisance: its simple
`query_candidate_filter` control reached perfect clean/wrong/query-mask
behavior.  Both that control and the triadic primary failed event permutation,
while Amazon token HSO found no load-bearing history order.  The remaining
misalignment is not admission complexity; it is treating an unordered
preference set as an absolute token sequence.

**Architecture consequence.** Frozen pretrained anchors admit candidate
WordPieces with positive current-query support.  All history WordPieces remain
available, so candidate-history relevance is learned rather than oracle
filtered.  Query/readout, admitted candidate tokens, and history tokens enter a
trainable bidirectional interaction Transformer.  Query/candidate positions
remain ordinary; each history event reuses the same within-item position IDs:

```text
position(history_event_j, token_k) = history_origin + k.
```

No event-index embedding exists.  Therefore permuting complete history events
permutes their internal states at every layer and leaves the candidate logit
exactly invariant.  Direct C-H and H-C edges remain dense.  The same graph with
H cross-edges cut provides a paired safety reference; its logit difference is
added to a protected query-candidate base.  Empty history/query returns the
base and exact recurrence retains item-only.

The primitive is the **event-set-equivariant raw-token interaction
Transformer**.  Query filtering is an inherited boundary, not a new claim.
There is no dataset/category/query-type branch, event-order feature, learned
admission threshold, pooled user vector, or external LLM call.

**Falsification.** C78 reuses C76's exact generator, split, nuisance, steps, and
thresholds.  The primary must have exact event-permutation invariance and beat
four equal-capacity reductions on worst(clean, shuffled) supported accuracy:

1. `positional_query_filter` — C77's tied query-candidate filter;
2. `ungated_set` — set-equivariant ordinary full-token interaction;
3. `pairwise_set` — query-free frozen C-H semantic admission;
4. `triadic_set` — C77's more restrictive triangle with the same set symmetry.

It must also pass wrong/query-mask/no-history/repeat and ordinary utility gates.
If a simpler pairwise or triadic set graph ties, C78 lacks a unique query-filter
claim and closes.  A pass authorizes one fresh real pretrained-LM probe.  No
position scheme, event width, filter, threshold, control, or generator rescue
is allowed.

## Novelty boundary

Set Transformers, permutation-equivariant attention, query token filtering,
and pretrained semantic routing are established.  C78 is initially an
`architecture-boundary candidate`, not a global novelty claim.  It can support
a paper primitive only if event-set symmetry pays measurable rent over the
positional control and query-conditioned admission beats pairwise/triadic
alternatives on fresh real data.
