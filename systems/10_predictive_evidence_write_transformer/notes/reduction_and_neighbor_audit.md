# Pre-outcome reduction and nearest-neighbour audit

## C04-like paired-logit delta

The paired-logit control reduces each candidate to
`mean_t(log p_H(x_it)-log p_0(x_it))` and adds the bounded scalar after the base
score.  C10 differs before the ranking head: each token's signed likelihood
ratio weights its tied token embedding, then a vector residual is written into
the candidate state.  The checked witness in `tests/test_model.py` holds summed
log-ratio fixed while exchanging evidence between orthogonal token embeddings;
paired-logit is unchanged and C10 changes.  Therefore C10 is not algebraically
the paired final-logit delta unless all token embeddings collapse to one rank-1
direction, which is an explicit learned-collapse diagnostic rather than an
identity.

## Single-pass LM

The exact-capacity `single_pass` control replaces `log p_H-log p_0` by
`log p_H`, retaining the same LM, token embedding, write projection, bound, and
ranking head.  Its candidate centring cannot generally subtract the
query-conditioned token prior because that prior differs by candidate token.
Any advantage over it is attributable to the internal counterfactual
prediction comparison, not capacity.

## Ordinary centred cross-attention

The exact-parameter `centered_attention` control lets each base candidate state
attend to contextualized history values, centres the resulting candidate writes,
and applies the identical write projection and norm bound.  It can select and
write history content but never evaluates whether history improved prediction
of the candidate's own tokens.  This is the nearest standard attention control.

## Capacity-matched dual stream

The exact-parameter `dual_stream` control retains both shared LM passes but uses
the hidden-state difference dotted with candidate token embeddings.  It tests
whether any dual-stream residual suffices.  All four modes instantiate the same
parameter set; the unit contract asserts exact equality of trainable counts.

## Remaining degeneracies and stop rules

- If token embeddings or the write projection become effectively rank one, C10
  has empirically collapsed toward a scalar gate even though it is not
  algebraically forced.  A future real gate must log singular values.
- If corruptions retain the clean ranking gain, predictive likelihood is only a
  response signal, not evidence fidelity.
- If single-pass or centred attention matches C10, the counterfactual token
  comparison has paid no architectural rent.
- If exact-repeat ranking falls below item-only beyond the frozen margin, the
  semantic write has diluted the reliable path and the candidate stops.

The exact item comparator is not a second scorer: item equality creates one
coordinate in the same late evidence vector, and the shared ranking head reads
it with a nonnegative parameterization.  Holding the other internal coordinates
fixed, the checked final logit is strictly monotone in recurrence count.
