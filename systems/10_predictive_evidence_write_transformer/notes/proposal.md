# C10: Predictive Evidence-Write Transformer

Status: architecture proposal and minimal synthetic gate only.  No real,
dev, test, or qrels access is authorized by this document.

## Single hypothesis

History should enter a late Transformer candidate residual only when the same
shared LM makes the candidate's own tokens more predictable under `(q,H)` than
under `q-only`.  This asks whether history contains candidate-token predictive
information, rather than whether another attention geometry can react to it.

## Internal information flow

One weight-shared Transformer `F_theta` is run under three masks:

1. `z_i^0 = F_theta([q,c_i])` is the history-blind base candidate state;
2. `p_H = softmax(E F_theta([q,H]))` predicts candidate tokens with history;
3. `p_0 = softmax(E F_theta([q,PAD_H]))` is the query-only counterfactual.

For candidate token `x_it`, the primitive is

```text
g_it = log p_H(x_it) - log p_0(x_it)
g'_it = g_it - mean_i(g_it)
v_i = mean_t tanh(g'_it) E[x_it]
w_i = BoundedZeroSum_i(W_v v_i)
z_i = z_i^0 + w_i
```

The ordinary ranking head reads `z_i`.  `w` is exactly zero-sum over candidates
and each candidate norm is below the frozen radius.  It is vector-valued and
token-specific: equal summed log-ratios with evidence assigned to different
tokens produce different hidden writes.

The first product token is the exact item token.  `log(1+count)` occupies a
reserved coordinate in the late evidence vector read by the same ranking head.
That coordinate's head weight is `softplus(alpha)`, so the final logit's partial
derivative with respect to exact recurrence is strictly positive.  It is not a
head-after-head score addition or an external fixed-score router.

## Structural contracts

- The base candidate path cannot attend to history.
- With no valid history, final scores are selected pointwise from the untouched
  base tensor, giving bitwise identity.
- Candidate permutation commutes with the whole computation.
- History can change relative scores only through the bounded zero-sum hidden
  write or the monotone exact-token channel.
- No dataset, category, query-type, or scorer branch exists.
- Synthetic category/attribute IDs exist only in the generator.  The model sees
  undifferentiated integer tokens plus masks and contains no knowledge of those
  factors or their token ranges.
- Candidate-token representations, predictor, write, and ranking head train
  jointly inside the Transformer ranker.

## Falsification prediction

If token-level conditional prediction is the missing abstraction, the frozen
synthetic task should show positive non-repeat transfer, protect exact repeats,
beat paired-logit, single-pass, ordinary centred-attention, and same-capacity
dual-stream writes, and lose its clean gain under wrong-user, order, and
evidence-query corruptions.  Failing any conjunct closes this primitive before
real training.
