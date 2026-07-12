# C70 proposal — logged-choice gradient Transformer

Status: architecture formulation; dual-domain data prerequisite failed before
implementation.

## Observation

C28--C33 showed that provenance and candidate-relative mechanics can be made
load-bearing, but the learned direction is unstable. C46 and C69 then showed
that positive-only next-item/adjacent-item training mainly recovers semantics;
semantic-matched negatives remove that shortcut but leave an unstable or
wrong ranking direction.

The missing object may therefore be the alternative set under which a past
choice occurred. A clicked item alone says what the user consumed. A logged
choice episode says what the user selected over contemporaneously exposed
alternatives under a historical query.

## Primitive

For historical episode `t`, a shared LM encodes its query `q_t`, selected
item(s) `c_t+`, and all exposed candidates `C_t`. Let

```text
mu_t+ = mean selected v(c)
pi_tk = softmax_k <k(q_t), k(c_tk)>,  k over the logged slate
mu_t0 = sum_k pi_tk v(c_tk)
g_t   = normalize(mu_t+ - mu_t0)
M_t   = g_t outer k(q_t).
```

`mu_t+ - mu_t0` is the score-gradient direction of the observed choice against
its actual opportunity set. It is zero when selection carries no direction
beyond the slate expectation and is signed without a mined cross-user
negative. The internal Transformer modification is a bounded sum of these
rank-one writes to one shared attention/FFN map:

```text
W_H(q) = W_0 + sum_t a(q,q_t) beta_t M_t
q_H    = normalize(q + W_H(q) - W_0(q))
score(c) = <LM(c), q_H>.
```

The episode gate `beta_t` may depend only on event masks, action strength, and
pre-outcome bounded normalization; it may not use dataset ID, label-selected
thresholds, or the current candidate rank. Exact recurrence remains a
protected final-order contract. No history returns the query-only LM function
exactly.

The final implementation must put the write inside the end-to-end Transformer
ranking core. Materializing frozen LM states is allowed only for a minimal
signal falsifier; an embedding-plus-MLP scorer is not the proposed system.

## Binding controls

All controls use the same LM, parameters, batches, optimizer, and current
ranking path:

1. positive-only fast write, replacing `g_t` by `v(c_t+)`;
2. semantic-matched pseudo-negative write, the C69 information object;
3. random cross-user negative write;
4. history-free equal-capacity Transformer;
5. pooled logged-choice vector without a function-valued write.

Passage requires utility over the strong base and every control, true-over-
wrong history specificity, all-seed/fold signs, exact fallbacks, and the same
operator on at least two real domains with logged historical choice context.

## Stop rule

Do not construct or train C70 while only KuaiSearch supplies real historical
slates. Replacing missing Amazon/JD slates with category, title-nearest, or
cross-user pseudo-negatives would return to the already failed C69 premise and
make the proposal dataset-conditioned.
