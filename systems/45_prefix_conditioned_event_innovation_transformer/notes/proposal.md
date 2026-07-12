# C45 proposal — Prefix-Conditioned Event Innovation Transformer

Status: pre-outcome design formulation. No C45 model outcome, repository
record, label, qrel, dev, or test data has been observed.

## Observation → architecture consequence → falsification

**Observation.** C43 showed that a request-global raw-history interaction can
weakly improve a KuaiSearch base while failing to distinguish true from
matched wrong-user history. A post-terminal diagnostic retained only strictly
prequentially authenticated events and restored a small positive true-minus-
wrong direction, but the interval crossed zero. C44 then ruled out changing
only candidate-axis normalization, NULL mass, or vector-versus-logit output.
The unresolved object is therefore the history-event representation itself.

**Architecture consequence.** A history item is not itself user evidence. Its
evidence is the change it causes to a state already formed by that user's
strict prefix. For event `h_t`, one shared recurrent Transformer transition is
evaluated twice from the identical factual prefix state:

```text
m_t       = F_theta(m_{t-1}, h_t, position_t)
m_t^NULL  = F_theta(m_{t-1}, NULL, position_t)
e_t       = LN(m_t - m_t^NULL).
```

Only `m_t` propagates. The NULL branch is a local counterfactual and cannot
invent a second user trajectory. The sequence `(e_1,...,e_H)` is the only
personalized value stream visible to a query-candidate read Transformer. Raw
history states have no bypass to the ranking logit. Empty history and absent
query skip the personalized path exactly; repeat-present requests retain the
registered item-only final fallback.

**Falsification.** The primitive is rejected if, under identical parameters,
initialization, training data, optimizer, and compute, it does not beat:

1. `ordinary_delta`: `LN(m_t-m_{t-1})`;
2. `factual_state`: `LN(m_t)`;
3. `raw_event`: `LN(W_h h_t)`.

It is also rejected if wrong-user histories or nontrivial event permutations
retain more than the frozen fraction of its clean gain, or if its exact
no-history/query-absent/repeat/candidate-permutation contracts fail. Synthetic
passage authorizes only a separately frozen train-internal signal gate.

## Why this is not a dataset recipe

The operator consumes only the common `(query, ordered strictly-prior history,
candidate, evidence masks)` interface. It has no dataset ID, category branch,
query-type rule, exposed-slate requirement, user-ID table, or handcrafted
semantic threshold. The same event/NULL transition is defined when item text
is present, missing, or replaced by another registered item representation.

## Predicted failure modes

- The NULL subtraction may collapse to an ordinary residual because the
  transition learns no state-only evolution.
- A factual state may already contain all useful information, making explicit
  innovation tokens unnecessary.
- Wrong histories may form internally coherent alternative users and retain
  ranking utility.
- The synthetic inductive bias may pass while the real history interface lacks
  enough non-repeat personalized signal.
- Recursive execution may be too slow relative to a causal Transformer even
  if accuracy improves.

Any of these failures closes C45 without threshold, width, step, loss, or
cohort rescue.
