# C58 proposal — fixed semantic candidate budget

Status: pre-outcome formulation gate.  C57 established ensemble-level
load-bearing candidate-axis behavior but failed all-seed identifiability: its
learned value/output correction ranged from exact zero to overactive.  C58
removes that gauge rather than tuning it.

## Fixed operator

Frozen BGE contextual tokens are L2 normalized.  For query tokens `q_k`,
candidate-title tokens `c_il`, and history-event tokens `h_jm`, fixed late
interaction gives

```text
a_j  = mean_k max_m <q_k, h_jm>
b_ij = mean_l max_m <c_il, h_jm>
ell_ij = a_j b_ij.
```

`a_j` requires the event to be query-relevant; `b_ij` requires the same event
to support the candidate.  No learned projection, temperature, gate, value,
head, or sign remains.

The primary allocates each event across all candidates and a zero-logit NULL:

```text
alpha_ij = softmax_{candidate + NULL}(ell_ij)
m_i = sum_j omega_j alpha_ij
r_i = zscore_candidates(m_i)
score_i = standardized_strong_base_i + r_i.
```

`omega` is the registered normalized event weight.  Request z-scoring is the
common score-unit rule established by C55, not a tuned coefficient.  Empty
history/query returns the base exactly; exact recurrence returns the
registered item-only anchor.  Candidate order is exchangeable and no dataset,
category, rank, or query-type branch exists.

## Binding controls

- `slot_budget_no_null`: identical candidate-axis assignment without NULL;
- `history_softmax`: ordinary per-candidate softmax over events plus NULL,
  scored by expected triadic compatibility;
- `pooled_history`: event-weighted mean `ell_ij` before request z-score;
- `raw_query`: fixed query/candidate token late interaction with no history;
- `wrong_history`: frozen donor history under the primary.

Every score receives exactly one request z-score correction on top of the same
strong base.  The primary must beat every control; otherwise any effect is
generic semantic reranking, Slot assignment, or ordinary target attention.

## Staging and stop rule

Four GPU shards compute scores without labels or optimizer steps.  Candidate
hash, determinism, candidate permutation, real no-history/repeat fallbacks,
base activity, wrong-history sensitivity, and candidate-axis/history-axis
sensitivity must pass before labels open.  A0 failure closes C58 immediately.

Only passed A0 permits reading the compact 1,200 fit-holdout labels with the
shared metric.  A1 requires positive-CI NDCG gains over base, wrong history,
and all four controls, with every fixed hash fold positive.  There is no
temperature, correction coefficient, NULL bias, interaction variant, token
pooling, subset, or domain rescue.  Failure closes fixed semantic
candidate-budget attention; success would authorize a fresh dual-domain
trainable architecture proposal, not a novelty claim by itself.
