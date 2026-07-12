# C59 proposal — exact symmetric realization of C58

Status: pre-outcome numeric-equivalence gate.  C58 passed every scientific A0
check but failed its frozen `2e-6` candidate-permutation tolerance because GPU
candidate-axis reductions depend slightly on storage order.  C59 changes no
architecture hypothesis or score law.

## Identical operator

For L2-normalized frozen BGE tokens, C59 retains exactly

```text
a_j = mean_query max_event_token cosine(query, event_j)
b_ij = mean_candidate max_event_token cosine(candidate_i, event_j)
ell_ij = a_j b_ij
alpha_ij = softmax_candidate+NULL(ell_ij)
m_i = sum_j omega_j alpha_ij
score_i = zscore(strong_base_i) + zscore(m_i).
```

The no-NULL candidate-axis, history-axis, pooled-history, raw-query, and wrong
history controls are identical to C58.  Empty history/query returns base and
exact recurrence returns item-only.  No coefficient, NULL value, threshold,
token rule, or cohort changes.

## Sole implementation change

Candidate+NULL softmax denominators are computed by sorting their float64
logits, subtracting the common maximum, and reducing that canonical order.
Request means and population variances likewise reduce sorted float64 feature
values before transforming the original candidate positions.  Outputs return
to float32.  Thus candidate storage permutation cannot change reduction order;
the mathematical real-valued operator is C58-equivalent.

## Gate

Four GPU shards rerun all 1,200 label-free requests, wrong histories, axis
control, determinism, reversed-candidate audits, and real fallback roles.  The
same C58 activity thresholds and `2e-6` permutation tolerance apply.  Failure
closes C59 and the labels stay closed.

Only passed A0 permits the compact holdout labels.  A1 is unchanged: primary
must have positive-CI NDCG gains over base, wrong history, no-NULL,
history-axis, pooled-history, and raw-query, with every hash fold positive.
No numeric or architectural rescue follows a utility failure.
