# 2026-07-11 C30 canonical continuation terminal

C30 performed no training and changed no C29 weights.  Stable item-ID ordering
made candidate-permutation and deterministic rescoring exact, so the causal
authentication architecture completed its intended A1 evaluation boundary.

The utility result was negative: mean NDCG@10 delta over D2p was -0.000240
with a zero-crossing CI; only one of three seeds and one of three folds was
positive.  True history did not beat authenticated wrong/base, and clicked
correction direction was slightly negative.  Delayed-B remained untouched.

This separates two problems that earlier candidates conflated:

1. provenance: C29 solves true-versus-wrong event admission strongly;
2. direction: the candidate-independent scalar CLS readout and candidate-wise
   BCE do not convert admitted evidence into ranking-aligned relative logits.

The next design may retain causal admission but must replace the independent
scalar output interface with a request-level candidate-relative mechanism and
an aligned listwise objective.  Authentication thresholds, seed selection,
score-cap tuning, and subgroup filtering are closed rescue paths.
