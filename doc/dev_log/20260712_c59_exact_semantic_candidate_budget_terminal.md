# 2026-07-12 — C59 exact mechanics pass, utility failure

C59 repaired C58's only numerical defect without changing its real-valued
operator. Eight tests and two locks preceded real scoring. Four A40 shards
then gave exact-zero candidate-permutation and determinism errors; A0 passed
all checks before the compact 1,200-request holdout labels opened.

A1 found a strong true-versus-wrong-history signal (`+0.027817` NDCG@10,
positive 95% CI in all three folds) but a much larger loss to the registered
strong base (`-0.070103`, wholly negative CI). Primary NDCG@10 was `0.507303`
versus base `0.577405`; candidate+NULL tied its no-NULL/history-axis/pooled
reductions. Thus semantic history is request-specific but not a safe
standalone ranking direction, and candidate-axis normalization contributes no
utility.

The authoritative report is
`reports/pps_c59_exact_semantic_candidate_budget_gate.json`, SHA-256
`5bd6ec8c67728d483c94b3b7c910b2a04c5a7d5d0f32b859b04aedc1769b3de4`.
C26 A/B/escrow, dev, test, and qrels remained closed. No scale/mixture/null
rescue is authorized.
