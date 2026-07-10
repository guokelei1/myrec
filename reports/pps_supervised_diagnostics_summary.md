# Supervised Motivation Diagnostics Summary

Status: complete under the frozen doc 18 protocol; test untouched.

## Main results

| Variant | Mean NDCG@10 | Sample SD | Seed values |
|---|---:|---:|---|
| D1q supervised query base | 0.3147 | 0.0007 | 0.3146, 0.3154, 0.3140 |
| D1m mean-history residual | 0.3145 | 0.0005 | 0.3151, 0.3141, 0.3143 |
| D1a query-attentive residual | 0.3148 | 0.0004 | 0.3151, 0.3150, 0.3143 |
| D1a matched wrong history | 0.3146 | 0.0007 | 0.3146, 0.3153, 0.3139 |

## Adjudication

D1q improves over B2z by +0.0089 (95% CI [+0.0041, +0.0136]) but remains below B7 by -0.0159 (95% CI [-0.0210, -0.0109]).

D1m and D1a each move above D1q in two seeds and below it in one. At the preselected seed, D1a-D1q is +0.0005 with CI [-0.0000, +0.0011]. D1a does not consistently exceed D1m.

Matched wrong-history rescoring shows a significant true-history advantage only at seed 20260708; the other seeds do not reproduce it. The correct conclusion is therefore that identity signal is visible but this simple train-fitted residual does not use it stably.

This is a representation/training negative result, not evidence against the locked C3-R identity effect. It prohibits claiming that query-attentive event selection is already established before proposed-system development.

## Integrity

All 12 dev evaluations use the shared evaluator and candidate hash. Training and scoring never read dev/test qrels; test remains untouched. The score audit verifies exact candidate coverage and exact D1q fallback on 4,110 no-history requests.
