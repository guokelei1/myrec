# D2/D2h Motivation Control Summary

> **Current supersession / 当前解释（2026-07-10）.** 下文的
> benchmark-only/no-design 表述是当时或该特定 gate 的结论。当前以
> [`doc/15_proposed_system_design_principles.md`](../doc/15_proposed_system_design_principles.md)
> 和 [`reports/pps_architecture_readiness.md`](../reports/pps_architecture_readiness.md)
> 为准：motivation complete，design formulation ready；implementation/training
> 仍由新的、design-specific pre-outcome falsifier 把关。C5-R3 FAIL 及全部数字不变。

Status: complete historical intermediate; D2h was superseded by the complete
D2s control in `reports/pps_d2s_summary.md` because D2h omits popularity.
Its train-frozen wrong-history identity interpretation is additionally
superseded by the C5-R2 temporal control; the numeric D2h results remain valid.
The later C5-R3 component audit makes item-only D2s (mean 0.3453755) the current
static waterline and terminates without design authorization.

| Control | Mean NDCG@10 | Sample SD | Three seeds |
|---|---:|---:|---|
| D2t fine-tuned text | 0.3141 | 0.0002 | 0.3140, 0.3143, 0.3140 |
| D2p text + train popularity | 0.3240 | 0.0002 | 0.3238, 0.3242, 0.3239 |
| D2h text + true causal history | 0.3352 | 0.0005 | 0.3352, 0.3357, 0.3347 |
| D2h text + matched wrong history | 0.3090 | 0.0004 | 0.3095, 0.3087, 0.3087 |

D2t significantly improves over zero-shot B2z but is statistically tied with D1q. D2p is significantly stronger than D2t/D1q/B0b, yet remains significantly below B7.

D2h exceeds B7 by +0.0046 (95% CI [+0.0012, +0.0080]) and D2p by +0.0113. D2h therefore replaced B7 as the static waterline at this historical intermediate stage.

On history-present requests, true D2h exceeds matched wrong D2h by +0.0396 on average across seeds. On the same-query donor subset the mean is +0.0315; every paired CI is positive.

On all 4,110 no-history requests, D2h and its seed-matched D2t have exactly the same NDCG@10, MRR, and Recall@10. The bounded conclusion is that a strong static query/history rule exposes aggregate correct-history value; same-query identity specificity was not restored by C5-R2.

Training/scoring did not read dev/test qrels; all metrics came from the shared evaluator. Test remains untouched.
