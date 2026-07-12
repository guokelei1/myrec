# D2/D2h Motivation Control Summary

> **Current supersession (2026-07-13).** The numeric controls remain historical
> evidence. C01--C80 is closed and current architecture entry requires an R0
> strong baseline and Failure Card under
> [`doc/31`](../doc/31_problem_discovery_and_architecture_iteration_protocol.md).

> **Terminal supersession / 当前解释（2026-07-11）.** 下文的
> benchmark-only/no-design 表述是当时或该特定 gate 的结论。当前以
> [`doc/15_proposed_system_design_principles.md`](../doc/15_proposed_system_design_principles.md)
> 和 [`reports/pps_architecture_readiness.md`](../reports/pps_architecture_readiness.md)
>、[`terminal closure`](../doc/dev_log/20260711_architecture_exploration_terminal_closure.md)
> 为准：motivation complete；后续 C01--C16 已关闭，未得到经过验证的架构
> primitive，也未授权 proposed-system dev/full/test evaluation。C5-R3 FAIL
> 及全部数字不变。正文为
> D2/D2h 与随后 C5-R3 gate 的历史中间记录。

Status: complete historical intermediate; D2h was superseded by the complete
D2s control in `reports/pps_d2s_summary.md` because D2h omits popularity.
Its train-frozen wrong-history identity interpretation is additionally
superseded by the C5-R2 temporal control; the numeric D2h results remain valid.
The later C5-R3 component audit makes item-only D2s (mean 0.3453755) the current
static waterline; its gate-local no-design authorization label was the decision
at that time and has since been superseded.

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
