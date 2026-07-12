# B5o KuaiSearch Official Summary

> **Current supersession (2026-07-13).** Baseline results remain valid, but
> C01--C80 later closed. Current architecture authorization comes only from
> [`doc/31`](../doc/31_problem_discovery_and_architecture_iteration_protocol.md)
> after an R0 strong baseline and passed Failure Card.

> **Terminal supersession / 当前解释（2026-07-11）.** 下文的
> benchmark-only/no-design 表述是当时或该特定 gate 的结论。当前以
> [`doc/15_proposed_system_design_principles.md`](../doc/15_proposed_system_design_principles.md)
> 和 [`reports/pps_architecture_readiness.md`](../reports/pps_architecture_readiness.md)
>、[`terminal closure`](../doc/dev_log/20260711_architecture_exploration_terminal_closure.md)
> 为准：motivation complete；后续 C01--C16 已关闭，未得到经过验证的架构
> primitive，也未授权 proposed-system dev/full/test evaluation。C5-R3 FAIL
> 及全部数字不变。正文的
> “current”与 design 边界仅属于原 Batch 2b / C5-R3 gate 时点。

Date: 2026-07-09

Status: accepted formal B5o dev baseline under
`official-code, proxy-aligned (last-time 10% split)`.

Current-boundary note (2026-07-10): all B5o results below remain valid historical
baseline evidence. C5-R3 item-only mean 0.3453755 is now the static waterline,
and at that gate-local stage the terminal label did not authorize system design.
That authorization label is historical. Any “current baseline” wording below is
scoped to the original Batch 2b decision.

## Boundary

- Upstream: `https://github.com/benchen4395/KuaiSearch`.
- Locked commit: `7ce0471b659112096f0aa7e892ed0aa4c972246a`.
- Implementation: official DNN/DCNv2 model and trainer classes through
  `src/myrec/baselines/kuaisearch_official_adapter.py`.
- Config: `configs/baselines/b5o_kuaisearch_din_dcnv2.yaml`.
- Artifact root: `artifacts/batch2b/b5o_stageb_standardized`.
- qrels read by method: false.
- Shared evaluator candidate manifest:
  `94eb667000e0d0f389d0a2a4d4730683b71c129043edbfcf627590376e9c123e`.

## Stage A Caveat

Stage A aligned the locked official code to the paper Table 7 metric scale under
a declared last-time 10% proxy split, not an upstream-confirmed paper split.
Therefore B5o must always be reported as
`official-code, proxy-aligned (last-time 10% split)`.

Stage A final proxy metrics:

| Model | Proxy LogLoss | Proxy AUC | Table 7 Logloss | Table 7 AUC |
|---|---:|---:|---:|---:|
| DNN | 0.160731 | 0.613133 | 0.1588 | 0.6258 |
| DCNv2 | 0.162635 | 0.616348 | 0.1603 | 0.6239 |

## Stage B Data

| Quantity | Value |
|---|---:|
| Train records | 163,717 |
| Train rank rows | 6,241,354 |
| Dev records | 12,229 |
| Dev score rows | 575,609 |
| Corpus items | 2,778,902 |
| Users | 52,133 |
| Query keys | 175,946 |

The official loader consumes click history only, capped at 20 events. Purchased
history is retained in the materialized JSONL for audit but ignored by the
locked loader.

## Frozen Seeds

| Model | Seed | Run | NDCG@10 | MRR | Recall@10 | pNDCG@10 |
|---|---:|---|---:|---:|---:|---:|
| DNN | 20260708 | `20260709_kuaisearch_b5o_dnn_dev_s20260708` | 0.3088 | 0.2850 | 0.5334 | 0.3191 |
| DNN | 20260709 | `20260709_kuaisearch_b5o_dnn_dev_s20260709` | 0.3030 | 0.2772 | 0.5291 | 0.3024 |
| DNN | 20260710 | `20260709_kuaisearch_b5o_dnn_dev_s20260710` | 0.3072 | 0.2817 | 0.5321 | 0.3164 |
| DCNv2 | 20260708 | `20260709_kuaisearch_b5o_dcnv2_dev_s20260708` | 0.3056 | 0.2855 | 0.5261 | 0.3168 |
| DCNv2 | 20260709 | `20260709_kuaisearch_b5o_dcnv2_dev_s20260709` | 0.3051 | 0.2824 | 0.5287 | 0.3150 |
| DCNv2 | 20260710 | `20260709_kuaisearch_b5o_dcnv2_dev_s20260710` | 0.3056 | 0.2842 | 0.5269 | 0.3125 |

Mean NDCG@10:

| Model | Mean | Sample std |
|---|---:|---:|
| DNN | 0.3063 | 0.0030 |
| DCNv2 | 0.3054 | 0.0002 |

Dev evaluations used: 6/16.

## Determinism

`reports/b5o_determinism_check.json` reloads the frozen DNN best checkpoint and
rescored the first 1000 dev requests without training and without reading qrels.
The request/key sequence matched and 42,968/42,968 score rows were exact
(`max_abs_score_diff = 0.0`).

## Comparisons

Best seed: `20260709_kuaisearch_b5o_dnn_dev_s20260708`.

| Comparison | Delta NDCG@10 | 95% CI | Interpretation |
|---|---:|---|---|
| vs Random | +0.0277 | [0.0224, 0.0331] | significant sanity pass |
| vs B0b | -0.0051 | [-0.0105, 0.0004] | below recent-behavior; CI crosses 0 |
| vs B7-bge | -0.0217 | [-0.0272, -0.0162] | significantly below B7-best |

## Conclusion

B5o is protocol-valid as a proxy-aligned official-code baseline, but it was not
the new baseline-to-beat. B7-bge remained the strongest formal deployable
baseline in the Batch 2b registry at that time.
