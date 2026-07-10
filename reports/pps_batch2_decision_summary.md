# PPS Batch 2 Decision Summary

> **Current supersession / 当前解释（2026-07-10）.** 下文的
> benchmark-only/no-design 表述是当时或该特定 gate 的结论。当前以
> [`doc/15_proposed_system_design_principles.md`](../doc/15_proposed_system_design_principles.md)
> 和 [`reports/pps_architecture_readiness.md`](../reports/pps_architecture_readiness.md)
> 为准：motivation complete，design formulation ready；implementation/training
> 仍由新的、design-specific pre-outcome falsifier 把关。C5-R3 FAIL 及全部数字不变。

Date: 2026-07-08

Scope: historical strong-baseline pass for KuaiSearch dev after C2 amendment
and Batch 1 reissue. No method used test data; held-out files had only been
structurally audited by C1.

Superseding note (2026-07-10): baseline numbers remain valid, but the M3
heterogeneity/headroom interpretation below is invalidated by
`reports/pps_m3_m4_random_canary_audit.json`. B8 is a history-aware B7-seeded
subset reranker, not part of the query-only control chain.
Later D1/D2/D2h/D2s strengthening then superseded B7 as the waterline:
D2s mean 0.3416 significantly exceeds the interim D2h mean 0.3352. All
"current baseline" wording below is the 2026-07-08 Batch 2 decision only.
The final C5-R3 component audit further supersedes D2s as the numeric waterline:
item-only mean is 0.3453755. C5-R3 terminates benchmark/analysis-only and does
not authorize proposed-system design.

## Decision

Batch 2 is accepted as protocol evidence, with one important boundary: B4/B5/B6 are audited local adapters, not official upstream reproductions. They are valid for diagnosing whether cheap history/full-feature adapters overturn the Batch 1 conclusion, but they must not be described as final RecBole, KuaiSearch-official, or HEM/ZAM/TEM numbers.

The practical baseline conclusion is unchanged:
`20260708_kuaisearch_b7_bge_dev_a02` remains the strongest formal baseline at
NDCG@10 = 0.3305. B8a h=50 comes closest at 0.3302 full-dev, but on the fixed
2000-request subset it is below B7-bge by -0.0019 with CI [-0.0089, 0.0050].
The defensible interpretation is that the pool is query-conditioned and the
tested B1/B2z/B3 query-only scorers add little marginal click signal. It does
not establish universal query saturation or authorize evidence routing.

## Formal Run Outcomes

| Method | Best run | NDCG@10 | Primary comparison | Decision |
|---|---|---:|---|---|
| B3 cross-encoder | `20260708_kuaisearch_b3_bge_reranker_base_zs_dev` | 0.3068 | vs B2z +0.0011, CI [-0.0031, 0.0053] | keep as query-only upper-bound check; not significant |
| B4 SASRec-style adapter | `20260708_kuaisearch_b4_sasrec_style_hashed_prior_dev_s20260709` | 0.2887 | vs B0b -0.0252, CI [-0.0306, -0.0197] | adapter negative result; official RecBole still pending |
| B5 KuaiSearch-style adapter | `20260708_kuaisearch_b5_dcn_din_style_hashed_dev_s20260709` | 0.2931 | vs B7-bge -0.0375, CI [-0.0430, -0.0317] | adapter negative result; no official alignment claim |
| B6 PPS-classic-style adapter | `20260708_kuaisearch_b6_pps_classic_style_hashed_dev_s20260709` | 0.2933 | vs B7-bge -0.0373, CI [-0.0429, -0.0316] | adapter negative result; no HEM/ZAM/TEM claim |
| B8a Qwen raw-history rerank | `20260708_kuaisearch_b8a_qwen25_7b_h50_dev` | 0.3302 | subset vs B7-bge -0.0019, CI [-0.0089, 0.0050] | keep as LLM upper-bound/cost negative result |
| B8b Qwen memory-style rerank | `20260708_kuaisearch_b8b_qwen25_7b_h50_dev` | 0.3293 | subset vs B8a h=50 -0.0053, CI [-0.0120, 0.0014] | retire as no-gain memory-style variant |

## Budget And Redline Audit

| Item | Status |
|---|---|
| Candidate manifest | all formal runs record `94eb667000e0d0f389d0a2a4d4730683b71c129043edbfcf627590376e9c123e` |
| qrels isolation | all formal run metadata record `qrels_read = false` |
| Dev logging | all dev evals appended to `reports/dev_eval_log.jsonl` |
| Shared evaluator | all NDCG/MRR/Recall/pNDCG values came from shared evaluator outputs |
| B3 budget | 1/1 zero-shot config |
| B4 budget | 7/16 dev evals, including retained weak variants; final adapter has 3 seeds |
| B5 budget | 3/16 dev evals; 3 seeds |
| B6 budget | 3/16 dev evals; 3 seeds |
| B8a budget | 3/3 history lengths on fixed subset |
| B8b budget | 3/3 history lengths on the same fixed subset |
| B8 subset | 2000 requests, `reports/b8_dev_subset_request_ids_seed20260708.txt`, SHA256 `a700e1b4e507e1b3375d3df17f6547a07c6f2711ba9703a8a7541fb74eb8f02f` |

## M3 Batch 2 Oracle

`reports/pps_batch2_m3_oracle_summary.json` re-runs the M3-style oracle over B2z, B3, B0b, B4, B5, B6, B7-bge, and all B8 variants. The oracle NDCG@10 is 0.5468, with +65.4% relative headroom over B7-bge and CI [+64.2%, +66.7%].

This is analysis-only. Because the oracle maximizes over many noisy channels,
the absolute +65.4% is selection-noise dominated. The later Random-channel
audit shows that even the three-channel +28.0% is reproduced by Random; neither
oracle is usable as qualitative heterogeneity evidence or as proof that a
learned router has a real target.

## Interpretation For The Paper

The result pattern matches the bounded C2 diagnosis: within KuaiSearch recall
candidates, the tested pure-query scorers have little marginal room. B3 barely
moves B2z. Separately, the history-aware B8 does not beat the static B7 mixture
even with a 7B instruction model reranking the top 20.

This makes the paper framing more specific. The task should be described less as "find query-relevant products" and more as "choose the right personalized evidence once the candidate pool is already query-conditioned." B7-bge is the current baseline to beat for proposed-system development. B8a can be kept as an expensive LLM upper-bound attempt, while B8b should be treated as a negative memory-style result unless a later protocol explicitly revisits it.

## Remaining Caveats

- RecBole SASRec/BERT4Rec is not complete in this environment because `recbole==1.2.1` depends on `ray<=2.6.3`, which has no Python 3.13 wheel.
- The KuaiSearch official ranking pipeline was not aligned within this pass because it expects precomputed query/title embeddings and raw user features outside the standardized blind-record interface.
- B6 is a local PPS-style fusion adapter, not an official HEM/ZAM/TEM reproduction.
- B8 full-dev numbers use B7-bge fallback outside the fixed 2000-request subset; formal B8 comparisons therefore use same-subset reports.
