# PPS Current Work Summary

> **Current supersession (2026-07-13).** This file is a 2026-07-09 snapshot,
> not the current work queue. C01--C80 is closed; current R0 authorization is
> defined by [`doc/31`](../doc/31_problem_discovery_and_architecture_iteration_protocol.md).

> **Terminal supersession / 当前解释（2026-07-11）.** 下文的
> benchmark-only/no-design 表述是当时或该特定 gate 的结论。当前以
> [`doc/15_proposed_system_design_principles.md`](../doc/15_proposed_system_design_principles.md)
> 和 [`reports/pps_architecture_readiness.md`](../reports/pps_architecture_readiness.md)
>、[`terminal closure`](../doc/dev_log/20260711_architecture_exploration_terminal_closure.md)
> 为准：motivation complete；后续 C01--C16 已关闭，未得到经过验证的架构
> primitive，也未授权 proposed-system dev/full/test evaluation。C5-R3 FAIL
> 及全部数字不变。正文中的
> `current` / `final` / `next` 均为 2026-07-09 快照或其历史 supersession。

Date: 2026-07-09

Status: historical snapshot, superseded by
`reports/pps_intro_motivation_repository_audit_20260710.md` and the
Random-channel construct-validity audit. Baseline numbers remain valid; the
router/readiness conclusions below do not.

Historical resolution recorded after this snapshot: C3-R/C5-R and C5-R2 were superseded by the finite C5-R3
component gate. Item-only mean 0.3453755 is now the binding static baseline;
category-only is nonsignificant in 3/3 seeds; primary and fallback both fail.
At that gate-local stage, motivation was recorded as benchmark/analysis-only
and proposed-system design was not authorized. See the now-superseded stage
wording in `reports/pps_intro_motivation_completion_20260710.md`.

Scope: concise status summary through Batch 2b official-baseline work. No method
used test data; C1 had only structurally audited the held-out files.

## Historical Position At This Snapshot

The project is past the early data/protocol and baseline credibility gates for
the current KuaiSearch dev workflow. The main active conclusion is that the
fixed candidate pool is already strongly query-conditioned, so the useful
research target is not first-stage lexical relevance retrieval. It is
personalized ranking inside query-relevant candidates. Evidence routing remains
a hypothesis because the original M3/M4 gate is reproduced by Random.

Batch 2b official-baseline work is complete for the current decision scope:

- B4o RecBole SASRec: completed as a formal KuaiSearch dev baseline.
- B5o KuaiSearch official ranking code: completed as a formal dev baseline, but
  only under `official-code, proxy-aligned (last-time 10% split)`.
- B6o HEM official: permanently downgraded to `alignment-not-verifiable`; no
  KuaiSearch dev evaluation was produced.

At this snapshot, the next numerical baseline was B7-bge and protocol-valid
proposed-system development was paused pending C3 construct repair and C5
adjudication.

## Key Decisions So Far

1. C2 gate amendment was accepted.

   The original rule expected BM25 to significantly beat popularity. That
   failed, but diagnostics showed BM25 was mechanically correct:
   shuffled-query canary passed, candidate-vs-random catalog separation passed,
   relevance-table lexical AUC was 0.6721, and B0a popularity was train-only.
   The bounded interpretation is that the pool is query-conditioned and the
   tested lexical/zero-shot query scorers add little marginal click signal.

2. Batch 1 and Batch 2 adapter baselines did not overturn B7-bge.

   B7-bge remains the strongest deployable formal baseline at NDCG@10 = 0.3305.
   B8a h=50 is close at 0.3302, but it does not beat B7-bge on the fixed subset.
   The Batch 2 oracle is selection-noise dominated and no longer supports a
   router-style proposed system.

3. B5o is useful but caveated.

   The official KuaiSearch ranking code was aligned on public data only under a
   proxy last-time 10% split, because the exact paper split remains unverified.
   Therefore B5o can be reported as a formal baseline, but not as a caveat-free
   official reproduction.

4. B6o is retired from the formal main-table path.

   External alignment failed, five public sources did not reveal the original
   split/checkpoint artifacts, and no deterministic reconstruction bug was
   found. Per the stop-loss rule, no further 20-epoch rerun or upstream issue is
   planned.

## Core Numbers

| Method | Current status | Main dev NDCG@10 | Notes |
|---|---:|---:|---|
| B0b recent behavior | completed | 0.3139 | strong cheap personalized signal |
| B7-bge | completed | 0.3305 | current baseline-to-beat |
| B4o RecBole SASRec | completed | 0.2976 | official RecBole run; high cold-start caveat |
| B5o DNN/DCNv2 | completed | 0.3088 | proxy-aligned official-code baseline |
| B8a Qwen h=50 | completed | 0.3302 | expensive LLM rerank; not above B7-bge |
| M3 oracle | protocol-valid analysis | 0.4232 | original oracle headroom, +28.0% rel. |
| Batch 2 oracle | analysis only | 0.5468 | multi-channel oracle; not a deployable gain estimate |
| B6o HEM | downgraded | n/a | no KuaiSearch formal dev eval |

## Batch 2b Details

B4o evidence:

- Best run: `20260709_kuaisearch_b4o_sasrec_recbole_t03_h128_dev_s20260708`.
- Best NDCG@10: 0.2976.
- Mean over frozen seeds: 0.2972 +/- 0.0004.
- It is significantly above Random, but significantly below B0b and B7-bge.

B5o evidence:

- Best run: `20260709_kuaisearch_b5o_dnn_dev_s20260708`.
- Best NDCG@10: 0.3088.
- DNN mean over frozen seeds: 0.3063 +/- 0.0030.
- DCNv2 mean over frozen seeds: 0.3054 +/- 0.0002.
- Determinism check passed on the first 1000 dev requests with exact score
  equality.
- It beats Random, is not significantly above B0b, and is significantly below
  B7-bge.

B6o evidence:

- Best external HEM MAP@100: 0.0759 vs target about 0.124.
- Best external HEM NDCG@10: 0.0932 vs target about 0.153.
- No original `query_split/`, `product_query.txt.gz`, or checkpoint artifact was
  found in checked public sources.

## Important Caveats To Preserve

- C0 data audit is passed, but history leakage evidence should be cited with
  its log-internal cross-reference caveat, not as an official per-event
  timestamp guarantee.
- C2 passed only after an approved post-hoc amendment. The original
  B1-vs-B0a non-significance remains reported as a dataset property.
- B5o must keep the exact identity label
  `official-code, proxy-aligned (last-time 10% split)`.
- B6o should not be revived without a new protocol decision.
- Oracle numbers are analysis tools. They justify headroom and heterogeneity,
  not direct deployable performance claims.

## Main Evidence Files

- `reports/pps_c2_gate_amendment.md`
- `reports/pps_batch2_decision_summary.md`
- `reports/pps_batch2b_completion_audit.md`
- `reports/pps_batch2b_b4o_summary.md`
- `reports/pps_batch2b_b5o_summary.md`
- `reports/b6o_official_alignment.md`
- `experiments/pps_results.md`
- `experiments/pps_baseline_cards.md`

## Suggested Next Step

Historical 2026-07-09 decision: use B7-bge as the formal baseline-to-beat and
move into proposed-system development. That action was first superseded by the
C5-R3 gate-local terminal decision; that no-design stage label is itself now
superseded by the current `doc/15`/`doc/24` four-candidate protocol. The original
M3 oracle must still not be reissued as headroom evidence.
