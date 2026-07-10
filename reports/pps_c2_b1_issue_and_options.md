# C2 B1 Issue Resolution

Date: 2026-07-08

## Problem

The frozen C2 gate requires B1 BM25 to be significantly better than B0a
Popularity. This did not hold on KuaiSearch Lite dev.

Current best B1 run:

- Run ID: `20260708_kuaisearch_b1_bm25_globalidf_exact10_dev`
- NDCG@10: 0.3054
- Reference: `20260708_kuaisearch_b0a_popularity_dev`
- B0a NDCG@10: 0.3013
- Delta: +0.0041
- 95% CI: [-0.0012, 0.0098]
- Significant: no

B1 tuning budget is effectively exhausted. Tried variants include request-local
BM25, CJK 2/3-gram, exact phrase boost, jieba, global item-catalog IDF, and
character coverage boost.

Resolution: the original dominance rule remains failed, but C2 has been
reissued as passed under the approved revised gate in
`reports/pps_c2_baseline_gate.json`. The failure is retained as a dataset
property, not hidden or overwritten.

## Diagnostic Evidence

Follow-up diagnostics are recorded in `reports/pps_c2_b1_diagnostics.json` and
`reports/pps_c2_b1_diagnostics.md`. They were run without the shared evaluator,
without reading `qrels_dev/test`, and without appending to `dev_eval_log`.

- Shuffled-query canary passed on all 12,229 dev requests: mean BM25 score
  delta original query minus shuffled query = +26.7293, 95% CI
  [+26.3462, +27.1263], original-greater rate 0.985.
- Candidate pool vs random catalog passed: mean BM25 score delta actual
  candidates minus 100,000-item random catalog reservoir = +26.7390, 95% CI
  [+26.3443, +27.1404], actual-greater rate 0.988.
- Relevance-table lexical check passed: rel=3 over rel=0 true-query-advantage
  AUC = 0.6721, 95% CI [0.6644, 0.6809]. Same-query rel3-vs-rel0 pairwise has
  low support because only 4 queries contain both labels.
- B0a audit passed: popularity stats exactly match `records_train`; train max
  ts 388532 is before dev min ts 388543. Dev candidate-order popularity
  Spearman is -0.0018, so there is no meaningful evidence that B0a is inflated
  by a popularity/position correlation in candidate order.

## Interpretation

This is not strong evidence that query text is unusable or that the proposed
PPS direction is wrong. The diagnostics support the data-property explanation:
the B1 instrument responds to the true query, but KuaiSearch fixed candidate
sets are already query-conditioned by recall. Within such a candidate set, most
candidates are already query-relevant, so lexical query reranking has limited
headroom. Popularity can be competitive because it ranks among already-relevant
candidates using aggregate click strength.

The generated top-5 review sheet (`reports/b1_bm25_top5_review.md`) was
confirmed by user decision on 2026-07-08: 16 direct pass and 4 pass as
documented lexical limitations. The documented classes are complementary-item
intent, question-like query with unverified attribute, single-character query
tokenizer mismatch, and partial attribute/geographic match.

## Decision

Option B was approved by user decision on 2026-07-08: amend C2 for
query-conditioned candidate sets.

The approved revised gate requires:

- Shuffled-query BM25 canary passes on dev.
- Relevance-table lexical check passes.
- Candidate-pool query conditioning is documented.
- B0a popularity audit passes.
- B1 top-5 manual review passes with documented allowed limitation classes.
- B1/B2z use the same document template and have nondegenerate score coverage.
- B2z and B1 remain logged query-only baselines.
- The C2 report explicitly states that KuaiSearch fixed candidates are already
  query-conditioned, so query-only reranking has limited marginal gain.

Consequences already applied:

- `reports/pps_c2_gate_amendment.md` is marked approved.
- `reports/pps_c2_baseline_gate.json` is reissued with `overall_status:
  passed`.
- `reports/pps_m3_headroom_summary.json` is reissued as protocol-valid; the
  exploratory pre-C2 copy is preserved separately.
- B7-bm25 was rerun against the final active B1 run, and the earlier grid was
  retired as a formal Batch 1 result.

## Rejected Alternatives

### Option A - Keep Frozen C2 As Written

Status: rejected by user decision.

Consequence:

- Stop before protocol-valid M3/C3.
- The M3 report would not have become protocol-valid.
- Next work is outside this execution task: redesign or strengthen query-only
  baseline, or revisit dataset choice.

This was the strictest protocol interpretation, but the diagnostics directly
tested the original bug modes and supported amendment rather than stopping.

### Option B - Amend C2 For Query-Conditioned Candidate Sets

Status: approved and applied.

### Option C - Replace B1 With Stronger Lexical Baseline

Try a stronger lexical stack such as Pyserini/Lucene analyzer over item text,
still scoring only fixed candidates.

Consequence:

- Requires reopening B1 tuning budget or declaring previous B1 variants invalid.
- May still fail if candidate sets are already query-conditioned.
- Adds implementation complexity without directly testing the proposed design.

Status: not pursued for this frozen slice. This is only worth doing if a later
protocol version requires literal B1 dominance again.

## Recommendation

No further B1 tuning is recommended in this Batch 1 slice. Future C3/M5 wording
should avoid the old claim that query-only methods fail specifically in high
entropy buckets. The supported statement is that the fixed pool is already
query-conditioned and the tested lexical/zero-shot query-only scorers have
limited marginal click signal; universal query saturation is not established.
