# C2 Gate Amendment

Date: 2026-07-08

Status: approved by user decision on 2026-07-08. This is a post-hoc amendment
to the frozen C2 gate for KuaiSearch fixed candidate pools. The original
B1-vs-B0a failure is retained as a dataset property in
`reports/pps_c2_baseline_gate.json`.

Authorization source: user message accepting the amendment and confirming the
four flagged top-5 review cases as documented lexical limitations rather than
instrument bugs.

## Original Gate

Doc 11 C2 baseline credibility gate says:

> B1 BM25 significant > B0a popularity; otherwise the query field or tokenizer
> has a bug.

Observed result:

- B1 run: `20260708_kuaisearch_b1_bm25_globalidf_exact10_dev`.
- B0a run: `20260708_kuaisearch_b0a_popularity_dev`.
- NDCG@10 delta: +0.004119.
- 95% CI: [-0.001230, +0.009752].
- Significant: no.

## Diagnostic Evidence

Evidence report: `reports/pps_c2_b1_diagnostics.json`.

- Shuffled-query canary passed: original query BM25 scores were higher than
  shuffled-query scores on 98.5% of dev requests; mean delta +26.7293, 95% CI
  [+26.3462, +27.1263].
- Candidate pool vs random catalog passed: actual candidates scored above a
  100,000-item random catalog reservoir on 98.8% of dev requests; mean delta
  +26.7390, 95% CI [+26.3443, +27.1404].
- Relevance-table lexical signal passed: rel=3 over rel=0 true-query-advantage
  AUC 0.6721, 95% CI [0.6644, 0.6809]. Same-query rel3-vs-rel0 pairwise is
  low support because only 4 queries contain both labels.
- B0a audit passed: popularity stats exactly match `records_train`, train max
  ts 388532 < dev min ts 388543, and dev candidate-order popularity Spearman is
  -0.0018.

Interpretation: B1 BM25 is mechanically responsive to query text, and the fixed
candidate pools are already query-conditioned. The failed dominance relation
therefore does not by itself prove a query field or tokenizer bug.

## Approved Revised Gate

For KuaiSearch fixed candidate pools, replace the single dominance rule
`B1 BM25 significantly > B0a popularity` with this query-only sanity suite:

1. Shuffled-query BM25 canary passes: original query scores fixed candidates
   significantly higher than shuffled query scores.
2. Relevance-table lexical signal passes: rel=3 pairs have higher true-query
   BM25 advantage than rel=0 pairs, or the report states why the official
   relevance table has insufficient same-query support.
3. Candidate-pool query conditioning is documented: actual candidates score
   significantly above random catalog items.
4. B0a audit passes: popularity stats are train-only, and candidate-order
   popularity correlation is reported.
5. B1/B2z use the identical document template and have nondegenerate score
   coverage.
6. Manual top-5 review for 20 sampled requests is completed and retained.

The original B1-vs-B0a result remains reported as a dataset property: popularity
is competitive inside an already query-filtered candidate set.

## Top-5 Review Confirmation

`reports/b1_bm25_top5_review.md` is confirmed passed under the revised C2
sanity suite:

- Direct pass: 16 / 20.
- Confirmed pass with documented lexical limitations: 4 / 20.
- Flagged request classes:
  `ks_472f62b7df908af764cd3512`,
  `ks_7cff314f1557d17102f2665a`,
  `ks_9095ea4a4cb8e31ccad3822b`,
  `ks_c5de6c84ba5da740b7a60ce6`.

The flagged cases are accepted as documented lexical limitation classes:

- complementary-item intent;
- question-like query with an unverified attribute;
- single-character query tokenizer mismatch;
- partial attribute/geographic match.

## Rationale

The original rule was designed to catch bugs where the query field, tokenizer,
or item text join is broken. In a fixed candidate set already produced by query
recall, lexical query-only reranking may have little marginal click signal even
when the instrument is correct. The proposed suite tests the original bug modes
directly without requiring query-only lexical dominance over popularity.

## Paper Caveat

Any later C0/C2/C3 write-up should state that KuaiSearch fixed candidates appear
query-filtered by log-internal diagnostics. The resulting task is better framed
as preference ranking within query-relevant candidates, not first-stage lexical
retrieval. M4/M5 wording should avoid claiming that query-only methods fail only
in high-entropy buckets; the stronger finding may be that query signal is
saturated throughout the fixed candidate set.
