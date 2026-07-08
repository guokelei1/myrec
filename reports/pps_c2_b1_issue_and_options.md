# C2 B1 Issue and Options

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

The generated top-5 review sheet (`reports/b1_bm25_top5_review.md`) still needs
human pass/fail review to satisfy the original C2 text. It now contains an
assistant review draft: 16 pass and 4 need human review.

## Options

### Option A - Keep Frozen C2 As Written

Status: C2 remains failed.

Consequence:

- Stop before protocol-valid M3/C3.
- Current M3 report remains exploratory only.
- Next work is outside this execution task: redesign or strengthen query-only
  baseline, or revisit dataset choice.

This is the strictest protocol interpretation.

### Option B - Amend C2 For Query-Conditioned Candidate Sets

Replace the hard rule `B1 BM25 significantly > B0a popularity` with a query-only
sanity suite:

- Shuffled-query BM25 canary passes on dev: original query scores fixed
  candidates significantly higher than shuffled query scores.
- Relevance-table lexical check passes: rel=3 pairs have higher true-query
  BM25 advantage than rel=0 pairs.
- Candidate-pool query conditioning is documented: actual candidate BM25 scores
  significantly exceed random catalog item scores.
- B0a popularity audit passes: train-only stats and no meaningful
  popularity/position inflation evidence.
- B1 top-5 manual review passes on 20 sampled requests.
- B1/B2z use exactly the same document template.
- B1/B2z score coverage is nondegenerate: low all-zero/all-tie request rate.
- B2z and B1 are both logged query-only baselines, even if neither significantly
  beats popularity.
- The C2 report states that KuaiSearch fixed candidates are already
  query-conditioned, so query-only reranking is expected to have limited
  marginal gain.

Consequence:

- Requires an explicit human protocol amendment before C2 can pass.
- After amendment, M3 should be rerun or at least reissued as protocol-valid.
- B7-bm25 should be rerun against the final active B1 run if it is kept as a
  formal Batch 1 line.

This is the most pragmatic route if the goal is dataset/baseline construction
rather than proving query-only lexical dominance.

### Option C - Replace B1 With Stronger Lexical Baseline

Try a stronger lexical stack such as Pyserini/Lucene analyzer over item text,
still scoring only fixed candidates.

Consequence:

- Requires reopening B1 tuning budget or declaring previous B1 variants invalid.
- May still fail if candidate sets are already query-conditioned.
- Adds implementation complexity without directly testing the proposed design.

This is only worth doing if C2 must remain unchanged and B1 must pass literally.

## Recommendation

Use Option B, but only after explicit human approval. It preserves the spirit of
C2: ensuring the query-only baseline is sane and comparable, without requiring a
dominance relation that may not be appropriate after query-conditioned recall.

Until then, C2 remains failed and M3 remains exploratory.
