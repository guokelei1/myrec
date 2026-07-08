# 20260708 C1 and Batch 1 KuaiSearch Lite

## Commands

```bash
python scripts/run_c1_canaries.py --standardized-dir data/standardized/kuaisearch/v0_lite --split dev
python scripts/audit_c1_protocol.py --standardized-dir data/standardized/kuaisearch/v0_lite --report-path reports/pps_c1_protocol.json

python scripts/run_core_baseline.py --method b0a --standardized-dir data/standardized/kuaisearch/v0_lite --split dev
python scripts/run_core_baseline.py --method b0b --standardized-dir data/standardized/kuaisearch/v0_lite --split dev
python scripts/run_core_baseline.py --method b1 --standardized-dir data/standardized/kuaisearch/v0_lite --split dev --run-id 20260708_kuaisearch_b1_bm25_globalidf_exact10_dev --bm25-idf-scope global --bm25-exact-match-boost 10.0
python scripts/run_dense_biencoder.py --standardized-dir data/standardized/kuaisearch/v0_lite --split dev
python scripts/run_b7_grid.py --standardized-dir data/standardized/kuaisearch/v0_lite --split dev
python scripts/run_b7_grid.py --standardized-dir data/standardized/kuaisearch/v0_lite --split dev --query-run-id 20260708_kuaisearch_b2z_bge_small_zh_dev --history-run-id 20260708_kuaisearch_b0b_recent_behavior_dev --run-prefix 20260708_kuaisearch_b7_bge_dev --method-id b7_bge --config configs/baselines/b7_bge.yaml
python scripts/run_m3_oracle.py
```

## C1

C1 passed. The report is `reports/pps_c1_protocol.json`.

- Dev records: 12,229 requests, 575,609 candidates, 7,028 users.
- Test records: 12,224 requests, 590,683 candidates, 7,231 users.
- Candidate manifest SHA-256:
  `94eb667000e0d0f389d0a2a4d4730683b71c129043edbfcf627590376e9c123e`.
- Dev/test records have zero `clicked` / `purchased` candidate fields.
- Candidate hash negative control failed as expected.
- Canaries: random NDCG@10 0.2811; shuffled-label 0.2831; positive-title
  leak 0.9579.

## Batch 1

All Batch 1 methods produced `scores.jsonl`, `metrics.json`, and
`per_request_metrics.jsonl` through the shared evaluator.

| Method | Run | NDCG@10 |
|---|---|---:|
| Random | `20260708_kuaisearch_random_c1` | 0.2811 |
| B0a | `20260708_kuaisearch_b0a_popularity_dev` | 0.3013 |
| B0b | `20260708_kuaisearch_b0b_recent_behavior_dev` | 0.3139 |
| B1 | `20260708_kuaisearch_b1_bm25_globalidf_exact10_dev` | 0.3054 |
| B2z | `20260708_kuaisearch_b2z_bge_small_zh_dev` | 0.3056 |
| B7-bm25 | `20260708_kuaisearch_b7_bm25_dev_a01` | 0.3276 |
| B7-bge | `20260708_kuaisearch_b7_bge_dev_a02` | 0.3305 |

C2 failed because B1 was not significantly better than B0a. After follow-up
diagnosis, the best B1 variant is
`20260708_kuaisearch_b1_bm25_globalidf_exact10_dev`: NDCG@10 0.3054 vs B0a
0.3013, delta +0.0041, 95% CI [-0.0012, 0.0098]. Tried variants include
request-local BM25, CJK 2/3-gram BM25, exact query phrase boost, jieba,
global item-catalog IDF, and character coverage boost. B1 dev tuning budget is
effectively consumed, so further B1 tuning is stopped pending a protocol
decision.

Interpretation note: this does not directly invalidate the PPS design. It
indicates that in a query-conditioned fixed candidate set, lexical query-only
reranking has weak marginal signal over popularity. See
`reports/pps_c2_b1_issue_and_options.md`.

Follow-up B1 diagnostics were run with:

```bash
python scripts/run_b1_diagnostics.py
```

They do not read `qrels_dev/test` and do not append to `dev_eval_log`. Results:

- Shuffled-query canary passed on all 12,229 dev requests: original-query BM25
  mean score exceeded shuffled-query mean score by +26.7293, 95% CI
  [+26.3462, +27.1263], original-greater rate 0.985.
- Candidate pool vs random catalog passed: actual candidates exceeded a
  100,000-item random catalog reservoir by +26.7390, 95% CI
  [+26.3443, +27.1404], actual-greater rate 0.988.
- Relevance-table lexical check passed: rel=3 over rel=0 true-query-advantage
  AUC 0.6721, 95% CI [0.6644, 0.6809]. Same-query pairwise has low support
  because only 4 queries contain both rel=3 and rel=0.
- B0a audit passed: popularity stats exactly match `records_train`; train max
  ts 388532 is before dev min ts 388543; dev candidate-order popularity
  Spearman is -0.0018.

Conclusion: the diagnostics support "candidate pool already query-filtered"
rather than "BM25 is broken" or "B0a leaked dev/test clicks". C2 still remains
failed unless a human explicitly approves the amendment draft in
`reports/pps_c2_gate_amendment.md` and completes the manual top-5 review.

The top-5 review sheet now has an assistant review draft: 16/20 pass and 4/20
need human review. The flagged cases are not random noise; they identify query
classes where lexical matching is brittle inside the fixed candidate set:
complementary-item intent, question-like intent, one-character all-zero BM25
scoring, and a partly missed geographic constraint.

M3 was run after C2 failed and is therefore exploratory. It showed oracle
NDCG@10 0.4232 and +28.0% relative headroom over B7-bge, with split-half
headroom +28.2% / +27.9%. This should not be promoted to C3 evidence until
the C2 query-only sanity check is resolved or revised.
