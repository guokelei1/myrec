# Batch 1 Decision Summary

Date: 2026-07-08

## Gate Status

- C0: passed after rejecting raw `recently_*` history and rebuilding history from prior recall-window events.
- C1: passed; shared evaluator, candidate hash check, label isolation, metric unit tests, and canaries all passed.
- C2: failed under the frozen rule. The active B1 BM25 run is not significantly
  better than B0a Popularity. Follow-up diagnostics support a protocol
  amendment, but no amendment has been applied.

## Main Dev Results

| Method | Run ID | NDCG@10 | Note |
|---|---|---:|---|
| Random | `20260708_kuaisearch_random_c1` | 0.2811 | C1 instrumentation |
| B0a Popularity | `20260708_kuaisearch_b0a_popularity_dev` | 0.3013 | significant vs Random |
| B0b Recent-behavior | `20260708_kuaisearch_b0b_recent_behavior_dev` | 0.3139 | significant vs Random |
| B1 BM25 | `20260708_kuaisearch_b1_bm25_globalidf_exact10_dev` | 0.3054 | not significant vs B0a |
| B2z BGE-small zero-shot | `20260708_kuaisearch_b2z_bge_small_zh_dev` | 0.3056 | not significant vs B1 |
| B7-bm25 | `20260708_kuaisearch_b7_bm25_dev_a01` | 0.3276 | alpha=0.1 |
| B7-bge | `20260708_kuaisearch_b7_bge_dev_a02` | 0.3305 | alpha=0.2; best static run |

## M3 Exploratory Readout

M3 oracle was run after C2 failed, so this is exploratory rather than a protocol-valid C3 input.

- Oracle NDCG@10: 0.4232.
- Best global method: B7-bge at 0.3305.
- Headroom: +28.0% relative.
- Bootstrap 95% CI relative: [+27.2%, +28.9%].
- Split halves: +28.2% and +27.9% relative.
- Oracle choices: B2z 60.6%, B0b 35.1%, B7-bge 4.3%.

## Decision

Do not advance to C3 yet. The headroom signal is strong, but C2 blocks it
because the lexical query-only baseline did not clear the predeclared dominance
check.

Follow-up diagnostics in `reports/pps_c2_b1_diagnostics.json` support the
data-property explanation:

- Shuffled-query canary passed: original query BM25 scores exceeded shuffled
  query scores on 98.5% of dev requests.
- Candidate pool vs random catalog passed: actual candidates exceeded random
  catalog items on 98.8% of dev requests.
- Relevance-table lexical signal passed: rel=3 over rel=0 true-query-advantage
  AUC 0.6721.
- B0a audit passed: stats exactly match train records, and train max ts is
  before dev min ts.
- Top-5 review draft completed: 16 pass, 4 need human review.

Next action is a human protocol decision: either keep C2 failed as written, or
approve the explicit amendment draft in `reports/pps_c2_gate_amendment.md`.
The 20-request top-5 review draft still needs human confirmation, especially
the 4 flagged cases.

See `reports/pps_c2_b1_issue_and_options.md` for amendment options. The active
B1 variant uses global item-catalog IDF plus exact query phrase boost; it still
does not significantly beat B0a. Since this consumed the B1 dev budget, further
B1 tuning should not continue without a protocol decision.
