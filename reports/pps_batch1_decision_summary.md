# Batch 1 Decision Summary

Date: 2026-07-08

## Gate Status

- C0: passed after rejecting raw `recently_*` history and rebuilding history from prior recall-window events.
- C1: passed; shared evaluator, candidate hash check, label isolation, metric unit tests, and canaries all passed.
- C2: passed under the approved revised gate. The original B1-vs-B0a dominance
  rule failed and is retained as a dataset property; the revised query-only
  sanity suite passed.

## Main Dev Results

| Method | Run ID | NDCG@10 | Note |
|---|---|---:|---|
| Random | `20260708_kuaisearch_random_c1` | 0.2811 | C1 instrumentation |
| B0a Popularity | `20260708_kuaisearch_b0a_popularity_dev` | 0.3013 | significant vs Random |
| B0b Recent-behavior | `20260708_kuaisearch_b0b_recent_behavior_dev` | 0.3139 | significant vs Random |
| B1 BM25 | `20260708_kuaisearch_b1_bm25_globalidf_exact10_dev` | 0.3054 | not significant vs B0a |
| B2z BGE-small zero-shot | `20260708_kuaisearch_b2z_bge_small_zh_dev` | 0.3056 | not significant vs B1 |
| B7-bm25 | `20260708_kuaisearch_b7_bm25_finalb1_dev_a01` | 0.3292 | alpha=0.1; rerun on final active B1 |
| B7-bge | `20260708_kuaisearch_b7_bge_dev_a02` | 0.3305 | alpha=0.2; best static run |

The earlier B7-bm25 grid `20260708_kuaisearch_b7_bm25_dev_a00..a10` is retired
because it used an earlier B1 score run. The final-B1 replacement grid uses the
same 11 alpha points and is documented as a replacement budget line.

## M3 Protocol-Valid Readout

M3 was originally generated before C2 was reissued. Because it is a read-only
oracle analysis over unchanged per-request metric inputs, it has been reissued
as protocol-valid after the C2 amendment approval. The pre-C2 copy is preserved
at `reports/pps_m3_headroom_summary_exploratory_pre_c2.json`.

- Oracle NDCG@10: 0.4232.
- Best global method: B7-bge at 0.3305.
- Headroom: +28.0% relative.
- Bootstrap 95% CI relative: [+27.2%, +28.9%].
- Split halves: +28.2% and +27.9% relative.
- Oracle choices: B2z 60.6%, B0b 35.1%, B7-bge 4.3%.

## Decision

Phase 0, C1, Batch 1, C2, and the M3 headroom readout are complete for the
current main-track objective. Stop before Batch 2, proposed-system work,
Amazon/JD, and test evaluation as required.

The accepted C2 interpretation is:

- Shuffled-query canary passed: original query BM25 scores exceeded shuffled
  query scores on 98.5% of dev requests.
- Candidate pool vs random catalog passed: actual candidates exceeded random
  catalog items on 98.8% of dev requests.
- Relevance-table lexical signal passed: rel=3 over rel=0 true-query-advantage
  AUC 0.6721.
- B0a audit passed: stats exactly match train records, and train max ts is
  before dev min ts.
- Top-5 review confirmed: 16 direct pass, 4 pass as documented lexical
  limitations.

The original B1-vs-B0a failure should stay visible in future writing as evidence
that KuaiSearch fixed candidate pools are already query-conditioned. That weakens
query-only lexical marginal gains inside the candidate set, but supports the
paper motivation that the remaining ranking problem is closer to preference
selection than first-stage topical retrieval.

Next work is outside this frozen slice: C3 follow-up analyses beyond M3
(especially M4/M5 wording updates), Batch 2 baselines, proposed-system
development, secondary/anchor datasets, and later test evaluation.
