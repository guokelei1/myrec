# C2 Resolution Checklist

Date: 2026-07-08

Current status: resolved. C2 is reissued as passed under the approved revised
gate in `reports/pps_c2_baseline_gate.json`.

## Evidence Now Available

- Original C2 failure: B1 BM25 is not significantly better than B0a Popularity
  (`delta_ndcg@10 = +0.004119`, 95% CI `[-0.001230, +0.009752]`).
- Diagnostic report: `reports/pps_c2_b1_diagnostics.json`.
  - BM25 shuffled-query canary passed.
  - Candidate pool vs random catalog passed.
  - Relevance-table lexical signal passed.
  - B0a train-only audit passed.
- Top-5 review: `reports/b1_bm25_top5_review.md`.
  - 16 direct pass.
  - 4 confirmed pass with documented lexical limitations.

## Human Decisions

1. Amendment approved by user decision on 2026-07-08.
2. Top-5 review confirmed passed with four documented lexical limitation
   classes.
3. C2 reissued with the revised gate. The original failed B1-vs-B0a result is
   retained.
4. M3 reissued as protocol-valid without rerun because it is a read-only oracle
   analysis and its inputs did not change. The exploratory pre-C2 report is
   preserved as `reports/pps_m3_headroom_summary_exploratory_pre_c2.json`.

## Superseded Branches

The rejection branch is closed by the user decision. Further B1 tuning remains
closed unless a new protocol decision reopens the dev-evaluation budget.

The top-5 failure branch is closed by the user decision. The flagged cases are
documented query classes: complementary-item intent, question-like intent,
single-character all-zero BM25 scoring, and a partly missed geographic
constraint.

## Follow-up Completed

B7-bm25 was rerun against the final active B1 run
`20260708_kuaisearch_b1_bm25_globalidf_exact10_dev`.

- New best run: `20260708_kuaisearch_b7_bm25_finalb1_dev_a01`.
- Best alpha: 0.1.
- NDCG@10: 0.3292.
- Compare vs B0b: +0.0153, 95% CI [0.0109, 0.0198].
- Compare B7-bge vs final-B1 B7-bm25: +0.0013, 95% CI [-0.0024, 0.0050].
