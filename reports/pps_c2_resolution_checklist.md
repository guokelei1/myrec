# C2 Resolution Checklist

Date: 2026-07-08

Current status: C2 remains failed under the frozen rule.

## Evidence Now Available

- Original C2 failure: B1 BM25 is not significantly better than B0a Popularity
  (`delta_ndcg@10 = +0.004119`, 95% CI `[-0.001230, +0.009752]`).
- Diagnostic report: `reports/pps_c2_b1_diagnostics.json`.
  - BM25 shuffled-query canary passed.
  - Candidate pool vs random catalog passed.
  - Relevance-table lexical signal passed.
  - B0a train-only audit passed.
- Top-5 review draft: `reports/b1_bm25_top5_review.md`.
  - 16 pass.
  - 4 need human review.

## Remaining Human Decisions

1. Decide whether the explicit amendment in
   `reports/pps_c2_gate_amendment.md` is approved.
2. Confirm or override the assistant top-5 review draft.
3. If the amendment is approved and the review passes, reissue C2 with the
   revised gate. Do not delete the original failed B1-vs-B0a result.
4. If C2 is reissued as passed, rerun or reissue M3 as protocol-valid and
   clearly separate it from the exploratory M3 generated while C2 was failed.

## If Amendment Is Rejected

Keep C2 failed. The current M3 report remains exploratory, and Phase 2 should
not advance to C3. Further B1 tuning should not continue without reopening the
dev-evaluation budget.

## If Top-5 Review Fails

Keep C2 failed unless the failure is narrowed to a documented query class and
the revised gate explicitly allows that class. The current flagged cases include
complementary-item intent, question-like intent, one-character all-zero BM25
scoring, and a partly missed geographic constraint.
