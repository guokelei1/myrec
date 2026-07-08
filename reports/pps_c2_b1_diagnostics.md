# C2 B1 BM25 Diagnostics

Date: 2026-07-08T13:48:30.979679+00:00

## Summary

Diagnostics support the data-property explanation: BM25 responds to the true query, the fixed candidate pools are already much more query-relevant than random catalog items, and B0a statistics match train-only records.

This report does not change C2 status by itself. It is evidence for an explicit gate amendment only.

## Results

- Shuffled-query canary: `passed`; mean delta 26.729338; CI95 [26.346233, 27.126286]; left-greater rate 0.985.
- Candidate pool vs random catalog: `passed`; mean delta 26.738988; CI95 [26.344306, 27.140377]; left-greater rate 0.988.
- Relevance rel=3 vs rel=0 pairwise: `passed`; accuracy 0.6721; AUC CI95 [0.6644, 0.6809]; same-query pairwise `low_support`.
- B0a train-only audit: `passed`; stats exact match True; train_before_dev True.

## Caveats

- The random catalog pool is `data/standardized/kuaisearch/v0_lite/item_catalog.jsonl`, not the full raw item table.
- The relevance-table check uses independent relevance labels, not dev/test click qrels.
- Candidate-order popularity correlation can indicate online-ranking or position bias, but it is not by itself a train-window leakage finding.
