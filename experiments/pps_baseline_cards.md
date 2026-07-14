# Active baseline boundary cards

These cards describe controls available to the history-response validation.
They do not authorize an architecture or preserve the old B0–B9 leaderboard.

| Role | Candidate implementation | Information boundary | Status |
|---|---|---|---|
| sanity | popularity / source order / random | request and candidate metadata only | reusable |
| lexical | BM25 | query and candidate text | reusable |
| dense | cross-encoder / dense reranker | query and candidate text | reusable |
| sequential | RecBole or SASRec-style control | history identity plus declared fields | reusable control |
| personalized search | ProdSearch or PPS-classic adapter | query, history, candidate fields | control candidate |
| ordinary history model | E-FULL / D-FULL | full standardized record | to be frozen after E0 |
| signal witness | train-only cross-fitted predictor | same information boundary | diagnostic only |

For each frozen run, record upstream commit, local adapter status, input
fields, candidate manifest hash, qrels boundary, seed, budget, and evaluator
run ID. Historical cards and results are in the dated legacy source archive.
