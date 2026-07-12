# 2026-07-12 — C58 numerical A0 terminal

C58 removed C57's learned value/output gauge and evaluated a fixed
token-semantic candidate+NULL budget on four GPUs without optimizer steps or
label reads.  All scientific activity checks passed by large margins and the
no-history/repeat contracts were exact.  However, the pre-registered
candidate-permutation tolerance failed because GPU reductions over the
candidate axis changed at most `3.0756e-5` after reversing storage order.

The promoted report is
`reports/pps_c58_semantic_candidate_budget_gate.json` (SHA-256
`e6aa1944176b21ac9fbbdc14acf282d9ae7db36ba5db1c9e6ab5f279b341e49c`).
It records `failed_label_free_mechanics_terminal`; both fit-train and
fit-holdout labels, C26 A/B/escrow, dev, test, and qrels remained closed.

Decision: do not relax the tolerance and do not inspect utility.  Authorize
only a separately locked, mathematically identical successor using sorted
float64 symmetric set reductions, then rerun the complete A0 contract.
