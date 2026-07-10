# 2026-07-10 C3 M5 Entropy Adjudication

Superseding note: this adjudication remains the correct handling of the frozen
M5-E1 failure, but it does not address the later M3/M4 Random-channel construct
failure. The historical C3 pass is therefore paused as positive motivation
evidence; see `reports/pps_m3_m4_random_canary_audit.json`.

Context: Step 1 of `doc/16_next_round_c3_router_neighbor_plan.md` initially
failed the frozen M5-E1 entropy-slice check:

- Frozen E1 proxy: train cross-user click entropy should stratify the
  query-only gap to oracle.
- Observed: high-entropy gap `0.1156`, low-entropy gap `0.1177`.
- Result: direction did not hold, so the frozen entropy proxy remains failed.

User adjudication on 2026-07-10 invoked the pre-registered doc/11 C3 action:
"direction reversed -> rewrite insight wording and re-review." The reviewed
wording is:

> Query-only failure is supported by direct M3 per-request oracle slicing, not
> by the single-variable entropy bucket proxy.

Evidence retained:

- Direct E1 evidence from `reports/pps_m3_bidirectional_slice.json`: history is
  oracle-optimal on 35.1% of requests, and query-only loses 57.1% on that slice.
- M4 remains passed: 5-fold LR macro OvR AUC `0.6688`; canaries passed.
- M5-E2 remains passed: low history-candidate overlap weakens B0b.
- M5-E1 entropy proxy remains a negative result and must not be rewritten as
  passed by changing relative gap, bucket boundaries, or entropy definition.

Post-hoc diagnostics were authorized as exploratory only:

- dev query-in-train coverage: 42.5% (`5203/12229` requests).
- high-entropy bucket is harder overall: oracle mean `0.3891` vs low `0.4408`.
- history-optimal rate by entropy bucket is nearly flat: low `33.9%`, medium
  `35.2%`, high `36.1%`.
- Consensus Law warning: Spearman rho between entropy and
  `oracle - query_b2z` is `-0.0110`, below the doc/11 `<0.2` falsification band.

Decision:

- `reports/pps_c3_motivation.json` now records both
  `frozen_gate_status_before_adjudication = failed` and final
  `status = passed` after the doc/11 rewrite-and-review action.
- Step 2 and Step 3 of doc/16 are unlocked.
- Step 4 paper wording must not claim that high-entropy queries need more
  personalization.
