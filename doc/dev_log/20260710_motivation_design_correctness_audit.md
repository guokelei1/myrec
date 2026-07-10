# Motivation And Design-Principle Correctness Audit

Superseding note: this earlier wording audit caught tie and identity issues but
did not test a Random-channel oracle. The later repository-wide audit found a
larger construct-validity failure; see
`reports/pps_m3_m4_random_canary_audit.json`.
The later D1/D2/D2h/D2s strengthening further replaces B7 and interim D2h with
D2s mean 0.3416 as the binding gate; B7 wording below is historical.

Date: 2026-07-10

Scope: read-only audit of `paper/introduction_and_motivation.md` and
`doc/15_proposed_system_design_principles.md`, followed by wording corrections
during doc/16 Step 4. No test file was read and no new dev evaluation was run.

## Findings

1. B8a had been placed in a query-only strength chain, but its card shows that
   it consumes query, raw history, candidates, and B7-bge base scores, reranks
   only a fixed 2,000-request subset, and preserves B7 outside that subset.
2. M3 labels have a 55.97% tie rate. The 60.6%/35.1%/4.3% choice distribution
   follows the frozen query-history-static tie order and cannot be described as
   strict channel preference or a query-favorable majority.
3. M3 split-half agreement shows aggregate sampling stability after each
   request's maximum is already taken. It does not bound winner's-curse or
   selection-noise inflation from the per-request argmax.
4. M3 establishes channel-level heterogeneity, not the event-level claim that
   a query identifies which history interactions are evidence or noise.
5. M4 passes its frozen gate at 0.6688 macro OvR AUC, but history/query AUC is
   only 0.5968/0.6168, static AUC is 0.7928, and labels are tie-heavy. R1b then
   scores 0.3072 with recovery ratio -0.2521. The evidence supports modest
   feature information, not reliable cheap routing.
6. Requiring a Transformer/LM is a project-scope decision, not a conclusion
   implied by M3/M4. Likewise, statistical parity with a nearest neighbor
   removes a performance-advantage claim but does not alone decide mechanism
   novelty; reducibility and ablations are separate tests.
7. Final dev-log reconciliation found two evaluator entries for each R1 run.
   The first triggered the frozen doc/16 §5.2 low-recovery branch and the second
   is its single allowed coupled recheck; aggregate metrics are identical, but
   the overwritten first score snapshots prevent a byte-identity claim.

## Corrections

- Added reproducible `reports/pps_m3_tie_aware_audit.json`: unique winners are
  history 28.86%, query 10.86%, static 4.32%; static is strictly below at least
  one alternative on 40.14% of requests.
- Reframed B8 as a history-aware subset reranker and added formal B9 ZAM/TEM
  evidence with all identity/cold-product caveats.
- Replaced majority/error-prevalence claims with tie-aware subset claims and
  described the +28.0% oracle only as a diagnostic upper bound.
- Split the design argument into established channel-level evidence and an
  event-level design hypothesis requiring query/history masking, event
  permutation, and matched-capacity controls.
- Recorded that B7-bge 0.3305 numerically dominates R1b 0.3072; both compares
  remain required; B7 was the binding gate at the time of this audit.
- Clarified that post-hoc fixed-score routing is prohibited while jointly
  trained internal gating remains eligible for a single end-to-end primitive.
- Retained both R1 evaluator entries and recorded the one-recheck timeline in
  `reports/pps_r1_dev_eval_reconciliation.json`; no entry was removed.

These changes alter no frozen metric, threshold, run, or result branch.
