# Reports

Curated, paper-ready results. Tracked selectively (small files only).

## Checkpoint audit reports

Each phase gate (C0-C5) produces a JSON audit report here:

```text
reports/pps_c0_data_audit.json
reports/pps_c1_protocol.json
reports/pps_c3_motivation.json
reports/pps_c4_data_final.json
reports/pps_c5_insight.json
```

The gate must pass before advancing to the next phase.

## Current Decision Reports

| File | Role |
|---|---|
| `pps_intro_motivation_completion_20260710.md` | Motivation completion and bounded design-stage transition decision |
| `pps_intro_motivation_repository_audit_20260710.md` | Repository-wide audit plus C3-R resolution pointer |
| `pps_m3_m4_random_canary_audit.json` | Permanent construct-validity failure for original M3/M4 |
| `pps_c3_motivation.json` | Historical C3 record; positive use superseded by C3-R |
| `pps_c3r_history_identity_control.json` | Historical train-frozen matched wrong-user result |
| `pps_c5r2_temporal_symmetric_identity.json` | Historical temporal-symmetric control and failed identity gate |
| `pps_c5r3_candidate_history_alignment.json` | Frozen item/category decomposition and `TERMINAL_FAIL` for the doc/23 recovery ladder |
| `pps_c5r3_consistency_audit.json` | Independent raw-metric recompute, hash/log/registry/test/repository audit for C5-R3 |
| `pps_c5_insight_audit.json` | Current insight status: motivation complete, formulation ready, implementation/training gated |
| `pps_supervised_diagnostics_summary.json` | D1 supervised base/residual negative result |
| `pps_d2_d2h_summary.json` | Fine-tuned controls and the valid but interim D2h waterline |
| `pps_d2_score_audit.json` | Label-free D2 candidate/metadata integrity audit |
| `pps_d2h_score_audit.json` | Label-free D2h coverage and no-history fallback audit |
| `pps_d2s_summary.json` | Historical complete D2p + bundled-history reference at 0.3416 |
| `pps_d2s_score_audit.json` | Label-free D2s coverage and no-history fallback audit |
| `pps_d2s_protocol_lock_manifest.json` | D2s protocol/calibration/config/scoring/evaluation ordering proof |
| `pps_d2s_calibration_semantics_verification.json` | Exact scorer-z-score verification of the frozen D2s beta selection |
| `pps_intro_motivation_dev_eval_reconciliation.json` | Reconciliation of all 55 R1/B9/C3-R/D1/D2/D2h/D2s/C5-R2/C5-R3 evaluator invocations |
| `pps_architecture_readiness.md` | Current formulation-readiness memo; item-only 0.3454 is the static waterline |
| `pps_b9_neighbor_summary.md` | ZAM/TEM multi-seed evidence and review status |

Frozen computations are not deleted when a later audit invalidates their
interpretation. The later report must be linked from the original artifact and
from `experiments/pps_results.md`.

The C5-R3 `TERMINAL_FAIL` is scoped: it closes the preregistered item/category
recovery ladder and validates neither candidate primitive. Its negative result
still establishes the design problem that exact recurrence is reliable in the
tested bundle while uncalibrated cross-item/category transfer is not. Therefore
architecture/protocol formulation may proceed, but implementation and training
must wait for a new design-specific pre-outcome falsifier.

## What goes here vs. elsewhere

| Directory | Content | Tracked? |
|---|---|---|
| `runs/` | raw experiment output (logs, score dumps, raw metrics) | no |
| `artifacts/` | generated intermediate plots, tables, predictions | no |
| `reports/` | curated final results, audit JSONs, paper-ready tables | yes (small) |
| `paper/` | manuscript source and final selected assets | yes |

The `.gitignore` explicitly allows `reports/**/*.csv`, `*.tsv`, `*.json`,
`*.jsonl` so curated tables can be tracked.
