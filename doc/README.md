# doc

Research notes, design constraints, direction decisions, and experiment plans.
Files here are intended to be tracked; the 2026-07-10 audit found recent doc/15,
doc/16, and dev-log work still untracked and awaiting an explicit evidence
commit.

Current stage: motivation is complete. C5-R3 `TERMINAL_FAIL` closes only the
preregistered doc/23 item/category recovery ladder and validates neither of its
two candidate primitives. The observed contrast between reliable exact
recurrence and unreliable uncalibrated cross-item/category transfer authorizes
architecture/protocol formulation; implementation and training remain gated by
a new design-specific pre-outcome falsifier.

## Key Documents

| File | Role |
|---|---|
| `07_paper_design_constraints.md` | Tier 1 rules governing all experiments |
| `10_direction_decision.md` | Final direction: PPS on KuaiSearch (main), Amazon-C4 (secondary), JDsearch (anchor) |
| `11_experiment_and_dataset_plan.md` | Full 6-phase experiment plan with checkpoints C0-C5 |
| `12_experiment_execution_protocol.md` | Environment isolation, GPU scheduling, run metadata, and evaluation boundaries |
| `13_baseline_implementation_plan.md` | Per-baseline implementation, scoring, and acceptance plan |
| `14_official_baseline_plan.md` | Official-baseline alignment, budgets, and stop-loss rules |
| `15_proposed_system_design_principles.md` | Current formulation principles and pre-implementation falsification boundary |
| `16_next_round_c3_router_neighbor_plan.md` | Historical M4/R1/B9 execution protocol and outcome |
| `17_intro_motivation_repair_protocol.md` | Historical train-frozen matched-history protocol |
| `18_supervised_motivation_diagnostics_protocol.md` | D1 supervised base and history-residual diagnostics |
| `19_finetuned_nonpersonalized_control_protocol.md` | D2 fine-tuned text and text/popularity control |
| `20_d2h_static_history_waterline_protocol.md` | D2h corrected static waterline and matched-history reissue |
| `21_d2s_static_full_waterline_protocol.md` | Post-result fairness repair combining complete D2p with causal history |
| `22_c5r_temporal_symmetric_identity_protocol.md` | Historical prequential freshness-matched C5-R2 protocol and failed identity gate |
| `23_c5r3_candidate_history_alignment_protocol.md` | Frozen item/category alignment gate; `TERMINAL_FAIL` scoped to its recovery ladder |

## Subdirectories

- `dev_log/` - concise development logs and decisions (chronological).
- `baseline_notes/` - per-baseline setup notes, gotchas, and reproductions.
- `review_prompts/` - bounded prompts for independent repository audits.
