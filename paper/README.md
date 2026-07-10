# Paper

Manuscript source, bibliography, and small manually selected assets.

Large generated figures or temporary export bundles should stay under
`artifacts/` until they are selected for the paper.

Current drafts:

- `introduction_and_motivation.md` - prose through the design transition;
- `introduction_motivation_sentence_plan.md` - sentence-level scaffold.

Both now follow the C5-R3 component audit. M3/M4 and C5-R2 remain failed
diagnostics; D1 query attention is negative; exact repeat-item memory is the
supported narrow signal; and C5-R3 item-only (mean 0.3454) is the static
baseline-to-beat. C5-R3 `TERMINAL_FAIL` closes only the preregistered doc/23
item/category recovery ladder and validates neither of its candidate
primitives. The motivation is complete: the evidence supports a bounded
transition to architecture/protocol formulation around unequal history-evidence
fidelity, while implementation and training remain gated by a new
design-specific pre-outcome falsifier. The current boundary is documented in
`reports/pps_c5_insight_audit.json` and
`reports/pps_c5r3_candidate_history_alignment.json`.
