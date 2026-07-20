# Motivation mechanism first diagnosis — 2026-07-17

## Question

Where does the frozen LLM4Rec pipeline lose candidate-disjoint preference
transfer, and which model-level directions are most defensible to optimize?

## Change and audit

Completed the preregistered M0--M3 sequence: data/power and recoverability,
input interventions, Q2/Q3 layerwise representations and activation patches,
Q2/Q3 gradient diagnostics, and the fixed-exposure Q2 matched training control.
The closeout binds 18 aggregate artifacts, retains all registered folds and
negative controls, lists eight mechanical non-results separately, and passes
the producer-token numeric-claim audit. No source-test content was read.

## Evidence

- Visible-field signal is incomplete, but Q2 contains localized brand/category
  decodability beyond random-label controls; Q3 does not reproduce the pattern.
- Correct block-27 restoration recreates harmful full-history target margin in
  Q2 and Q3. Correct block-13 restoration moves the margin above null, but Q2's
  cross-request donor moves it farther, so this is not universal preference
  mediation.
- Final gradient conflict with the other-overlap surface is shared but narrow;
  recurrence mass dominance is model-dependent.
- Fixed-exposure surface balancing provides no credible NDCG DID and causes a
  coherent adverse target-margin DID.

## Decision

H0 and H5 remain unresolved; H1--H4 are weakened. The leading design candidate
is a query-conditioned, ID-free factorized preference state feeding an explicit
signed candidate residual alongside a query-only relevance path, with an
abstention gate and causal audit interface. Factor bottleneck and signed
residual are secondary separable candidates; a sparse router is boundary-only;
general surface balancing is deprioritized.

The design is not implemented or presented as the paper method. Attention
output, MLP branch, final-normalization decomposition, and optimizer-aware
effective-update attribution remain explicit coverage gaps.

## Next action

Stop at the authorized first-diagnosis boundary and wait for user direction.
If a new phase is authorized, preregister the smallest branch-decomposition and
factor-to-residual falsification sequence before implementing a transfer
architecture. Keep the source test closed.

Reports:

- [`../../reports/motivation_mechanism_first_diagnosis.json`](../../reports/motivation_mechanism_first_diagnosis.json)
- [`../../reports/motivation_mechanism_first_diagnosis.md`](../../reports/motivation_mechanism_first_diagnosis.md)
