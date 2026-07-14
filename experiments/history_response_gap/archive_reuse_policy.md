# Selective archive reuse

The archive is evidence and a local implementation reference, not an active
source tree. Do not copy an old round, config, threshold, report, or proposed
system back into the current workspace.

## Assets that may be selectively reused

- standardized-record field parsing and history-causality checks, after they
  are rewritten against the current dataset-independent contract;
- candidate-manifest/hash and shared-evaluator utilities;
- wrong-user donor matching and collision enumeration algorithms, after a new
  E0 field/power review and new unit tests;
- old true/null score bundles or checkpoints for the non-claim E-1
  instrumentation pilot only;
- upstream baseline patches whose provenance and current boundary remain
  valid.

## Assets that must remain archived

- C01--C80 model trees and all successor logic;
- R0/round1--5 state machines, Motivation Briefs, thresholds and trial budgets;
- historical dev/test outcomes as tuning evidence for the new experiment;
- old Lite/constructed-query dataset admission decisions;
- candidate-local evaluators or any method code that reads evaluation qrels.

## Promotion procedure

Before restoring an implementation idea, write a one-line need in the current
E0/E1 artifact, identify the archive source path, copy only the minimal logic
through a new tracked implementation, and add a current hand-computed test.
The active file must not import from `archive/` at runtime.
