# C42 Metric-Coupled Transport Confirmation

C42 is a weights-preserving confirmation, not a new architecture. It freezes
the three C41 `coupled_content` checkpoints that were trained and scored before
C41-A labels opened, then evaluates them on untouched C38 escrow. C41 matched
controls and C38 unprojected checkpoints are frozen at the same boundary.

No C42 retraining, seed selection, hyperparameter change, or checkpoint
selection is allowed. C42-A is feature/score/label unopened until the C42
proposal and execution locks authorize their respective stages. Dev/test and
qrels remain closed.

C42 is now closed at `failed_A1_terminal`. The exact frozen model replicated
positive margins over base and C38 and preserved true-over-wrong specificity
on untouched escrow, but its advantages over semantic and asymmetric routing
did not have strictly positive paired intervals. See
`notes/confirmation_outcome.md`; no Amazon rescue is authorized.
