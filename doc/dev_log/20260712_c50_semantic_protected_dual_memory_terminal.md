# 2026-07-12 — C50 semantic-protected dual memory terminal

C50 tested the only bounded continuation of C49: retain the raw KRR read and
project the prequential innovation read into its exact orthogonal complement.
Six frozen checkpoints were rescored with zero optimizer steps.  Structural
protection held to below 8.2e-8, but ranking utility did not.

On Kuai C50 reached 0.307006, below both same-checkpoint raw memory 0.308307
and C47 plain KRR 0.310208.  On Amazon it reached 0.234996, far below raw
0.274713 and C47 posterior 0.277001.  Wrong-history and clicked specificity
also failed the full gate.  The failure shows that C49's problem is not merely
parallel interference with semantic memory: its behavioral value direction is
wrong outside and inside that span.

Prediction-error values, DeltaNet, unprojected semantic-plus-error, and
orthogonally protected error are now one closed family.  A successor must use
a different observable relation rather than another projection or mixture of
these two reads.
