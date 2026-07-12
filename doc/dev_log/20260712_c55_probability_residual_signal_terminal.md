# 2026-07-12 — C55 probability-residual signal terminal

C55 tested whether C54 failed because ordinary listwise training encouraged
base reproduction.  It standardized base logits identically in Kuai and
Amazon and trained directly on the exact base probability error.  The
history-carrier model did not beat wrong history in Kuai and lost to an
equal-capacity history-free raw model in Amazon.  No non-fit role was opened.

This separates two effects.  Amazon's earlier structural activity was largely
an incompatible base/correction scale artifact.  After fixing units, frozen
pooled representations still lack a stable user-specific residual direction.
The next search location is token representation formation, not another
pooled readout, score scale, or supervised loss.
