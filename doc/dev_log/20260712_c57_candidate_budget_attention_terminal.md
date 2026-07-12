# C57 candidate-budget attention terminal

C57 moved the softmax axis itself: each ordered history event allocated a
finite multi-head budget over actual candidates and a NULL slot.  It bound a
no-NULL Slot Attention reduction, ordinary history-axis attention, pooled
history, and raw candidate controls.  The mechanism is architecture-level and
dataset-agnostic, but the nearest-neighbor audit explicitly records that Slot
Attention already establishes the normalization-axis idea.

The ensemble repaired C56's rank-inactivity and passed all three load-bearing
activity thresholds.  Across seeds, however, the primary ranged from exact
zero to large corrections; one seed alone supplied most ensemble activity,
and multiple primary/control losses worsened.  The gate therefore failed
before holdout labels.  This is progress in localization, not a positive
architecture result: candidate-axis competition prevents a deterministic
common-mode shortcut but the learned bilinear value/readout still has an
unidentified sign/scale gauge.

The next bounded test removes that learned gauge and scores the candidate-axis
allocation with a fixed frozen-LM semantic direction and common request scale.
It must beat no-NULL Slot, history-axis, pooled, and raw semantic reductions;
otherwise this family closes.
