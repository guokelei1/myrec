# 2026-07-12 — C53 strong-anchor joint-context terminal

C53 tested a known PEAR/PRM-like foundation rather than claiming novelty.  It
started from registered D2p on Kuai and frozen BGE on Amazon, used the same
two-layer directed-mask Transformer and optimizer in both domains, and bound
independent-candidate, wrong-history, base, and exact no-history controls.

The six GPU fits completed.  Amazon passed A0, while Kuai failed because two
seeds did not reduce the frozen loss statistic and wrong history changed at
most one Top-10 set per seed.  The combined A0 therefore failed and no A label
was read.  Post-terminal label-free scores showed true/wrong correction
correlations above 0.983 in every run.  The model primarily learned a generic
candidate-list reranker, not candidate-conditioned user evidence.

This is useful negative progress: merely concatenating query, history, and all
candidates leaves a large history-free shortcut, and its apparent activity is
strongly amplified by a weak upstream base.  The next mechanism must remove
that shortcut structurally rather than specialize a threshold or scale to
either dataset.
