# C63 mechanism outcome

Status: **failed terminal at data-free G0**.  No repository data or label was
read.

The finite-evidence operator behaved exactly as specified.  Per-event real plus
NULL mass error was at most `1.19e-7`; all four real slots carried at least
24.44% of total mass; memory was exactly query-independent; empty-history and
repeat error were zero; candidate-permutation error was at most `3.58e-7`; all
five modes had 307,492 parameters; and gradients reached every binding and
ranking component.

It did not learn the information object.  Primary four-interest accuracy was
`0.2773/0.2480/0.3008` against the frozen `0.75` threshold.  No nearest-control
margin passed, wrong history never reduced accuracy, and it changed at most
5.47% of complete orders.  Nuisance-minus-useful NULL mass was only
`+0.0145/-0.0006/+0.0032` rather than the required `+0.15`.  Finite allocation
therefore distributed mass evenly without discovering what it should bind or
reject.

Together, C62 and C63 close rank-loss-only query-independent latent preference
binding with standard softmax, inverted/Slot Attention competition, balanced
transport, finite stick-breaking, and pooled reductions.  More slots, a
different temperature, a learned break prior, extra steps, or real data would
be rescues of the same unsupported information object.

The next prerequisite should leave this memory family and test whether
end-to-end adaptation of pretrained LM token representations can learn any
wrong-history-specific non-repeat ranking signal.  Frozen LM states plus small
ranking Transformers have now been tested extensively; a full-backbone probe
is a representation-learnability test, not another latent-memory variant.
