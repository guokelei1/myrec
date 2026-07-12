# C59 — Exact semantic candidate-budget Transformer

C59 is the single numeric-equivalence successor authorized by C58.  It keeps
the same frozen BGE token interactions, candidate+NULL budget, score rule,
controls, data, and gates.  Its only change is a permutation-stable numerical
realization: candidate-set denominators and request moments are reduced after
sorting in float64.

No optimizer or train label is used.  C59 must rerun the full label-free A0;
only a pass may expose the untouched 1,200 fit-holdout labels.  C26 A/B/escrow,
dev/test, and qrels remain closed.
