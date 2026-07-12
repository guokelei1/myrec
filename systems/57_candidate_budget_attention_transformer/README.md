# C57 — Candidate-budget attention Transformer

C57 moves candidate competition into the attention normalization that forms
history evidence.  Each ordered history event allocates a finite multi-head
evidence budget across the current candidate set and a NULL sink; candidates
cannot independently receive the same complete history value.

This is an architecture-level successor to C56's candidate-common carrier
failure, but global novelty is not claimed: the normalization axis is closely
related to Slot Attention.  The primary must beat an explicit no-NULL Slot
Attention reduction, ordinary history-axis target attention, pooled history,
and a history-free raw candidate control.

The first gate reuses C56's label-blind fit split and hash-verified frozen BGE
contextual tokens.  It may read only the already opened 4,800 fit-train labels
after the execution lock.  The 1,200 holdout labels remain closed until the
label-free mechanics gate passes; C26 A/B/escrow, dev/test, and qrels remain
closed throughout.
