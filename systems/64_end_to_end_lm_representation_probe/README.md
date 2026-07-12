# C64 — End-to-End LM Representation Probe

C64 is a prerequisite, not a proposed-architecture claim.  It asks whether
unfreezing pretrained LM token layers can expose any non-repeat,
wrong-history-specific ranking signal that the many frozen-state candidates
could not use.

The first gate uses only an exposed-fit KuaiSearch split.  It compares an
adaptive history LM, an equal-capacity adaptive query-candidate LM, a frozen
history LM, wrong history, and the registered strong base.  Fresh roles, dev,
test, and qrels remain closed.
