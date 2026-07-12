# C23 — Recurrence-Reset Survival Transformer

C23 tests one architecture-level correction to the strongest established
history signal.  Static item-only scoring treats exact recurrence as an
additive count/recency prior.  RRST instead treats the **last exact recurrence
of each candidate as a reset boundary** inside a candidate-local Transformer.
Only the trajectory after that boundary may modify the protected recurrence
ranking.

The first stage is deliberately narrow.  It asks whether post-recurrence
trajectory carries train-internal ranking value beyond the registered
item-only control.  It does not claim cross-item transfer.  A soft-anchor
extension is authorized only if this hard-anchor primitive passes its locked
gate.

Core operator, controls and stop rules are in `notes/proposal.md`,
`notes/reduction_audit.md`, and `notes/train_gate_protocol.md`.

No dev/test record, qrel, evaluator output or full-train popularity is an input
to this candidate.
