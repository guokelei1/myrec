# C17 — Evidence-Ledger Margin Transformer

Status: **rejected before implementation; no model, data, GPU, or outcome was
used**.

C17 examined whether a Transformer could prevent history evidence from
collapsing into a candidate-common residual by carrying a persistent
`candidate × history-event` ledger through every layer and allowing only its
candidate-relative margin readout to affect ranking.

The pre-implementation audit closes the proposal.  A freely learned ledger is
an edge-state Transformer or generic edge-conditioned message-passing network.
A ledger tied exactly to the content path by the chain rule is an attribution
decomposition and does not change the ranker's function.  Using that
decomposition to gate the score creates an ordinary attribution gate, while an
antisymmetric margin/divergence readout returns to the already-closed C06 flow
family.  There is therefore no surviving C17 mechanism fingerprint.

Binding evidence:

- `mechanism_fingerprint.md` — proposed state and required novelty witness;
- `algebraic_reduction_audit.md` — exhaustive branch reduction;
- `nearest_neighbors.md` — primary-source and local-candidate comparison;
- `preimplementation_decision.md` — terminal decision.

No source model, runner, config, lock, checkpoint, synthetic record,
repository record, label, evaluator, or score artifact is authorized or
present in this directory.
