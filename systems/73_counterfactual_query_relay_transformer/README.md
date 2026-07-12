# C73 — Counterfactual Query-Relay Transformer

C73 tests one internal Transformer primitive: history may influence a
candidate only after it has changed the current query-token trajectory, and
the candidate block receives only the shared-attention difference between the
factual and structurally NULL query trajectories.

The first stage is a data-free, GPU-trained design falsifier.  Passing it
authorizes a separately frozen exposed-train/internal-label-free probe with a
pretrained LM; it does not authorize dev, test, qrels, or a result claim.

See `notes/proposal.md`, `notes/design_gate_protocol.md`, and
`notes/nearest_neighbors.md`.
