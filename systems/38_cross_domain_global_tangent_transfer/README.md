# C38 cross-domain global tangent transfer

C38 is a confirmatory transfer falsifier, not a new architecture claim.  C37
showed that candidate-specific barycentric residuals were structurally real
but utility-indistinguishable from the simpler candidate-shared global write.
C38 asks whether that weak surviving signal transfers from KuaiSearch/BGE-zh
to Amazon-C4/BGE-en without changing its operator or tuning it on Amazon
outcomes.

The Transformer encoder remains the end-to-end ranking state space.  A shared
low-rank adapter produces a query-attended history profile, removes its radial
component at the adapted query, transports the query in the tangent space, and
scores every fixed candidate against that transported state.  There is no
dataset, category, or query-type branch and no candidate scalar head.

Only the upstream Amazon-C4 history **train** split is eligible.  It is divided
by a frozen request-key hash into fit, internal-A, delayed-B, escrow, and an
unused reserve.  Upstream dev/test records and qrels are outside C38.  A0 is
entirely label-free; A labels may open only if every A0 invariant passes.

See `notes/proposal.md` and `notes/train_gate_protocol.md`.  Amazon-C4 C0/C1
have passed.  No GPU training is authorized until proposal and execution locks
bind the standardized manifest, candidate manifest, code, configuration, and
selection hashes.
