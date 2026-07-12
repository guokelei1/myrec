# C31 Authenticated Collaborative Query Transport Transformer

C31 tests one new primitive: causally authenticated history moves the query
once in a collaborative-semantic LM space, and every candidate is rescored
against that same transported query.  There is no independent per-candidate
history head.

A frozen shared BGE Transformer encodes query, history items, and candidates.
A rank-16 residual adapter is trained by full-request listwise supervision so
the LM embedding space acquires collaborative direction while retaining text
generalization.  Query-to-history semantic attention produces one authenticated
profile; adding it to the adapted query gives a single personalized query
vector.  The cosine difference before/after transport is the only non-repeat
history write.

C31 is motivated by C29/C30: strict authentication solved provenance, but
factual/null CLS, candidate-token, eventwise, pretrained-reranker, and
candidate-only cross-encoder readouts did not generalize.  A post-terminal
fit-only low-rank transport diagnostic produced stable positive estimates on
the already-open C30 A; C31 validates the frozen primitive on C29 delayed-B,
which has never been feature-materialized, scored, or labeled.

No delayed C31 role, escrow, dev, or test access is authorized.
