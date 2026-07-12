# C12 pre-implementation decision

Decision: **REJECT; DO NOT IMPLEMENT OR RUN.**

Candidate-prefix causal conditioning successfully fixes the algebraic flaw that
rejected C11: the vocabulary partition increment becomes candidate-specific and
survives candidate centring.  Explicit witnesses separate it from target-hidden
similarity, event-only cross-attention, and scalar paired-logit controls.

However, that surviving partition increment requires exact vocabulary
normalization for every candidate × event × predicted token.  The resulting
`Omega(C(H+1)TVd)` decoder cost is load-bearing; removing, sharing, or replacing
it with unnormalized logits recreates the hidden-similarity reduction.  A tiny
synthetic run would therefore test a mechanism that is not currently a viable
full ranking architecture.

Per the pre-implementation rule, C12 stops here.  There is no source model,
runner, GPU lock, cohort, or outcome.  Reopening requires a new fingerprint for
an exact and generic low-cost normalized decoder, with its own reduction and
complexity audit.  Dev/test/qrels and real records remain untouched.
