# 2026-07-12 — C73 counterfactual query-relay terminal

C73 was the first post-C72 candidate to leave pooled BGE readout formulas and
modify token-level Transformer information flow.  It forced history to pass
through factual/NULL current-query trajectories before candidate attention and
bound late-state, pooled-relay, and factual-only reductions.

The proposal was locked before outcome (`275176a2531bd1b5d323b4306462d2ce9d65c086bb91c505a6f5990b4ee13b39`).
Three A40 runs trained all four equal-parameter modes.  Mechanics, exact
fallbacks, all corruptions, gradients, and numerical contracts passed.  The
primary beat late-state and factual-only controls in every seed, but absolute
gain was `+0.0505/+0.0012/+0.0500` against a frozen `+0.10` minimum, and its
margin over pooled relay was only `+0.0218/+0.0184/+0.0204` against `+0.025`.

The result is useful but terminal: query mediation is a plausible path, while
token-resolved counterfactual attention is not yet a stable architecture
advantage.  No repository data or label was read.  The next design question is
whether the two attention stages need a shared, identifiable relational
coordinate rather than independently learned projections; this must first be
audited against C40 metric coupling and cannot be treated as a C73 rescue.
