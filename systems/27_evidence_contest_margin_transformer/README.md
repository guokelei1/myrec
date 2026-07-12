# C27 — Evidence-Contest Margin Transformer

C27 tests whether C26 failed because personalized evidence was written through
an independent additive scalar head.  It keeps token-level
query/candidate/history representation, but replaces that head with a
permutation-equivariant, antisymmetric pairwise contest readout.  Candidate
scores are induced from evidence-conditioned pair probabilities rather than a
bounded residual added after ranking.

This is a train-internal signal/architecture gate, not a novelty claim.
Pairwise rankers, differentiable sorting, candidate-set Transformers, and
user-dynamic self-attention are established prior art.  C27 is eligible only if
its constrained evidence contest beats matched candidate-only, generic
contest, and additive-node controls.  Dev/test remain unauthorized.
