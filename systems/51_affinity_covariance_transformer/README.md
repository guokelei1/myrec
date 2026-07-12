# C51 — Affinity-Covariance Transformer

C51 tests whether a candidate is supported by the same *variation across user
history events* that makes those events relevant to the query.  Its fixed
operator is the centered covariance between query-history and
candidate-history affinity profiles.  The current gate uses only exposed
C47-A and cannot authorize fresh data or training.
