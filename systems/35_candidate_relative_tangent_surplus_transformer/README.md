# C35 Candidate-Relative Tangent-Surplus Transformer

C35 changes the candidate axis of history attention.  C34 showed that absolute
positive query-centred cosine is nearly always satisfied when several events
are present.  C35 admits event `j` for candidate `i` only when its tangent
compatibility exceeds that event's mean compatibility across the current
candidate set.  Thus history must discriminate a candidate from its
contemporaneous alternatives, not merely resemble it in absolute terms.

The model remains dataset-agnostic and parameter-matched to the absolute C34
cone, candidate-axis softmax, and candidate-shared global transport.  C35 uses
C34's fixed fit but promotes C34's never-materialized delayed-B to C35-A and
C34 escrow to C35 delayed-B.  No dev or test access is authorized.
