# C34 Candidate Tangent-Cone Attention Transformer

C34 tests one architecture change derived from the C31--C33 terminal result:
a request-global query displacement is too coarse, so every candidate must
receive its own history-conditioned query state.  Candidate and authenticated
history states are projected into the tangent space at the query.  Only events
inside a candidate's positive tangent half-space can write; orthogonal or
opposed events produce exact zero attention mass for that candidate.

This is a dataset-agnostic Transformer attention law.  C34 uses no query type,
category, label-derived slice, or dataset branch.  Even the 10,000-request fit
cohort is newly hash-selected and has no overlap with any C32/C33 target role.
Previously used wrong-history donor requests may enter the new hash pool because
their own query/candidate target surface was never feature-materialized,
scored, or labeled; only their history was used as label-free corruption. Three
parameter-identical modes—tangent cone, forced-softmax target
attention, and candidate-shared global tangent transport—train before A labels
open.  No dev or test access is authorized.
