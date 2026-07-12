# C51 proposal — cross-event affinity covariance

Prediction-error memories failed because their value direction was not
relevance-aligned.  C51 uses no predicted value.  For query `q`, candidate
`c_i`, and history events `h_j`, let `a_j=<q,h_j>` and
`b_ij=<c_i,h_j>`.  The candidate correction is

`cov_j(a_j,b_ij)`.

It asks whether query and candidate agree on *which events within this user's
history* are salient, after removing the user's common semantic affinity.  It
is candidate-specific, query-conditioned, event-permutation invariant, exact
zero for fewer than two events, and contains no learned/dataset branch or
output rescaling.

The exposed formulation gate binds query base, uncentered second moment,
Pearson/CKA-style normalization, plain KRR, C47 posterior support, fixed
softmax, and wrong history.  C51 must beat every control with positive
intervals and all fold signs on both domains and retain clicked direction and
specificity; otherwise it closes before training or reserve.
