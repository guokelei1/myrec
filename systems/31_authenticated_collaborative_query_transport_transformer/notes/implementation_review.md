# C31 pre-lock implementation review

Status: accepted for one train-internal falsifier; not accepted as a result.

## Core architectural test

C31 tests whether the missing operation is a request-level representation
change rather than another candidate-local readout.  A frozen BGE Transformer
places query, authenticated history items, and candidates in one coordinate
system.  One shared rank-16 residual adapter learns a collaborative deformation;
one authenticated profile then transports the query.  Every candidate is
compared with that same transported query.  The only trainable tensors are the
two shared low-rank matrices (16,384 parameters); there is no scalar candidate
head, user-ID edge, router, or dataset-specific branch.

The registered D2p baseline retains its independently frozen, fine-tuned query
tower.  C31 transport embeddings are deliberately produced by the untouched
BGE snapshot, matching the raw item-title embeddings and the diagnostic that
authorized this candidate.  Mixing those two query towers would invalidate the
shared-space hypothesis, so G0 materializes and hashes them separately.

## Outcome and leakage review

- The old C30-A cohort was used only to formulate the primitive and is excluded
  from C31 fit and all C31 gates.
- C31 fit is exactly the already-open C29 10,000-request fit cohort.
- C31-A is the former C29 delayed-B: it was selected earlier but never
  feature-materialized, scored, or labeled.  C31 delayed-B is the former C29
  escrow and remains closed.
- Selection uses label-free train metadata.  G0 may open fit labels only after
  the proposal lock.  A labels may open only after all label-free A0 checks pass.
- No dev/test record, qrel, or evaluator call is authorized.

## Structural contracts

- strict-past, same-timestamp score-before-update authentication;
- wrong-user history corruption with user and candidate overlap forbidden;
- exact D2p for no authenticated history, explicit no-auth, or query mask;
- exact item-only fallback for any repeat request;
- stable item-ID canonical candidate computation and caller-order restoration;
- all candidate hashes asserted before scoring and before label access;
- one attempt, three new seeds, full candidate sets, no candidate sampling.

## Main threats and decisions

The positive diagnostic is selection-biased and therefore not evidence of
generalization.  C31-A is the required independent falsifier.  The fixed rank,
temperature, scales, losses, epoch count, and thresholds are inherited without
a sweep.  Average clicked-direction is report-only because the primitive makes
one request-level ranking displacement; utility still requires a positive
confidence interval, every seed, every hash fold, and true history over wrong
history.  A pass authorizes, but does not itself satisfy, semantic-identity,
unauthenticated, and uniform-history reductions on delayed-B.

Decision: the implementation is sufficiently narrow and falsifiable to freeze.
