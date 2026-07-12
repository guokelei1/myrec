# C31 train-only protocol

The 10,000 C29 fit requests remain the only training set.  C31 internal-A is
C29 delayed-B: 600 strict-nonrepeat requests never feature-materialized,
scored, or labeled.  C31 delayed-B is C29 escrow and also untouched.  No C30 A
request is added to fit.

Phase 1 uses seeds 20260871/72/73, rank 16, temperature 0.1, profile scale 1,
correction scale 2, one epoch, batches of at most 32 complete requests, and all
candidates.  The equal-weight listwise and correction-margin losses are fixed.

No A label is read until label-free A0 passes.  Passing A1 authorizes the
predeclared semantic-identity, unauthenticated, and uniform-history controls on
still-unopened delayed-B.  It does not authorize dev/test.
