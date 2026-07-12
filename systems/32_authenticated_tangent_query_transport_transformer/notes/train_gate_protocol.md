# C32 train-only protocol

The 10,000 C31 fit requests remain the only training set.  C32 internal-A is
C31 delayed-B: 600 strict-nonrepeat requests never feature-materialized,
scored, or labeled.  C32 delayed-B is C31 escrow and is also untouched.  The
now-open C31-A cohort is not added to fit or any gate.

Phase 1 uses seeds 20260901/02/03, rank 16, temperature 0.1, profile scale 1,
correction scale 2, one epoch, at most 32 complete requests per batch, and all
candidates.  Equal-weight listwise and correction-margin losses are unchanged.

No A label is read until label-free A0 passes.  Passing A1 authorizes the
predeclared unprojected, adapted-attention-tangent, and unauthenticated-tangent
controls on still-unopened delayed-B.  It does not authorize dev/test.
