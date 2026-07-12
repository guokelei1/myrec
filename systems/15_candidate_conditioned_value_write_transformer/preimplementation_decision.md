# C15 pre-implementation decision

Decision: **REJECT; DO NOT IMPLEMENT OR RUN.**

C15 tries to innovate in value direction rather than attention mass, but its
allowed design envelope has no surviving mechanism:

1. every vector-valued linear/bilinear candidate/event value is
   `Phi(z,h)=M(z)h` and pulls through event aggregation;
2. the natural low-rank form becomes candidate FiLM/hyperadapter on an ordinary
   pooled value, and cannot produce the required same-aggregate/different-output
   multi-event witness;
3. adding a joint nonlinearity before summation can produce that witness and a
   non-factorized Jacobian, but it is exactly dynamic-filter/edge-conditioned
   message passing, established long before this application;
4. an unrestricted pair MLP offers no falsifiable structure beyond extra
   capacity and incurs `C x H` pairwise compute;
5. candidate centring, bound, zero NULL, monotone exact coordinate, and non-zero
   LayerScale are necessary safety/optimization contracts but do not distinguish
   equal value aggregates.

No source model, runner, synthetic outcome, real A0, GPU lock, or
real/dev/test/qrels access exists.  A successor must impose a new, motivation-
derived structural law on pairwise value direction that is neither separable
FiLM nor generic edge-conditioned messaging.
