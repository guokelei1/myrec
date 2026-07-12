# C37 barycentric residual transport terminal outcome

C37 terminated at train-internal A1. The authoritative report SHA-256 is
`cc3abb8dc9d90915eb2fadfefa2cea3c2c002d671fb05dd93513a1b89dd23742`.
The G0 report SHA-256 is
`4eba1686b807b4bf2b234d21a7ef7cdf228280cbf437cd2810158040d0783710`.

G0 passed all five strict-past authentication checks. Twelve fixed GPU fits
(four modes by three seeds) completed with paired initialization, equal 16,384
parameters, finite gradients, exact repeat/no-history/no-auth/query fallbacks,
and zero determinism/permutation error. All 34 frozen label-free A0 checks
passed on untouched C37-A. The primary changed 36.5% of complete orders and
5.0% of top-10 sets versus global transport, 34.33%/4.33% versus uncentered
transport, and 46.5%/14.0% versus relative-only transport.

The defining structural law replicated across all seeds. Exact differential
abstention covered 26.07%--26.21% of candidate rows; 91.30%--91.58% of active
requests mixed admitted and rejected candidates; the active-set barycenter
matched the shared global write to at most `2.98e-8`; inactive candidates kept
the exact global state; and no candidate residual reversed the global write.

A1 did not validate ranking utility. Mean NDCG@10 was 0.607189 for the primary
versus 0.605516 for D2p, a nominal `+0.001673` whose 95% paired-bootstrap CI
was `[-0.001733, 0.005159]`. All three seed differences versus D2p were
positive, but the candidate-specific barycentric law was indistinguishable
from the simpler global transport (`-0.000003`) and uncentered additive
transport (`-0.000007`). It exceeded relative-only by `+0.001056`, also with a
CI crossing zero. True authenticated history did not significantly beat the
wrong-history control.

Therefore C37 establishes that the conservation primitive is structurally
real, selective, and problem-aligned, but not that its candidate-specific
residual pays utility rent. The weak positive signal is attributable at most
to the shared authenticated global transport, not to barycentric centering.
No threshold, temperature, cohort, or loss rescue is authorized on C37-A.
C37 delayed-B, escrow, dev, and test remain unopened, and C37 is closed.
