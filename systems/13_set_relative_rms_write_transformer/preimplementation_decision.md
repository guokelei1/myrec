# C13 pre-implementation decision

Decision: **REJECT; DO NOT IMPLEMENT OR RUN.**

The structural parts are sound: candidate centring eliminates C05-style common
mode, set spectral whitening is permutation equivariant, a global bound is
possible, no-history can be bitwise, and exact recurrence can remain one
monotone coordinate inside the same head.

The proposed scientific primitive does not pass:

1. scalar set RMS is exactly request-adaptive scalar rescaling of ordinary
   centred attention;
2. full whitening is non-scalar only for rank-two-or-higher residuals and is a
   generic set/activation normalization, not a new candidate-conditioned
   evidence mechanism;
3. its spectral gain is largest on the weakest modes.  Small epsilon therefore
   equalizes weak noise with strong signal; large epsilon approaches fixed
   scalar rescaling;
4. without the explicitly forbidden reliability gate or base-score router, it
   cannot distinguish a tiny aligned residual from tiny wrong-user noise;
5. forcing a `1.83e-5` zero-order-change residual to fixed magnitude assumes its
   direction is useful, which the preceding experiments did not establish.

Thus C13 either reduces to scale calibration or pays for order changes by
amplifying uncertain directions.  Existing set normalization and whitening work
also makes the architecture-level novelty claim too weak.  No source code,
runner, synthetic outcome, real A0, GPU lock, dev/test/qrels access, or data read
is authorized.  Reopening requires a different primitive that learns or proves
directional evidence fidelity rather than normalizing uncertainty away.
