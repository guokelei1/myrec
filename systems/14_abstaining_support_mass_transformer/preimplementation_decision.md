# C14 pre-implementation decision

Decision: **REJECT; DO NOT IMPLEMENT OR RUN.**

The desired structural behavior—history can abstain, no-history writes zero,
and absolute real-event mass is visible—is useful.  The proposed operator does
not provide a new mechanism:

1. every candidate/event/head subprobability write `w=rho p` is exactly a
   softmax distribution over real events plus a zero-value NULL;
2. it is simultaneously ordinary attention multiplied by a candidate/head
   scalar gate;
3. the radial-support/tangent-allocation Jacobian is null-softmax under a smooth
   coordinate transform, so separate heads change parameterization, not the
   attainable function or matched gradients;
4. `rho->0` still makes support and allocation gradients vanish; small non-zero
   LayerScale prevents only exact zero initialization and further scales those
   gradients;
5. exact zeros are already sparsemax/entmax territory, while non-normalized
   absolute event support is already sigmoid attention/Multiscreen territory;
6. per-head post-attention sigmoid suppression is directly covered by recent
   gated-attention work, and ZAM/C03 already cover zero-vector/dustbin target
   attention in the personalized-ranking neighbourhood.

Candidate centring and a hidden bound applied afterward are sound contracts but
cannot make identical upstream writes novel.  No admissible one-primitive
variant survives: changes that break equivalence become signed/vector output
gating, independent screening, or transport/value modulation, all already
covered or outside the fingerprint.

There is no model, runner, synthetic result, real A0, GPU lock, or
real/dev/test/qrels access.  A successor must change the information carried by
the history write, not only reparameterize its nonnegative mass.
