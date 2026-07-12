# C36 conservative barycentric formulation

C35 separated selectivity from utility: candidate-relative surplus repaired the
near-always-on support geometry but lost to both D2p and its candidate-shared
global tangent control. C36 therefore does not tune C35's centering statistic
or threshold. It introduces a common-plus-contrastive residual law inside
history-to-candidate attention.

The candidate-specific surplus displacement is centered over only admitted
candidates, leaving exact abstainers at zero differential write. A shared
query-tangent history displacement is then added, while one request-wise
max-norm coefficient bounds the centered deviation. This makes the
pre-normalization candidate mean exactly equal to the global write and prevents
any candidate deviation from reversing that write.

Two label-free audits used the permanently excluded C35-A surface and read no
labels or metrics. Across three prior adapter states, the formulation had:

- global-mean maximum error `2.98e-8`;
- inactive-candidate global-state error `0`;
- zero nonpositive global alignments;
- 25.32%--25.85% exact relative abstention;
- 90.53%--91.12% mixed-admission requests;
- 50.59%--53.25% full-order and 5.03%--7.69% top-10 change versus global;
- at least 29.88% full-order and 1.18% top-10 change versus the unbounded
  reduction, so the norm bound is load-bearing;
- material differences from uncentered and relative-only controls in every
  seed.

These checks establish algebraic identity and activity only. They do not use or
predict ranking quality. C36-A is untouched C35 delayed-B; C36 delayed-B is
untouched C35 escrow. All five modes must train before A labels, and the primary
must beat D2p plus every exact reduction. The local selection SHA-256 is
`ad7aab244a21d6eae9b4d5d310cacb8b9b31b27160323733069d76fb81f7a146`.
