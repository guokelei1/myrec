# C33 fresh tangent confirmation terminal

C33 asked whether C32's spherical-tangent query transport would reproduce on
fresh requests and beat its exact capacity-matched unprojected reduction.  It
reused no C32 A/delayed-B/escrow request, trained both modes before A labels,
and paired each mode by seed, initialization, request order, capacity, loss,
and fixed hash-fold partition.

The experiment separates two conclusions.  First, the broad direction did
replicate: tangent beat D2p by +0.002988, with every seed and every fold
positive.  This is evidence against a one-cohort accident.  Second, the effect
was not statistically secure, and the tangent-specific margin over unprojected
transport was only +0.000583 with a zero-crossing interval and one negative
fold.  Therefore tangent projection has not paid robust architecture rent.

The result sharpens the architectural failure mode.  Moving one shared query
for the whole candidate set is too coarse: it is active and causally
authenticated, yet its held-out candidate correction is not reliably aligned
with clicked versus unclicked candidates.  Further projection scaling,
attention-temperature tuning, layer-position sweeps, or delayed-B rescue would
optimize the same weak primitive against KuaiSearch and are prohibited.  The
next admissible hypothesis must make evidence selection and state change
candidate-specific inside the Transformer while retaining exact recurrence and
no-history contracts, and it must be frozen on another untouched cohort.

Authoritative curated report: `reports/pps_c33_train_gate.json` (SHA-256
`0df8191071f0cffc951f60d644d7fd057dac2bab9153dab59cfe37ee14a81cdd`).
No dev/test evaluator was called; delayed-B and escrow remain unopened.
