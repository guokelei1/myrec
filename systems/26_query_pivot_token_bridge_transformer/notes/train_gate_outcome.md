# C26 train-gate outcome

Status: **terminal failure at label-free A0**.

The locked v2 run completed three seeds, four compute-matched modes, and two
epochs per mode in 5,402.82 seconds.  Each mode had 152,193 trainable
parameters (10,969,729 including the shared frozen word table).  Training was
finite, gradients were active, initialization and capacity matched, and the
determinism, candidate-permutation, query-absent D2p, no-history D2p, and
repeat-present item-only contracts passed.

The query-pivot token bridge changed 66/1,200 complete candidate orders (5.5%)
relative to D2p, just passing the 5% activity floor, but changed only 1/1,200
top-10 sets (0.083%, required 1%).  Frozen wrong histories changed the numeric
correction on 99.08%, 92.25%, and 100% of requests, yet changed only 0.083%,
0.167%, and 10.667% of complete orders and zero top-10 sets.  Thus the token
bridge carried history information numerically but did not reliably carry the
ranking margin.  End-of-training loss summaries also collapsed to nearly the
same values across all four modes.

Candidate-centering accumulated `2.27e-5` versus the strict `1e-5` tolerance.
That numerical failure is secondary: top-10 activity and all-seed wrong-history
order sensitivity fail independently, so no precision repair, score scaling,
or rerun can rescue C26.

Internal-A, delayed-B, escrow, dev, and test labels remain unopened.  C26
closes the specific combination of shared-query-token alignment followed by a
bounded additive D2p correction.  It does not close token-level modeling.  A
successor must make evidence change candidate competition or readout margins
inside the Transformer, with a matched additive-residual control, rather than
merely increase this correction's scale.

The pre-outcome v1 runner interface abort and its clean v2 supersession remain
recorded in `pre_outcome_execution_abort.md`; no ranking output preceded v2.

Raw report SHA-256:
`a62a6b1a99a7337254c9cedbd1c423722f65af7601245f6705cc2eb6712844c0`.
