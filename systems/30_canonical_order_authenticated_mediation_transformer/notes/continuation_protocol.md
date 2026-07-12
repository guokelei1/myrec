# C30 canonical-order continuation protocol

Status: pre-outcome; immutable after continuation lock.

C29 is terminal at label-free A0 and remains unchanged.  Its sole failure was
one seed's 1.3709e-6 candidate-permutation score difference against 1e-6.  All
other 18 checks passed, and A labels remain unopened.

C30 binds C29's three final checkpoints.  For each request it accepts any
caller candidate order, sorts positions by the stable string form of item ID,
runs the unchanged Transformer in that canonical order, centers the correction
in canonical order, and recovers scores to caller order.  Candidate item IDs
must be unique within a request.  This is the only implementation change.

C30 performs zero optimizer steps.  It reruns clean, wrong-history,
deterministic, and reversed-caller-order scoring on the same A.  It inherits
only C29 checks whose inputs are unchanged: G0 authentication, training,
parameter, fallback, selection, and candidate hashes.  Activity and corruption
checks are recomputed from canonical scores.  The candidate-permutation limit
remains 1e-6; it is not relaxed.

Only if every A0 check passes may the common evaluator open A labels and apply
the unchanged C29 A1 requirements: mean NDCG@10 gain over D2p >=0.001, positive
95% CI, positive gain in every seed and hash fold, positive-CI true-over-wrong,
and positive-CI clicked correction direction.  Failure is terminal.  Passing
A1 authorizes a new review; it does not itself open delayed-B, dev, or test.
