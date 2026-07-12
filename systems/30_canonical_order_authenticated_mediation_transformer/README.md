# C30 Canonical-Order Continuation of C29

C30 is a mechanical, weights-preserving continuation of C29.  It does not add
an architecture primitive, retrain, change a threshold, alter a candidate set,
or inspect internal-A labels before A0.  Its only change is to serialize every
request's candidates by stable item ID before Transformer batching and to
recover scores to the caller's order afterward.

C29 passed 18/19 label-free A0 checks; one seed's fp32 score difference under a
reversed candidate computation order was 1.3709e-6 against a frozen 1e-6
tolerance.  Canonical serialization makes the actual sequence of Transformer
operations independent of caller order.  C30 binds the three final C29
checkpoints and reruns canonical clean/wrong/permutation scoring on the same
label-unopened A role.  Only a complete A0 pass may open A labels.

Delayed-B, escrow, dev, test, full training, and any weight update remain
unauthorized.
