# HSO same-checkpoint recovery protocol

Status: pre-outcome recovery protocol, frozen before any HSO held-out-fold
label or metric is opened.

HSO v1 produced all twelve label-free OOF score artifacts.  Full, text, and ID
history modes passed every mechanical check in all three folds and reduced fit
loss strongly.  The separately trained null mode failed its predeclared loss
trend in all three folds, although its scores were finite and mechanically
valid.  Therefore the original independent-null gate is invalid and its labels
remain closed.

This recovery does not train, rescore, select a checkpoint, alter a threshold,
or reuse the failed independent-null scores.  It evaluates only outputs already
produced by the three mechanically valid history checkpoints.  Each checkpoint
was trained under the locked 15% request-level null-history dropout and already
scored, before labels, four fixed inputs: true, matched wrong, reversed-event,
and empty history.  Its empty-history path is consequently a trained,
parameter-identical control.  The fold-legal frozen BGE plus train-fold
popularity score is a second, parameter-free anchor.

A source (`full`, `text`, or `id`) is observable only when all of the following
hold on the 29,277 strict-nonrepeat OOF requests:

1. true minus the same-checkpoint empty-history path is at least `+0.002`
   NDCG@10, has a positive 95% user-cluster interval, and is positive in every
   user fold;
2. true minus the fold-legal fixed base is at least `+0.002`, has a positive
   interval, and is positive in every fold;
3. true minus matched wrong history has a positive interval and is positive in
   every fold;
4. all three source checkpoints retain their already-recorded finiteness,
   candidate hash, determinism, and permutation contracts.

Event reversal remains classificatory rather than binding.  The source-to-next
architecture mapping is unchanged: text authorizes a semantic-history target;
ID without text authorizes collaborative identity memory; only full authorizes
a coupled semantic/collaborative carrier; no passing source triggers the
preregistered Amazon counterpart before any further proposed architecture.

This recovery cannot validate a proposed system or authorize dev/test/qrels.
