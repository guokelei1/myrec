# C29 train-only gate

Status: pre-outcome draft; immutable after proposal lock.

The label-free selection freezes 10,000 fit, 600 internal-A, 600 delayed-B,
600 escrow, 256 repeat, and 256 no-history requests.  C28 escrow is the C29
internal-A because it was never feature-materialized, scored, or labeled.
All other new roles and donors are disjoint and selected without labels.

After proposal lock, G0 may materialize fit, A, repeat, and no-history features
and open fit labels.  It must pass the frozen authentication thresholds in the
proposal.  Delayed-B remains feature-, score-, and label-closed until A1.

Phase 1 uses seeds 20260831/32/33, one full epoch, full candidate sets, and one
physical A40 per seed.  It trains only authenticated mediation.  Fit selection
has no dataset/category/query-type branch and no authentication-present filter.

A0 requires finite training; identical parameter counts/config; at least 5%
complete-order and 1% top-10 change over D2p; wrong history changes at least
20% corrections, 5% orders and 0.5% top-10 in every seed; exact D2p for empty
authentication/no-history/query absence; exact item-only for repeat; candidate
permutation <=1e-6; deterministic rescore exact; and strict prequential
authentication with same-timestamp score-before-update.  No A label is read.

After A0, A1 requires seed-averaged primary-minus-D2p NDCG@10 >=0.001 with
positive 95% CI, positive differences in every seed and three hash folds;
true-minus-wrong and clicked-minus-unclicked correction CIs strictly positive.
Failure is terminal without delayed-B access.

Matched-control training is deferred until A1.  The random-initialization
control has the identical Transformer config and parameter count.  Delayed-B then additionally
requires primary-minus-each control >=0.0005, positive CI/all seeds, and the
same fidelity checks.  Escrow, dev, and test remain closed regardless of
train-gate outcome.
