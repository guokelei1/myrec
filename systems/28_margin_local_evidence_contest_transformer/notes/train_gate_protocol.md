# C28 train-only gate

Status: pre-outcome draft; immutable after proposal lock.

Roles are outcome-isolated before selection: fit reuses 3,000 already-open C27
fit labels; internal-A is all 600 C27 escrow requests (never materialized or
scored); delayed-B is the disjoint remaining 600 C26 escrow requests; C28
escrow is 600 strict-nonrepeat request donors never used as ranking outcomes;
repeat/no-history are the 256 C26 structural requests not selected by C27.
Wrong-history donors are generated label-free, matched by registered history
length/time buckets, outside all outcome roles, with zero recipient-candidate
overlap.

Three seeds train five equal-parameter/equal-compute modes for two epochs.  A0
uses the 600 new internal-A requests without labels.  Only A0 opens their
labels; only A1 opens delayed-B.  C28 escrow and all dev/test remain closed.

A0 repeats C27's complete contract: finite/matched training; pair
complement/diagonal `<=1e-6`; exact neutral base order, deterministic rescore,
candidate permutation `<=1e-6`; at least 5% order and 1% top-10 change versus
D2p; wrong-history correction/order/top-10 changes at least 20%/5%/0.5% in
every seed; exact query/no-history D2p and repeat item-only fallbacks.

A1/A2 require primary-minus-D2p `>=0.001` NDCG@10 with positive CI, all seeds
and folds; primary-minus-each control `>=0.0005` with positive CI and all
seeds; wrong-history gain retention `<=0.25` with CI high `<=0.50`; and
positive-CI true-minus-wrong and clicked-minus-unclicked corrections.

The margin kernel, its unit scale, thresholds, roles, donors, tokenization,
schedule, and checkpoints are immutable after lock.  Passing A2 authorizes
review, not dev/test.
