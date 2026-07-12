# C27 train-only gate

Status: pre-outcome draft; immutable after proposal lock.

C27 deterministically subsamples only C26 roles whose internal-A, delayed-B,
and escrow labels were never opened: 3,000 fit, 600 internal-A, 600 delayed-B,
600 escrow, plus 256 repeat and 256 no-history structural requests.  Donor
mappings remain paired with their recipients.  C26 token arrays and compact fit
labels are immutable read-only inputs; original train labels remain inaccessible
until the staged gates.

Three seeds train four equal-parameter/equal-compute modes for two epochs.  A0
uses 600 internal-A requests without labels.  Only A0 opens internal-A; only A1
opens delayed-B; escrow and all dev/test data remain closed.

A0 requires finite active training, matched initialization and parameters,
pair complement/diagonal errors `<=1e-6`, exact deterministic rescore,
candidate permutation error `<=1e-6`, at least 5% order and 1% top-10 change
versus D2p, wrong-history correction/order/top-10 changes of at least
20%/5%/0.5% in every seed, query/no-history D2p identity, repeat item-only
identity, and exact base-order preservation under neutral evidence.

A1 and A2 each require primary-minus-D2p `>=0.001` NDCG@10 with positive CI,
all seeds and folds; primary-minus-each-control `>=0.0005` with positive CI and
all seeds; wrong-history gain retention `<=0.25` with CI high `<=0.50`;
true-minus-wrong and clicked-minus-unclicked contest correction CI lows `>0`.

No threshold, architecture, selection, donor, token input, schedule, or
checkpoint may change after lock.  Passing A2 authorizes review, not dev/test.
