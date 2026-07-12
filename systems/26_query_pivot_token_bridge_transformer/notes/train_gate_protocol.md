# C26 train-only gate

Status: pre-outcome draft; immutable after proposal lock.

C26 reuses the label-free C25 role partition because internal-A, delayed-B and
escrow labels were never opened.  C25 compact fit labels may be copied only
after the C26 proposal lock.  Tokenization reads the label-free registered
corpus and frozen train query-token arrays; it never parses train records.

Three seeds train four equal-parameter/equal-compute modes for two epochs on
6,000 strict non-repeat requests.  A0 uses 1,200 internal-A requests without
labels.  Only A0 opens internal-A; only A1 opens the frozen 1,200 delayed-B;
1,200 escrow and all dev/test data remain closed.

A0 requires finite active training, matched initialization, candidate-centred
corrections `<=1e-5`, exact deterministic rescore, candidate permutation error
`<=1e-6`, at least 5% order and 1% top-10 change versus D2p, wrong-history
correction/order changes of at least 20%/5% in every seed, query/no-history D2p
identity, and repeat item-only identity.

A1 and A2 each require primary-minus-D2p `>=0.001` NDCG@10 with positive CI,
all seeds and folds; primary-minus-each-control `>=0.0005` with positive CI and
all seeds; wrong-history gain retention `<=0.25` with CI high `<=0.50`;
true-minus-wrong and clicked-minus-unclicked correction CI lows `>0`.

No threshold, token length, architecture, donor or checkpoint may change after
lock.  Passing A2 authorizes review, not dev/test.
