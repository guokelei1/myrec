# Frozen synthetic GPU falsifier

Protocol ID: `c10_predictive_evidence_write_synthetic_v1`
Physical device: GPU 3, current repository environment
Run prefix: `20260711_kuaisearch_c10_`

This is a learned-mechanism falsifier, not a dataset result.  The executable
generator interleaves multiple query categories in every user history.  Each
user has a category-conditional preferred attribute, the final event is the
reliable current event for the requested category, and the candidate pool has
query-compatible wrong-attribute hard negatives.  Exactly 35% of requests put
the positive item token in history; the rest use a guaranteed unseen item token
with the same query/category-conditioned preference.

The config freezes 3,072 fit examples, 1,536 untouched synthetic evaluation
examples per seed, three seeds, four epochs, and every optimizer/model value.
Models share the same seed-specific initialization and data order.  All receive
the same listwise loss and positive-candidate token NLL auxiliary term.

## Conjunctive decision

The primary must satisfy every threshold in
`configs/synthetic_gpu_gate.yaml`:

1. non-repeat NDCG gain over its own history-blind state is at least 0.02 in all
   three seeds;
2. mean non-repeat advantage is at least 0.002 over paired-logit, single-pass,
   dual-stream, and centred-attention, with at least two nonnegative seed wins;
3. exact-repeat NDCG is no more than 0.005 below its internal item-only path in
   any seed;
4. wrong-user/query-mask/shuffle gain retention is at most 0.30/0.55/0.75;
5. candidate order changes on at least 5% of transfer requests, score-delta
   standard deviation is at least 1e-4, write is bounded/zero-sum, and the
   no-history comparison is bitwise exact.

One failed conjunct is a gate failure.  Thresholds, generator, code, and tests
are hash-locked before any learned outcome.  A failure forbids real training;
a pass authorizes only preparation and independent review of a label-safe,
train-internal real gate.  It does not authorize dev/test access.
