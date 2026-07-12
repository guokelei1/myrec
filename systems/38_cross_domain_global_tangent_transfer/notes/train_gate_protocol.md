# C38 frozen train-internal transfer protocol

This protocol is pre-outcome.  Numeric thresholds may not change after the
proposal lock or after any internal-A score is produced.

## Data and isolation

- Dataset: standardized Amazon-C4 plus temporal history release.
- Eligible source: upstream history `train` split only.
- Candidate construction: BM25 top-100 over the official sampled-1M catalog,
  positive union, deterministic `item_id` tie-break; at most eight unique
  query terms selected by lowest frozen catalog document frequency.
- Split by SHA-256 of `(request_id, user_id, qid)` after C0/C1 passes:
  6,000 fit; 1,200 internal-A; 1,200 delayed-B; 1,200 escrow; remainder unused.
- Internal-A features and candidate identities are label-free.  Its labels may
  open only after A0.  Delayed-B opens only after A1.  Escrow is not authorized.
- Candidate selection, encoding, wrong-donor construction, and A0 read only
  `records_train_blind.jsonl`; the dedicated label opener may read
  `records_train.jsonl` for the currently authorized role after lock checks.
- Upstream dev/test records, labels, qrels, and evaluator are forbidden.
- Three new seeds are required; seed values are frozen in the execution lock.

Wrong-user donors are selected without labels from another user in the same
retained-history-length bin, with deterministic hash tie-break.  Donors may
not share `user_id` with the target.  Positive category is deliberately not
used because it is a target-label attribute.  If no same-bin donor exists,
use the nearest nonempty length bin and record the fallback rate.

## Modes and fixed budget

All modes share rank, parameter count, initialization, fit requests, order,
optimizer, one epoch, candidate set, listwise loss, direction loss, and three
seeds.

1. `query_attended_tangent` (primary);
2. `query_attended_unprojected` (remove only tangent projection);
3. `mean_history_tangent` (replace only query-conditioned history weights by
   uniform weights).

Frozen carry-over values: rank 16, temperature 0.1, correction scale 2.0,
exact-recurrence base boost 3.0,
learning rate 0.001, weight decay 0.0001, gradient clip 1.0, listwise and
direction loss weights both 1.0, no candidate sampling.

## G0: data/authentication gate

All must pass before implementation training:

1. Amazon C0/C1 report passes, candidate hash is fixed, and dev/test labels are
   physically isolated.
2. Every selected request has nonempty released history; all retained event
   timestamps are strictly below the standardized surrogate request timestamp.
3. The positive item is absent from every retained history.
4. Wrong donors have zero same-user assignments, 100% request coverage, and
   at least 95% same-history-length-bin matching.
5. True and wrong histories are candidate/order independent and are selected
   without labels.

## A0: label-free mechanism gate

All must pass on untouched internal-A before labels open:

- identical trainable parameter count and paired initial state for all modes;
- finite training, nonzero adapter gradients, and updated parameters;
- exact deterministic repeat and candidate-permutation equivariance;
- empty/masked history gives bitwise-zero correction and base-rank equivalence;
- an absent/masked query gives bitwise-zero correction and base-rank
  equivalence;
- any true-history exact candidate recurrence gives bitwise-zero transport
  correction, preserving the common fixed item-only base component;
- primary tangent orthogonality error at most `1e-6`;
- primary changes at least 5% of complete orders and 1% of top-10 sets versus
  base, wrong history changes at least 2%/0.5%, and each equal-capacity
  reduction differs from primary on at least 2%/0.5%;
- no candidate scalar head, dataset/category/query-type branch, or access to
  upstream dev/test/qrels.

## A1: utility and causal gate

Using the shared metric implementation and 10,000 paired bootstrap samples,
all must pass:

1. primary minus frozen BGE base NDCG@10 mean at least `+0.002`, all three
   seeds positive, all three request-hash folds positive, and 95% CI lower
   bound above zero;
2. primary minus each equal-capacity reduction mean at least `+0.0005`, all
   three seeds nonnegative, and 95% CI lower bound above zero;
3. true-history primary minus wrong-history primary has 95% CI lower bound
   above zero;
4. clicked-positive mean correction exceeds candidate-negative mean correction
   with 95% CI lower bound above zero.

Any failure is terminal for C38.  No Amazon threshold, temperature, scale,
loss, history length, candidate pool, cohort, or encoder rescue is allowed.
