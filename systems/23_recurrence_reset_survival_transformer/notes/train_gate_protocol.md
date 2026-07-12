# C23-A locked train-only gate protocol

Status: pre-outcome draft; becomes immutable in `proposal_lock.json` before any
C23 fit label is opened.

## Cohort and label barrier

- source: packed KuaiSearch train only; complete history identity determines
  recurrence;
- pre-cut repeat pool: 12,000 fit requests;
- post-cut repeat pool: 1,200 internal-A, 600 delayed-B, 958 escrow requests;
- 512 no-history and 512 non-repeat requests are structural audits only;
- selection is ascending SHA-256 by role and request ID and is written before
  opening any label-shaped array;
- G0 materializes correct seed-20260708 calibration D2p states/scores for fit,
  internal-A and structural audits, but copies labels only for fit;
- internal-A labels may be opened only after every trained variant passes A0;
  delayed-B/escrow and their features remain unopened.

## Models

Three seeds train each equal-parameter mode for exactly two epochs:

1. `reset_suffix` (primary): last exact occurrence is the origin; only anchor
   and later events enter the causal Transformer;
2. `unreset_history`: all events enter with the same tokenization/backbone;
3. `orderless_suffix`: same anchor/suffix set, position embeddings zeroed;
4. `query_independent`: same reset graph, query projections zeroed.

All modes use full candidate sets, identical dimensions, initialization by
parameter name, optimizer and number of steps.  Fixed item-only and D2p require
no fitting and are binding controls.

## A0: label-free load-bearing gate

Every trained mode must be finite and deterministic.  The primary additionally
must satisfy, on internal-A before labels open:

- candidate-centred correction absolute sum `<=1e-5`;
- `>=5%` requests with any order change and `>=1%` with top-10 membership
  change versus item-only;
- `>=5%` requests whose learned correction changes under a deterministic
  post-anchor shuffle;
- replacing/masking pre-anchor events changes scores by exactly zero;
- query absence returns item-only bitwise;
- no-history and non-repeat audit rows return D2p bitwise;
- candidate permutation is equivariant and repeated scoring is bitwise exact.

Failure stops before internal-A labels.

## A1: train-internal utility gate

After A0, all conditions are binding:

- primary minus item-only NDCG@10 `>=0.001`, paired bootstrap 95% CI lower
  bound `>0`, positive in every seed and every one of three request-hash folds;
- primary minus each learned control NDCG@10 `>=0.0005`, positive in every
  seed; the across-seed paired CI lower bound is `>0`;
- clicked-minus-unclicked primary correction CI lower bound `>0`;
- post-anchor shuffle retains at most 25% of clean gain over item-only and its
  bootstrap 95% upper bound is at most 50%; query-masked gain is exactly zero;
- no threshold, epoch, subset, action multiplier or score cap is tuned after
  observing fit/internal outcomes.

A pass authorizes design of C23-B only.  A failure closes RRST and forbids dev,
test, delayed-B, escrow and soft-anchor implementation.
