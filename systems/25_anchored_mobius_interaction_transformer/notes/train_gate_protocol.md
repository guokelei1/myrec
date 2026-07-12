# C25 locked train-only protocol

Status: pre-outcome draft; immutable after proposal lock.

## Roles and barriers

- structurally select strict history-present non-repeat requests before labels;
- 6,000 fit, 1,200 internal-A, 1,200 delayed-B and 1,200 escrow requests are
  disjoint hash partitions;
- 512 repeat-present and 512 no-history requests are structural audits;
- frozen wrong-history donors are matched by coarse history-length and time
  bins, come from outside all outcome roles, and cannot overlap the recipient
  candidate set;
- G0 opens fit labels only.  Internal-A opens after A0; delayed-B opens only
  after A1; escrow/dev/test remain closed.

## Fixed training

Three seeds train `mobius3`, `joint_delta`, `pairwise_ch`, and `trilinear` for
two epochs on full candidate sets.  Modes share parameters, initialization,
all eight presence-lattice potential evaluations plus the direct-product
evaluation, optimizer schedules and listwise click loss.  There is one attempt, no sweep,
candidate sampling or checkpoint selection.

## A0 — label-free load-bearing and contracts

- finite training, active gradients, matched parameters/initialization;
- correction sum absolute maximum `<=1e-5`, deterministic rescore exact and
  candidate permutation error `<=1e-6`;
- seed-averaged primary changes at least 5% of internal-A orders and 1% of
  top-10 memberships relative to D2p;
- wrong history changes primary corrections on at least 20% and rankings on at
  least 5% in every seed;
- query absence and no history return D2p bitwise; repeat-present requests
  return registered item-only bitwise;
- the primary event cell is exactly null when query, candidate or event is
  replaced by its zero anchor.

Failure stops before internal-A labels.

## A1 — internal utility

- primary minus D2p NDCG@10 `>=0.001`, paired 95% CI lower bound `>0`, positive
  in all seeds and all three request-hash folds;
- primary minus each trained control `>=0.0005`, paired CI lower bound `>0`,
  positive in every seed;
- wrong-history gain retention at most 25%, bootstrap CI upper bound at most
  50%, and true-minus-wrong paired CI lower bound `>0`;
- clicked-minus-unclicked primary correction CI lower bound `>0`.

Failure closes C25 and keeps delayed-B closed.

## A2 — delayed confirmation

Without retraining or threshold changes, repeat the D2p/control/wrong-history
conditions on delayed-B.  Only A0+A1+A2 authorizes a later full design review;
it does not authorize dev/test by itself.
