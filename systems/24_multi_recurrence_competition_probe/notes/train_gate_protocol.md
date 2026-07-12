# C24 locked train-only protocol

Status: pre-outcome draft; immutable after `proposal_lock.json`.

## Roles and label barrier

- 6,000 multi-repeat fit requests are selected from C23 fit; their compact
  labels were already opened by C23 G0 and are reused without reopening the
  original train label array;
- the union of C23 delayed-B and escrow contains 940 multi-repeat requests
  whose labels remain untouched; a new hash split freezes 600 internal-A and
  340 escrow requests;
- 512 single-repeat, 512 no-history and 512 non-repeat requests are structural
  A0 audits only;
- C24 G0 materializes D2p/query/item states for fit, internal-A and structural
  roles, but no new label values;
- internal-A labels are opened from the original train label array only after
  all A0 checks pass; escrow remains closed.

## Frozen training

Three seeds train `set_attention`, `independent`, and `query_independent` for
two epochs on full candidate sets.  All modes have identical parameters,
initialization, optimizer steps and listwise click loss.  No candidate sampling,
dev evaluator, hyperparameter sweep or checkpoint selection is allowed.

## A0 — label-free

- finite training with at least one nonzero-gradient parameter in every
  seed/mode; equal parameters/initialization;
- candidate correction sum absolute maximum `<=1e-5`;
- deterministic rescore exact; candidate permutation error `<=1e-6`;
- primary changes at least 5% of request orders and 1% of top-10 memberships
  versus item-only;
- disabling candidate-candidate edges in the trained primary changes
  corrections on at least 10% and rankings on at least 5% of requests;
- query absence returns item-only bitwise;
- single-repeat returns item-only bitwise; non-repeat/no-history return D2p
  bitwise.

Failure stops before internal-A labels.

## A1 — utility

- primary minus item-only NDCG@10 `>=0.001`, paired 95% CI lower bound `>0`,
  positive in all seeds and all three request-hash folds;
- primary minus each learned control `>=0.0005`, paired CI lower bound `>0`,
  positive in all seeds;
- same-checkpoint cross-edge ablation retains at most 25% of clean gain and its
  bootstrap CI upper bound is at most 50%;
- clicked-minus-unclicked correction CI lower bound `>0`.

A pass authorizes a separate C25 design formulation only.  A failure closes
multi-recurrence candidate competition and forbids escrow/dev/test access.
