# C05 staged gate protocol - pre-run review amendment

Status: **frozen before any C05 data-fit outcome**.

The order is binding.  A later stage refuses to run unless every earlier
machine-readable status is `passed`.

## G0 - clean coordinate and data fidelity

No optimizer may exist before G0 passes.

1. Candidate manifest SHA256 must equal
   `94eb667000e0d0f389d0a2a4d4730683b71c129043edbfcf627590376e9c123e`.
2. Use the existing D2 boundary `cut_request_index=87245`.
3. Before opening train labels, select by request-ID hash:
   - 10,800 non-repeat, history-present fit requests from `[0, 87245)`;
   - 1,200 non-repeat, history-present internal requests from `[87245, 96939)`.
4. Define non-repeat using the complete standardized/packed history, not the
   model's last-20 input truncation.
5. D2p must use only:
   - calibration checkpoint SHA256
     `247f27e2cbeef90e3fa5a5f5c2b8488db26e1542aa1617cec19e4b8316373572`;
   - internal-train popularity SHA256
     `d8fd4b618454d8797408bec9d2d219c3bf3059b34283db0816cf059a46ae8e7b`;
   - alpha `0.6`, full FP32 and no autocast.
6. Query state is normalized calibration-encoder CLS.  Candidate/history states
   use the same checkpoint's item adapter over the frozen raw item embedding.
7. Use every candidate.  Negative/candidate subsampling is forbidden.
8. Materialized base scores are keyed by `(request_id, candidate_item_id)`.
   Duplicate, missing, unknown, or non-finite rows fail closed.  A shuffled-row
   realignment must be bit-identical with zero rank mismatches.
9. Training entry points reject dev/test records, qrels, metrics, and any final
   D2 checkpoint/full-train-popularity path.
10. A real all-candidate forward/backward/reload/reporter smoke and wall-time
    projection must finish before the formal G2a attempt.

## G1 - implementation contracts

- no-history probe scores equal the immutable base bit-for-bit;
- all-no-history, all-repeat, all-nonrepeat, no-positive, request-mask-empty,
  and true `T=0` corruption paths are finite;
- padding is zeroed or rejected before normalization/projection, so `NaN * 0`
  cannot contaminate valid rows;
- target-attention candidate permutation is equivariant;
- two optimizer steps from the real initialization yield finite nonzero
  ranking-path gradients;
- base scores are detached and cannot receive gradients;
- deterministic repeat of the synthetic smoke is byte-identical.

The parked CCEB prototype is expected to be history-set invariant.  Shuffle is
therefore not a G1 failure for G2a.  CCEB cannot enter the doc-15 full gate
until a later pre-outcome revision either adds validated time/action encoding
or explicitly changes the full-system claim with coordinator approval.

## G2a - non-repeat signal existence

This stage tests the representation, not CCEB.

- model: one minimal ordinary query/candidate target-attention history update;
- data: only the G0 non-repeat fit/internal requests;
- exact relation: absent by construction and not provided to the model;
- history input: the latest 20 events in chronological order; packed click and
  purchase multipliers are `1.0` and `1.5`, respectively, then multiplied by
  `1/sqrt(reverse_age)` and added to attention logits as a log prior;
- objective: multi-positive listwise loss only;
- request weighting: mean loss within each frozen dynamic batch, with one
  optimizer step per batch (candidate-heavy batches can therefore carry more
  per-request optimization weight; final metrics remain request-equal);
- corruption loss: exactly zero / not instantiated;
- initialization: zero output projection, so the same-seed untrained probe is
  bit-exactly D2p even when history is present; learning opens the residual;
- epochs: exactly two; final epoch is evaluated, with no internal-selected
  checkpoint or hyperparameter retry;
- seed: `20260708`;
- primary metric: request-level NDCG@10 from the shared metric implementation.

Pass only if all are true:

1. final internal NDCG@10 delta over both clean D2p and the same-seed untrained
   zero-residual probe is at least `+0.001`;
2. paired request-bootstrap 95% CI lower bound is above zero;
3. the delta is positive in all three frozen request-ID hash folds;
4. no score is non-finite and a repeated final-epoch rescore is bit-identical;
5. projected end-to-end G0+G2a wall time remains within 2 A40 GPU-hours.

Failure closes only the claim that a shallow target-attention update over the
frozen D2 representation can learn useful non-repeat transfer.  It does not
prove that every larger LM or every cross-item mechanism is impossible.

## G2b - held-out evidence audit (conditional, not yet implemented)

Run only after G2a passes.  No twin is used in G2a training.  Freeze donor and
replacement construction before scoring:

- different-user, strictly-prior histories matched on query/context,
  history-length bin, action mix, and freshness;
- query-masked inputs;
- matched event replacement that preserves position/action statistics;
- coarse-only history evidence as a descriptive negative-control family;
- shuffle is descriptive unless a temporal representation has first been
  frozen.

True-history ranking gain must exceed each hard twin separately with paired CI;
attention mass alone is not evidence.

## G3 - CCEB mechanism attribution (conditional)

Only after G2a/G2b pass may CCEB be revised and re-locked.  Required controls
include ordinary target attention, no candidate contrast, positive-only
centering, Denoising-Attention-plus-groupwise context, history-free groupwise
ranking, and a real score-space trust-region ablation.  Active parameters,
steps, FLOPs, and candidate sets must match.  Nested-pool/distractor stability
is binding.

## G4 - repeat protection and full Transformer (not authorized)

Only after G3 passes may the system add the established exact recurrence.  The
relation must reproduce action/recency semantics from the registered item-only
coordinate and be monotone at the final logit.  The complete verified base,
including its ranking head, is frozen.  No-history requires zero score/rank
mismatches, not 0.95 concordance.

Dev scoring/evaluation, multi-seed, cross-dataset work and test all require a
new explicit authorization after G4.

## Current budget

| Resource | Authorization |
|---|---:|
| Environment | `myrec-c05` |
| Physical GPU | 0 |
| Run prefix | `20260711_kuaisearch_c05_` |
| G0 + G2a cumulative A40 hours | 2.0 |
| Implementation attempts | 2 |
| Dev evaluator calls | 0 |
| CCEB/full training | no |
| Test | no |
