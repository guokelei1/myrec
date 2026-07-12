# C02 frozen screening protocol

Status: pre-outcome.  Thresholds below must not be relaxed after the single dev
call.  Seed `20260708`; physical GPU 1; private environment `myrec-c02`.

## Inputs and integrity

- standardized data: `data/standardized/kuaisearch/v0_lite`;
- candidate hash:
  `94eb667000e0d0f389d0a2a4d4730683b71c129043edbfcf627590376e9c123e`;
- frozen query base: D2t seed 20260708 checkpoint;
- frozen no-history anchor: D2p seed 20260708 scores;
- train records may supply click labels; dev records are label-free;
- scoring/training has no argument for separated dev/test labels;
- test records, test labels, test metrics, and sibling outcomes are prohibited.

The label-free structural subsets are fixed before scoring:

- repeat-present: history present and at least one candidate item exactly
  occurs in history (expected 3,442 requests);
- non-repeat history-present: history present and no candidate exactly occurs
  in history (expected 4,677 requests);
- no-history: empty history (expected 4,110 requests).

Unexpected counts stop execution before evaluation.

## Frozen architecture

- query/history Transformer layers: 1;
- query/candidate Transformer layers: 1;
- hidden width / heads / FFN width: `96 / 4 / 192`;
- Cayley subspace rank: `8`;
- only the last candidate FFN map is modulated;
- skew radius: `0.35`; bounded score residual: `1.5`;
- history truncation: most recent 20 events, order preserved;
- no-history `Delta W=0` by construction.

No layer/rank search is authorized.

## Frozen optimization budget

- one implementation attempt; a second attempt is allowed only for a
  mechanical/numerical defect before the dev call;
- deterministic train sample: 24,000 requests from the first 90% retained
  train region, selected by the frozen seed;
- internal validation: final 3,000 retained train requests, disjoint from the
  sample;
- two epochs for every variant, same request/batch/optimizer-step envelope;
- AdamW, learning rate `1e-3`, weight decay `1e-4`, gradient clip `1.0`;
- primary loss: multi-positive listwise softmax;
- preservation/corruption/core-norm weights: `1.0 / 0.05 / 0.001`;
- at most 8 A40 GPU-hours total across feature preparation, five variants,
  scoring, corruptions, and deterministic rescore;
- zero online API/LLM calls.

Checkpoint selection uses only the frozen train-internal composite in the
config.  Dev never selects epochs, rank, layers, thresholds, or controls.

## Controls

Every control uses the same compact backbone, D2p anchor, train sample, two
epochs, optimizer, candidate batches, and maximum history length:

1. ordinary static LoRA (content-independent internal low-rank update);
2. output-layer history gate;
3. mean-history residual;
4. history-only Cayley HyperAdapter (candidate/query removed from generator).

Total trainable parameter counts must be within 2% of CHHT or the matched-
capacity claim fails.  Observed wall time must be within 25% unless the
operator's declared `r x r` solve explains the difference.

## Train-internal falsifier

All conditions are computed on the frozen internal validation labels with the
shared metric implementation and are diagnostics, not paper results:

1. loss is finite and decreases from first to final epoch;
2. CHHT non-repeat NDCG@10 minus D2p is at least `+0.001`;
3. CHHT repeat-present NDCG@10 minus the item-only teacher is at least
   `-0.003`;
4. CHHT non-repeat NDCG@10 exceeds the best of the four controls by at least
   `+0.0005`;
5. mean true-core norm is at least 1.05 times each wrong/shuffle/coarse/query-
   mask core norm, and mean paired core distance is nonzero;
6. every no-history score equals its D2p base exactly.

Failure is recorded; it does not license threshold changes.  Unless execution
is numerically invalid, the one preregistered dev screening is still produced
so the requested bounded probe is complete.

## Single dev screening

Run ID:
`20260710_kuaisearch_c02_chht_screen_s20260708`.

The scorer first asserts the candidate hash and writes only the unified
`scores.jsonl`.  Exactly one shared-evaluator invocation is allowed and must be
serialized with `tmp/pps_dev_evaluator.lock`.  The resulting evaluator log must
contain exactly one C02 row.

Screening survival requires all of:

1. overall NDCG@10 is no lower than seed-matched D2p `0.3238158367`;
2. on the 4,677 non-repeat requests, CHHT minus D2p is positive with point
   delta at least `+0.001` and paired-bootstrap 95% CI lower bound `> 0`;
3. on repeat-present requests, CHHT minus seed-matched item-only is
   non-inferior: point delta at least `-0.001` and 95% CI lower bound at least
   `-0.003`;
4. all 4,110 no-history scores are bitwise equal to D2p and produce zero rank
   mismatches;
5. wrong-user, shuffled-event, coarse-only, and query-masked label-free
   rescoring each has mean core norm no greater than `0.95 * true`, and at
   least 80% of affected requests change the candidate core tensor;
6. the history-only HyperAdapter does not match CHHT on the internal
   non-repeat threshold in §4;
7. the 1,000-request deterministic rescore has zero missing rows and maximum
   absolute score difference `0.0`;
8. all integrity, environment, budget, file-hash, and log-count checks pass.

Conditions 2–3 use the existing shared per-request evaluator outputs and
`scripts/compare_runs.py` with frozen label-free request-ID files.  No
candidate-local metric implementation or extra evaluator call is allowed.

## Decision and stop-loss

- all survival conditions pass: `advance-to-full-gate`; stop and request a new
  registered budget—do not run multi-seed/full training automatically;
- integrity passes but any scientific condition fails:
  `pivot-before-more-dev` if the audited Cayley operator remains plausible,
  otherwise `stop`;
- any leakage, candidate mismatch, second evaluator row, test access, sibling
  access, or post-outcome threshold/config edit: invalidate C02 and `stop`.

The full common gate remains a later authorization stage.  This screen cannot
establish semantic transfer, identity causality, or paper-level superiority.
