# C05 G2a final report

Date: 2026-07-11
Decision: **failed; stop the current C05 ladder before G2b/CCEB/dev**

## What was actually tested

The review-amended lock first separated signal existence from CCEB mechanism
attribution.  G2a trained only an ordinary query/candidate target-attention
residual for two fixed epochs on 10,800 history-present requests whose complete
history contained no candidate repeat; the probe itself consumed the latest 20
events with the frozen action/recency prior. A disjoint 1,200-request train-internal
cohort was opened only after the final checkpoint.  Exact recurrence,
candidate centering, corruption training, CCEB, dev and test were absent.

G0 reproduced the calibration D2p coordinate with alpha 0.6 in FP32 over every
candidate.  Its shuffled `(request_id, candidate_item_id)` realignment was
bit-identical with zero duplicate, missing, unknown, non-finite or rank-changing
rows.  G1 used the maximum estimated-work formal batch; the zero residual opened
on optimizer step one and gradients reached query/history/value/head paths on
step two.  The same-seed untrained probe was bit-identical to D2p.

## Frozen gate result

| Quantity | Result |
|---|---:|
| Internal requests | 1,200 |
| D2p NDCG@10 | 0.3118520542 |
| Final probe NDCG@10 | 0.3118520542 |
| Delta | **0.0000000000** |
| Paired bootstrap 95% CI | `[0.0, 0.0]` |
| Three hash-fold deltas | `0.0 / 0.0 / 0.0` |
| Deterministic rescore max difference | `0.0` |
| Cumulative A40 hours | `0.01763` |
| Dev evaluator calls / test reads | `0 / 0` |

The required delta was at least +0.001 with a positive CI lower bound and all
three folds positive.  All three outcome conditions failed.

## Why it failed

This was not a zero-residual or zero-parameter-movement no-op: G1 verified the
two-step gradient path and the parameter vector moved by L2 4.636.  The final
tanh was saturated, so this does not imply useful gradients remained at the
end.  A read-only post-outcome diagnostic found a sharper collapse:

- all 54,637 internal model-internal tanh deltas were exactly `+1`;
- raw pre-tanh differences ranged from 15.76 to 29.55 (median 25.35);
- the observed `(base + 1) - base` standard deviation `2.48e-8` was FP32
  subtraction rounding, not candidate discrimination;
- mean history-update L2 was 25.95 versus candidate-state L2 1.0;
- mean within-request delta range was `1.07e-7`;
- no request changed any candidate order or top-10 order;
- mean positive-minus-negative delta was effectively zero (`-3.08e-9`).

The common update overwhelmed candidate variation and saturated the tanh bound
in a candidate-common translation.
Listwise ranking is invariant to adding the same constant to every candidate,
so the mechanism learned a ranking-null direction rather than
candidate-discriminative history evidence.  This explains the exact zero metric
delta despite finite gradients and substantial parameter movement.

## Binding decision

The current result closes only this claim:

> This fixed two-epoch, zero-initialized, tanh-capped shallow target-attention
> recipe over frozen calibration-D2 states can learn useful non-repeat transfer.

It does not prove that every deeper LM or relative cross-item mechanism is
impossible.  However, the current lock explicitly requires failure to stop the
ladder: no G2b, CCEB training, dev scoring, retry on the exposed internal cohort,
or full Transformer implementation is authorized.

The D2 epoch and alpha had already been selected on the broader calibration
partition containing this cohort.  G2a is therefore a preregistered falsifier
relative to registered D2p, not a completely untouched unbiased evaluation.
That caveat does not affect the observed zero order changes.  The saturation
analysis is a read-only explanatory audit, not gate or tuning evidence.

The curated machine-readable record is
`reports/pps_c05_g2a_signal_gate.json`.  Raw generated evidence remains under
`artifacts/c05_candidate_contrastive_evidence/g2a_v1/`, the final checkpoint
under `models/c05_candidate_contrastive_evidence/`, and run metadata under
`runs/20260711_kuaisearch_c05_g2a_signal_s20260708_attempt1/`.
