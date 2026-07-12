# C05 pre-run review and response

Date: 2026-07-11
Outcome status at review time: no C05 data-fit, dev, or test outcome existed.

## Decision

Keep the research question, but do not run the locked CCEB prototype directly.
The original protocol conflated:

1. whether non-repeat history contains learnable ranking signal; and
2. whether candidate centering, a signed dead zone, and L1 attention mass are
   the right mechanism.

The first question must be answered with a simpler nearest-neighbor probe before
the second can be interpreted.

## Blocking findings

- The current CCEB is history-permutation invariant, so its shuffle corruption
  gate was impossible by construction.
- The positive exact alignment bias does not protect final score direction;
  value/output/head maps and a negative global scale can reverse it.
- `sum |attention_weight| < 1` does not bound hidden-state or score residuals
  because downstream operator norms are unconstrained.
- Training on the same corruption families later used as the fidelity gate is
  circular and can teach the diagnostic.
- The first config pointed at final-D2t/full-train-popularity artifacts.  Those
  are legal for dev scoring but invalid for a clean held-out train-internal
  coordinate.  Candidate sampling would also invalidate a candidate-relative
  operator.
- The true empty-twin dimension in the corruption loss produced NaN and was not
  covered by the previous empty-mask test.

## Pre-outcome response

### G2a - signal existence

- non-repeat history-present train requests only;
- full candidate sets;
- exact recurrence disabled;
- one ordinary query/candidate target-attention layer;
- listwise ranking loss only (`corruption_weight=0`);
- fixed two epochs, one seed, no checkpoint selection from internal metrics;
- exact-zero output projection at initialization, making the untrained probe
  bit-identical to D2p before learning opens the residual;
- compare final epoch against clean D2p on a frozen train-internal split.

The clean D2p coordinate uses the D2 calibration checkpoint, internal-train
popularity, alpha 0.6, FP32, and key-based alignment.  The selection is frozen
before train labels are opened.

### Later stages

If G2a fails, close only "cheap non-repeat transfer under the frozen D2
representation"; do not claim all cross-item transfer is impossible.  If it
passes, run held-out freshness/query/length-matched wrong-user and event
replacement audits without training on them.  Only then revise and train CCEB.

CCEB must subsequently add a real final-score trust region, nested-candidate-pool
stability tests, and complete groupwise/Denoising/Differential controls.  Exact
recurrence is deferred until a monotone action/recency-aware final-logit path is
specified; a positive internal bias alone is no longer called protected.

## Resource amendment

The user explicitly authorized experimental validation after this review.
Physical GPU 0 is reassigned from closed C01 to C05.  Environment `myrec-c05`,
run prefix `20260711_kuaisearch_c05_`, and at most 2 cumulative A40 GPU-hours
are authorized for G0 and G2a only.  Dev evaluator calls remain zero; test is
locked.
