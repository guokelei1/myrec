# 2026-07-11 C06 candidate-local Hodge-trusted architecture

## Why C06 exists

C05 moved its trainable residual but produced the same saturated `+1` for all
54,637 internal candidates and changed no ranking. C06 therefore removes
request-common personalization from the hypothesis space instead of adding a
loss penalty or a dataset-specific rule.

## Architecture decision

C06 uses one jointly trained block-sparse Transformer. Query tokens see only
query; each candidate segment sees query plus itself; history sees
query/history. The history-blind candidate state produces the base score.
History can affect final logits only through a bounded, permutation-equivariant
conservative flow.

The primary flow is candidate-local Hodge trust. A learned history-conditioned
skew field is decomposed into a rankable gradient and cyclic residual. The
cycle residual controls candidate/event-local trust, but only the projected
gradient can supply preference direction. Symmetric endpoint trust preserves
skew symmetry, so every history update is zero-sum in final score space.

No dataset ID, category bucket, query type, hand-built click/purchase weight or
KuaiSearch branch exists in the primitive.

## Pre-outcome review changes

Two earlier formulations were rejected before any C06 data outcome:

1. one global Hodge scalar per event reduced exactly to ordinary event gating;
2. gating the original field as `W F W` could turn a divergence-free cycle
   back into a signed ranking signal.

The current operator gates only the projected gradient. The earlier FP32 Gram
energy subtraction was also replaced by FP64 centered-factor moments after a
near-collinear-factor stress case exposed catastrophic cancellation.

## Current evidence and boundary

The CPU prototype includes the full Transformer information barrier, jointly
trained base head and flow head. Thirty-eight unit/structural tests pass,
including explicit small-graph Hodge oracles, pure potential/cycle endpoints,
cycle-sign invariance, near-coincident factors, mixed precision, no-history,
candidate permutation, score conservation, two-step autograd and multi-layer
no-bypass gradients.

The subsequently locked bidirectional synthetic probe passed all frozen
conditions. Local trust gained `+0.01341` pairwise accuracy over `t=1` and
`+0.01348` over the global event gate when cycle energy was planted as a local
error cue. Its delta became `-0.02771` after random decoupling and `-0.05577`
under reverse coupling. Thus the operator is conditionally load-bearing, but
the result also demonstrates that an invalid reliability cue harms ranking.
See `reports/pps_c06_synthetic_mechanism_probe.json`.

These are architecture contracts, not ranking evidence. The `t=1`, global
event, direct learned candidate-gate and centered-attention control sources are
now implemented and included in the 38-test CPU suite. Pairwise-additive and
MIR/SetRank-style controls remain deferred; centered-attention FLOP matching is
not yet closed. No cohort materialization, GPU, train-internal label, dev, full
training or test is authorized.
