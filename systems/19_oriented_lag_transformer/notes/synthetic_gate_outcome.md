# C19 synthetic gate outcome

Status: **failed-stop; C19 is closed before repository data, dev, test or
qrels access**.

## Integrity

- physical GPU 3 (NVIDIA A40), CUDA 12.4, PyTorch 2.6.0+cu124;
- elapsed time: 188.583 seconds;
- proposal aggregate: `5453db5423e896f1a2487a3abe948a7f7813caacf93152920c1bb1f2ab0ded49`;
- proposal-lock SHA-256: `b1f619879400466f9f05d9c2e94e535571bfbd14612d6efffb71acbbc9e9f87a`;
- raw artifact SHA-256: `7cf2c3d9de6841cf08ed12d69bb1daaf56d7a11bc10cfd9e329b6c9f63644487`;
- all modes had 28,388 parameters and identical per-seed initialization;
- repository/standardized data reads, dev calls and test access: zero.

## Frozen outcome

| seed | repeat | supported | base supported | best structured | free signed | corruption retention | permutation error | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 20260721 | 1.0000 | 0.9983 | 0.8733 | 0.9375 | 0.9427 | 0.000003 max | 3.81e-5 | no |
| 20260722 | 0.9983 | 0.9774 | 0.9306 | 0.9878 | 0.9375 | 0.1940 max | 6.10e-5 | no |
| 20260723 | 1.0000 | 0.9097 | 0.0000 | 0.9288 | 0.9201 | 1.0005 max | 1.05e-4 | no |

`best structured` is the maximum supported accuracy among diagonal, forward
and symmetric controls.  Corruption retention is the largest of wrong-history,
shuffle, query-mask and coarse-only target-margin retention.

## What the result establishes

1. The hidden-state operator is trainable and load-bearing: repeat accuracy is
   at least 0.9983, supported accuracy is at least 0.9097, the score-delta
   load-bearing fraction is 1.0, and no-history remains bitwise base.
2. The algebraic orientation witness is exact in practice: reversal correlation
   is `-1` to numerical precision for all seeds.
3. Those facts are insufficient.  OLT beats the best structured control on
   supported accuracy in only one of three seeds and fails the worst-subset
   advantage in two seeds.  The reverse subtraction is therefore not a stable
   improvement over ordinary forward induction.
4. Counterfactual behavior is unstable.  Seeds 20260721/20260722 mostly suppress
   corrupted evidence, while seed 20260723 retains essentially the entire
   margin under wrong history, shuffle and reversal.
5. The synthetic generator exposed a candidate-only shortcut: on supported
   rows the target has quality 0, all ordinary distractors have quality below
   0, and the predecessor has quality 1.  This makes the target the unique
   second-highest-quality candidate.  The frozen base-gain condition correctly
   rejects the resulting 0.8733/0.9306 base accuracy in two seeds, but the gate
   cannot cleanly attribute the remaining high accuracy to temporal transfer.
6. GPU candidate-permutation errors of `3.81e-5` to `1.05e-4` exceed the frozen
   `1e-5` bound.  CPU structural equivariance passed; this is most consistent
   with floating-point execution order, but it remains a formal gate failure.

## Decision and next constraint

C19 receives no post-lock repair and no real-data gate.  A successor must make
candidate features exchangeable conditional on the target, so neither the base
branch nor an evidence-independent hidden write can identify the label.  It
must also require a request-specific counterfactual closure: the personalized
write should disappear when the particular query/history support path is
broken, rather than merely relying on a learned affinity to become small.
