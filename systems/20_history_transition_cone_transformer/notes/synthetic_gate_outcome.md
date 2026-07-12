# C20 synthetic gate outcome

Status: **failed-stop; C20 is closed before repository data, dev, test or
qrels access**.

## Integrity

- physical GPU 0 (NVIDIA A40), CUDA 12.4, PyTorch 2.6.0+cu124;
- elapsed time: 912.198 seconds;
- proposal aggregate: `3655ab38b2b16402e6c1625d8044c1d6c7346b5326c34a7fe5f747b1146a364f`;
- proposal-lock SHA-256: `37411d703c00913d9406f6c24e9d162b6dbbbbac495320d12cac49c3d217c3d5`;
- attempt-marker SHA-256: `eceb7093eef2470186c0e10374ddbad943b0cc32e3783f35bd06a36471e10d8e`;
- raw artifact SHA-256: `dafc29032586d282b85c8892d349f976b3ccb6159821b4275d7af8b4f05617f8`;
- all modes had 26,683 parameters and identical per-seed initialization;
- repository/standardized data reads, dev calls and test access: zero.

## Frozen outcome

| seed | no history | repeat | supported | base supported | best control | clean target-vs-reverse | multi-transition | max primary retention | pass |
|---|---:|---:|---:|---:|---|---:|---:|---:|---|
| 20260724 | 0.9766 | 0.9913 | 0.3264 | 0.1319 | pooled MLP 0.6076 | 2.5090 | 0.9965 | 0.6784 | no |
| 20260725 | 0.9792 | 0.9913 | 0.2257 | 0.1319 | pooled MLP 0.5625 | 1.3447 | 0.9705 | 0.8380 | no |
| 20260726 | 0.9844 | 0.9896 | 0.3576 | 0.1181 | span 0.6094 | 1.9815 | 0.9896 | 0.6299 | no |

Primary retention is the maximum of wrong-history, event-shuffle, query-mask
and coarse-only target-margin retention.

## Interpretation

1. The safety/base contracts worked.  No-history accuracy is at least 0.9766,
   repeat accuracy is at least 0.9896, base supported accuracy stays near the
   exchangeable chance level 0.125, no-history is bitwise base, and all runs
   are finite/deterministic.
2. The solver was not dormant.  Between 97.0% and 99.7% of supported targets
   used at least two positive coefficients and met the reconstruction-reduction
   threshold.  Clean target-versus-reverse margins are positive in every seed;
   reversing the sequence makes them negative.  The nonnegative sign witness
   therefore survives training.
3. That witness does not yield good listwise ranking.  Supported accuracy is
   only 0.2257–0.3576 and the target margin against the strongest distractor is
   negative in every seed.  The cone separates the exact reverse but admits
   enough partial reconstructions of other candidates to confuse the ranker.
4. Every seed loses to a simpler control by a large margin.  The pooled MLP is
   best in two seeds (0.6076/0.5625); unconstrained span is best in the third
   (0.6094).  Exact iterative conic reconstruction therefore pays no predictive
   rent even on its own mechanism task.
5. Wrong/coarse/query corruptions often collapse the write, but event shuffle
   retains 62.99%–83.80% of the clean margin.  Once adjacent differences are
   placed into an unordered dictionary, NNLS forgets their chronological
   locations; shuffled trajectories can still generate a similar cone.  This
   is a primitive-level limitation, not a threshold issue.
6. Feeding only the reconstructed vector `p_i` also hides the closure residual
   `r_i-p_i` from the upper Transformer, forcing it to relearn equality from
   random directions.  Adding that residual to the same order-blind cone would
   not fix item 5 and would be a post-outcome readout repair, so C20 is not
   amended.

## Decision

C20 receives no retry, threshold change, readout patch or real-data gate.  A
successor must retain chronology in the evidence object itself—for example a
contiguous path/segment relation—rather than solving another unordered set of
transition vectors.  It must still compare against one-step transition
attention, pooled history and the failed cone primitive.
