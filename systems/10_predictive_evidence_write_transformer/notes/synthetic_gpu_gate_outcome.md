# C10 frozen synthetic GPU gate outcome

Status: **FAIL; C10 real train-internal/dev/test training is not authorized.**

The run executed once on physical GPU 3 after independent lock verification.
It used three frozen seeds and all six registered modes, took 24.28 seconds,
and made zero real-record/dev/test/qrels accesses.

- lock manifest SHA-256:
  `38f584c1d0488bdfbb3b7bb0b18f451fa9928e35424a82ee3caeed17975bc567`
- raw ignored report:
  `runs/20260711_kuaisearch_c10_synthetic_gpu_gate/result.json`
- raw report SHA-256:
  `cf2ea247cea1785c8e438ac1c03cb829a11f6b3cdad3d3f691fdbb9e7827665f`
- environment: PyTorch 2.12.1+cu130, NVIDIA A40

## Binding results

| seed | primary transfer | own base | gain | centred attention | primary - centred | order changed |
|---:|---:|---:|---:|---:|---:|---:|
| 2026071101 | 0.947360 | 0.947737 | -0.000377 | 0.998869 | -0.051509 | 3.06% |
| 2026071102 | 0.944490 | 0.944490 | 0.000000 | 0.996651 | -0.052161 | 3.98% |
| 2026071103 | 0.947241 | 0.947241 | 0.000000 | 0.996185 | -0.048944 | 2.80% |

The required primary gain was at least 0.02 in every seed and the required
mean advantage over every neighbour was 0.002.  Neither held.  Mean primary
advantage was -0.050871 versus centred attention, -0.001925 versus paired
logit, -0.000126 versus dual stream, and +0.000008 versus single pass.  Only
repeat non-inferiority, no-history bitwise identity, and the zero-sum/bound
contracts passed.  Conditional order change also missed the 5% minimum.
Clean transfer gain was non-positive, so all corruption-retention conjuncts
failed by the frozen definition rather than being reinterpreted after outcome.

The primary loss moved substantially (first-to-last: 2.3863→1.2333,
2.3120→1.0112, 2.1808→1.0254), but its ranking behaviour remained almost the
same as single-pass and generic dual-stream controls.  In contrast, the exact
capacity centred-attention control reached last losses 0.5845/0.4739/0.5336
and changed 74–83% of transfer rankings.  Thus this is not merely a dead GPU
or universally unlearnable generator: the proposed predictive-gain
compression failed to pay rent relative to ordinary candidate/history
interaction on the same task.

## Post-outcome construct caveat

The frozen generator also contains an unintended base shortcut.  A non-repeat
positive uses the smallest item variant absent from relevant history.  Although
the base cannot see that history, this makes variant 0 the positive in 84.97%
of a separately generated 13,025-request post-outcome diagnostic, versus 6.27%
for negatives; variants 0 or 1 cover 98.14% of positives.  This explains the
abnormally saturated 0.944–0.948 base NDCG and invalidates the absolute
synthetic-difficulty calibration.

The caveat does not rescue C10: centred attention shared the same shortcut and
still recovered about +0.053–0.055 NDCG, while the primary recovered none.
Therefore the relative nearest-neighbour failure remains informative, but the
gate must not be cited as a clean quantitative estimate of predictive-history
headroom.

## Lesson and next boundary

C10 compresses `(q,H)` versus `q-only` into one global vocabulary distribution
before gathering candidate tokens.  Candidate specificity then exists only in
the gather and token-embedding write.  The outcome is consistent with that
bottleneck discarding the event-level interaction that ordinary centred
attention learned.  More epochs or looser thresholds are not an authorized
repair.  A future, separately fingerprinted candidate would need a genuinely
different primitive—such as candidate-conditioned token prediction at the
event interface—and a generator whose base shortcut is removed before lock.

No real gate config/materializer was prepared because the preregistered
synthetic authorization condition did not pass.
