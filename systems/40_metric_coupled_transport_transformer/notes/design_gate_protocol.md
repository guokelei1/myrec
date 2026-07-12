# C40 data-free design gate

Status: frozen before running `probe/run_design_gate.py`.

## D0 structural checks

All four modes use dimension 32, four heads, rank four, and exactly the same
parameter count and initial state hash per seed. The gate requires:

- finite deterministic forward/backward on CUDA;
- nonzero gradients for both low-rank factors in every mode;
- candidate permutation error at most `1e-6`;
- exact zero correction for no-history, absent-query, and repeat-present;
- primary identity loop assignment and a fixed-point-free shifted cycle;
- primary and all reductions differ on a constructed nondegenerate example.

## D1 planted recovery

A fixed synthetic teacher is a four-head metric-coupled transport model. It
generates undifferentiated semantic vectors, then defines one positive among
twelve candidates by its non-repeat transported score. Student modes receive
only vectors and labels; no mode ID, latent-factor ID, or category token exists.
Wrong history is a fixed cross-request permutation and is never used for
training. Three seeds train every mode for the same steps with full candidate
sets and listwise loss.

The primary must, in every seed:

1. exceed random/base NDCG@10 by at least `0.10`;
2. exceed `single_wide_coupled` and `shifted_loop` by at least `0.01`;
3. have clean-minus-wrong NDCG@10 at least `0.05`;
4. retain at most `0.50` of clean-over-base gain under wrong history;
5. keep exact no-history and repeat corrections at zero after training.

`selection_only` is reported and must be finite and active, but is not a D1
loss target because identity values are already a strong conditional solution.
It becomes a mandatory real-data control.

Any D0 or D1 failure closes C40 before repository-data training. Passing only
shows recovery when loop closure is the planted law; it does not validate real
ranking utility, novelty, or transfer.
