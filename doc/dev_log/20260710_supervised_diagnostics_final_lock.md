# Supervised Diagnostics Final Lock

Locked at: 2026-07-10T16:35:29+08:00, before any D1 supervised dev evaluation.

The train-only calibration selected D1q=1 epoch, D1m=3 epochs, and D1a=1
epoch under the predeclared `min_delta=0.0001` rule. All final models use seeds
20260708/20260709/20260710 and the unchanged hyperparameters in the base config.

Frozen final config:

- path: `configs/analysis/supervised_motivation_diagnostics_final.yaml`
- SHA-256: `32059922bf41e136982910757931a964aebff8f54c3290ba18b75f0e537ba6d7`

Nine final checkpoints had completed at lock time. Their training summaries
all state `qrels_read=false` and `test_read=false`. No supervised diagnostic
score file had been evaluated against dev qrels at lock time.

The label-free explanatory slices added to doc 18 section 5.1 were also frozen
at this time. They are descriptive only and cannot change the headline result
or trigger a retry.
