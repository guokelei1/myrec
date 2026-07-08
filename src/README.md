# Source Code

Shared project implementation lives under `src/myrec/`.

```text
src/myrec/data/          dataset readers, converters, standardized record schema
src/myrec/eval/          metrics, score dump readers, candidate-pool checks
src/myrec/models/        proposed-system model code (early prototyping)
src/myrec/baselines/     self-implemented baselines (B0a, B0b, B1, B7)
src/myrec/utils/         hashing, manifests, logging helpers
```

Upstream-code baselines (B4 RecBole, B5 KuaiSearch official, B6 PPS
classic) live under `baselines/`, not here.

Proposed-system source that has passed the C3/C5 motivation gates lives
under `systems/`.

Keep generated data, weights, checkpoints, and run outputs outside `src/`.
