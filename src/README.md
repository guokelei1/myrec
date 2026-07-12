# Source Code

Shared project implementation lives under `src/myrec/`.

```text
src/myrec/data/          dataset readers, converters, standardized record schema
src/myrec/eval/          metrics, score dump readers, candidate-pool checks
src/myrec/baselines/     self-implemented baselines (B0a, B0b, B1, B7)
src/myrec/utils/         hashing, manifests, logging helpers
```

Upstream-code baselines (B4 RecBole, B5 KuaiSearch official, B6 PPS
classic) live under `baselines/`, not here.

C01--C80 are historical and no architecture candidate is active. Shared R0
code for full-token baselines, observability, power analysis, and Failure Cards
may live here when it is method-independent and reviewed.

Future Hxx source belongs under `systems/<hypothesis>/` from its first line,
but only after a doc/31 Failure Card passes. Components may be promoted here
after review establishes that they are shared infrastructure and promotion does
not erase hypothesis-specific ablation boundaries.

Keep generated data, weights, checkpoints, and run outputs outside `src/`.
