# Configs

Small, reviewable configuration only:

```text
configs/datasets/      reviewed dataset versions and sampling manifests
configs/baselines/     reviewed baseline boundary and tuning settings
configs/methods/       current Motivation method configs
configs/env/           environment manifests
```

The `kuaisearch_motivation_v12_*` files are frozen first-round method and
witness configs. Mechanism probes must use separate configs and may not mutate
these evidence identities. Superseded Lite/C0/B0–B9 and V1/V1.1 configs were
removed. Proposed architecture configs remain outside the current stage.
