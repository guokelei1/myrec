# Configs

Small, reviewable configuration only:

```text
configs/datasets/      reviewed dataset versions and sampling manifests
configs/baselines/     reviewed baseline boundary and tuning settings
configs/methods/       future reviewed survivor configs
configs/experiments/   future composite run configs
configs/env/           environment manifests
```

The old Lite/C0/B0–B9 configs are archived. The active dataset and baseline
config directories are currently empty except for README markers: E0 must
define the new Full track before any dataset or baseline config is frozen. No
proposed-system config belongs here before a Failure Card; future
hypothesis-local configs start under the authorized hypothesis tree and are
promoted only after review.
