# Configs

Reusable, reviewable experiment configuration. Keep files small.

```text
configs/datasets/      per-dataset paths, splits, sampling seeds
configs/baselines/     per-baseline hyperparameters
configs/methods/       proposed-system configs
configs/experiments/   composite experiment configs (dataset + method + eval)
```

Do not put credentials, private paths, or machine-specific settings here.
Use a local ignored `.env` file or command-line arguments for those values.
