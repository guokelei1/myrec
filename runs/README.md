# Runs

Raw experiment output. Ignored by Git except for this README.

One directory per run:

```text
runs/YYYYMMDD_<dataset_id>_<method_id>_<short_purpose>/
```

Typical local contents:

- copied resolved config;
- stdout/stderr logs;
- raw metric JSON;
- score dumps;
- profiler output;
- tensorboard/wandb/mlflow exports;
- temporary predictions.

## Boundary with artifacts/ and reports/

| Directory | Content |
|---|---|
| `runs/` | raw output straight from training/scoring/evaluation |
| `artifacts/` | post-processed intermediate products (plots, exported tables, converted predictions) |
| `reports/` | curated final results selected for the paper |

Promote only concise summaries to `experiments/`, `reports/`, or
`doc/dev_log/`.
