# Configs

Reusable, reviewable experiment configuration. Keep files small.

```text
configs/datasets/      per-dataset paths, splits, sampling seeds
configs/baselines/     per-baseline hyperparameters
configs/methods/       reviewed configs promoted from a selected system
configs/experiments/   composite experiment configs (dataset + method + eval)
```

Do not put credentials, private paths, or machine-specific settings here.
Use a local ignored `.env` file or command-line arguments for those values.

C01--C80 are closed. Current configs support doc/31 R0 information-object
audits, full-token observability, strong-baseline tuning, and failure discovery.
There is no active four-track architecture round.

Before a Failure Card passes, do not create a new architecture config. Future
Hxx exploratory configs stay under `systems/<hypothesis>/configs/`; only a
reviewed survivor may be promoted to `configs/methods/` or
`configs/experiments/`. Every score-affecting Txxx config records its change
class and dev-call index.
