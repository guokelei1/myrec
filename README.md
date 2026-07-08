# myrec

This repository is the compact working home for the
**Query-conditioned Personalized Product Ranking (PPS)** paper. It
supports the full local workflow for dataset download, baseline
evaluation, motivation experiments, proposed-system development, and
paper writing, while Git only stores the reproducible core.

The research scope is documented in [doc/](doc/).

## Scenario (Frozen)

```text
User issues a real product search query. Given the user's behavioral
history (click/purchase sequence with item plaintext text), candidate
item text/attributes, and the fixed exposed candidate set, produce a
personalized ranking of the candidate pool.
```

See [doc/10_direction_decision.md](doc/10_direction_decision.md) for the
full direction decision and [doc/11_experiment_and_dataset_plan.md](doc/11_experiment_and_dataset_plan.md)
for the 6-phase experiment plan with checkpoints C0–C5.

## Repository Contract

Commit these:

- source code, tests, scripts, and reusable configuration;
- proposed-system source under `systems/`;
- tracked baseline source trees, local patches, adapters, and notes under
  `baselines/`;
- research notes, protocol documents, and concise development logs;
- paper source, small curated tables, and final hand-selected figures.

Do not commit these:

- downloaded datasets or processed dataset records;
- model weights, checkpoints, embeddings, indexes, or caches;
- raw experiment logs, score dumps, tensorboard/wandb/mlflow output, or
  scratch files;
- credentials, API keys, tokens, private paths, or machine-specific
  settings.

## Directory Layout

| Path | Git policy | Purpose |
|---|---|---|
| `doc/` | tracked | research objective, direction decision, experiment plan, lessons, concise dev logs |
| `src/myrec/` | tracked | shared project implementation: data interfaces, evaluators, self-implemented baselines, model code, utils |
| `systems/` | tracked | proposed-system source (query-conditioned evidence routing) |
| `scripts/` | tracked | command-line entry points for download, preprocessing, training, scoring, evaluation |
| `configs/` | tracked | reproducible dataset, baseline, method, and experiment configs |
| `tests/` | tracked | unit/integration tests and tiny fixtures only |
| `baselines/` | tracked | upstream baseline code trees (KuaiSearch official, RecBole, PPS classic), local patches, adapters, and notes |
| `experiments/` | tracked | experiment plans, config templates, and short run manifests |
| `reports/` | tracked selectively | curated metrics tables, checkpoint audit JSONs, paper-ready summaries |
| `paper/` | tracked | manuscript source, bibliography, and small manually selected assets |
| `data/` | ignored except README | raw, intermediate, processed, and standardized dataset files |
| `models/` | ignored except README | downloaded weights, trained checkpoints, embeddings, and model caches |
| `runs/` | ignored except README | raw outputs from training, scoring, evaluation, sweeps, and logs |
| `artifacts/` | ignored except README | generated plots, temporary tables, exported predictions, packaged outputs |
| `tmp/` | ignored except README | disposable scratch space |

## Local Workflow

1. Put real datasets under `data/raw/<dataset>/`.
2. Write dataset converters in `src/myrec/data/` and runnable commands in
   `scripts/`.
3. Export standardized records under `data/standardized/<dataset>/<version>/`.
4. Put upstream baseline working copies under `baselines/<name>/` and
   track local patches there.
5. Develop the proposed system under `systems/`.
6. Put trained checkpoints and downloaded model weights under `models/`.
7. Write raw run outputs under `runs/<run_id>/`.
8. Promote only concise, paper-relevant summaries to `reports/`, `doc/`,
   or `paper/`.

## Experiment Naming

Use stable run identifiers so local files stay searchable:

```text
YYYYMMDD_<dataset_id>_<method_id>_<short_purpose>
```

Example:

```text
20260708_kuaisearch_bm25_motivation_m1
```

For every run that matters, keep the exact command, config path, git
commit, dataset version, checkpoint reference, and metric summary in
either `experiments/` or `doc/dev_log/`.

## Data Interface

The main standardized record contract is in
[doc/11_experiment_and_dataset_plan.md](doc/11_experiment_and_dataset_plan.md)
(Phase 1, §1.2). Generated `records_train.jsonl`, `records_dev.jsonl`,
`records_test.jsonl`, `item_catalog.jsonl`, and `manifest.json` files
belong under `data/` and are not tracked by Git.
