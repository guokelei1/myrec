# AGENTS.md

These instructions apply to the whole repository.

## Current source of truth

`AGENTS.md` is a repository bootstrap, not the scientific plan. For the current
Motivation V1.2 work, read and follow these three files as the primary source of
truth:

1. `experiments/motivation_v1_2/execution_prompt_zh.md` -- requested work and
   completion point;
2. `experiments/motivation_v1_2/plan.md` -- methods, data flow, seed/compute
   policy, evaluation, and reporting;
3. `doc/43_llm_rerank_recurrence_transfer_research_logic_zh.md` -- scientific
   motivation, prior evidence, and hypothesis.

Use `doc/11_experiment_and_dataset_plan.md` and
`doc/12_experiment_execution_protocol.md` only as supporting technical
contracts for unified records, label isolation, run metadata, and shared
evaluation. Frozen V1/V1.1 documents (`doc/40*`, `doc/41*`) are prior evidence,
not current execution plans.

Older references to doc 34, E0--E8, Failure Cards, R0, C01--C80, or the former
`history_response_gap` controller are inactive and must not redirect or block
Motivation V1.2. Files that describe those workflows are retained only as
historical context unless the user explicitly reactivates them.

The current authorized stopping point is the first-round V1.2 implementation,
real evaluation, and concise evidence summary described by the execution
prompt. Do not proceed into a proposed architecture or a different dataset;
stop and wait for user direction after that summary.

## Repository purpose

This repository is the compact working home for the **Query-conditioned
Personalized Product Ranking (PPS)** paper. It supports local data preparation,
baseline implementation, motivation experiments, evaluation, and paper
writing. Git stores only the reproducible core.

Before adding a file, decide whether it is tracked source/protocol evidence or
local experiment state.

## What to track

- Project source under `src/myrec/`.
- Runnable scripts under `scripts/`.
- Reusable configuration under `configs/`.
- Tests and tiny fixtures under `tests/`.
- Baseline reference trees, adapters, patches, and provenance notes under
  `baselines/`.
- Experiment plans and short manifests under `experiments/`.
- Concise research notes under `doc/`, especially `doc/dev_log/`.
- Curated paper-ready results under `reports/` and manuscript files under
  `paper/`.

Do not add active source under `systems/` during Motivation V1.2.

## What not to track

- Downloaded, interim, processed, or standardized datasets. Keep them under
  `data/`.
- Model downloads, weights, checkpoints, embedding tables, indexes, and
  caches. Keep them under `models/` or `artifacts/`.
- Raw logs, score dumps, TensorBoard/W&B/MLflow output, sweeps, and run state.
  Keep them under `runs/`.
- Scratch files. Keep them under `tmp/`.
- Credentials, tokens, API keys, private paths, or machine-local settings.

## Current implementation policy

Follow the exact active methods and roles in the V1.2 plan; do not add, replace,
or outcome-select a method or seed.

Published papers and repositories may be downloaded and read, but active
variants should be independent minimal reimplementations of their load-bearing
mechanisms:

- do not adopt an upstream trainer, data pipeline, checkpoint manager, or
  private evaluator as the experiment framework;
- reuse the project's existing Qwen code, configs, checkpoints, score bundles,
  and evaluator when their data/candidate/history boundary is compatible;
- keep shared loading, field whitelisting, candidate scoring, checkpoint
  resume, and score export in project-owned code;
- isolate only the method-specific prompt, sampling, module, or loss;
- describe adaptations as `-style reimplementation`, not official
  reproductions;
- record source URL, commit, license, inspected files, migrated mechanism,
  omitted components, and local implementation in
  `experiments/pps_baseline_cards.md`.

Do not place downloaded weights, checkpoints, generated data, raw logs, or
score dumps inside tracked baseline source trees.

## Data and evidence safety

- All methods consume the unified standardized JSONL interface. A dataset
  adapter may construct it, but scoring code may not reopen raw data or add a
  method-only dataset branch.
- Use the KuaiSearch populations, candidate contracts, and holdout sequence in
  the active V1.2 plan. Do not open source test.
- Training records may contain supervision. Internal-dev/confirmation records
  are label-free, with qrels stored separately.
- `clicked`, `purchased`, and `relevance` may be training targets but must never
  be serialized as model inputs. Use an explicit input-field whitelist.
- Scoring/training code must not read confirmation or test qrels. Only the
  shared evaluator may open a frozen qrels file after score-bundle integrity
  checks.
- Assert candidate/request hashes and complete finite score coverage before
  evaluation.
- Use one shared evaluator for every method. A method may not produce its own
  paper-table metrics.
- Preserve all valid pilot seeds and contradictory outcomes. A mechanical,
  numerical, or under-converged run is a run-state diagnosis, not a transfer
  result.
- Follow the active plan for `full`, `null`, `wrong-user`, recurrence, strict
  transfer, overlap surfaces, uncertainty, and weighted contributions.

## Runs and reporting

Use run IDs of the form:

```text
YYYYMMDD_<dataset_id>_<method_id>_<short_purpose>
```

Raw state belongs in `runs/<run_id>/`. An important run promotes only a concise
tracked summary containing command, config, code revision, dataset/manifest
hash, seed, checkpoint reference, metrics, conclusion, and next action.

Seed staging, four-hour resumable job boundaries, and four-GPU scheduling are
defined only in `experiments/motivation_v1_2/plan.md`. Do not duplicate or
reinterpret those rules here. Every job must have an independent writable
output/checkpoint directory and a recorded lineage.

Append every internal-dev evaluation to `reports/dev_eval_log.jsonl`. Register
paper-table numbers in `experiments/pps_results.md`, copied verbatim from the
shared evaluator's `metrics.json`. Never overwrite a frozen report; add a new
versioned report or concise superseding summary.

## Engineering rules

- Prefer existing repository patterns over new abstractions.
- Keep changes scoped to the paper workflow unless the user requests broader
  refactoring.
- Preserve unrelated user changes in a dirty worktree.
- Use `rg` for search when available.
- Use structured parsers/configs instead of ad hoc text manipulation when
  practical.
- Add tests when changing shared behavior, evaluators, data conversion,
  checkpoint resume, pairwise/listwise conversion, or metric code. Metric tests
  require hand-computed assertions.
- Before finalizing, run relevant tests, check `git status --short`, and ensure
  ignored large-file areas are not accidentally staged or visible as untracked
  files.
