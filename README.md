# myrec

Compact working repository for the **Query-conditioned Personalized Product
Ranking (PPS)** paper. It supports dataset preparation, baseline evaluation,
problem discovery, proposed-system development, confirmation, and paper
writing while Git stores only the reproducible core.

## Current Stage

C01--C80 architecture search is terminally closed. C80 failed a pre-label
event-permutation contract, so its 365 fresh labels remain unopened and its
ranking utility is unknown. There is no C81 or C80 rescue.

The active workflow is a research reset:

1. audit the information objects and confirmation data across datasets;
2. reproduce full-token history observability on KuaiSearch and Amazon;
3. normally tune an ordinary full-token joint Transformer as the strong base;
4. discover and replicate a ranking-relevant failure of that strong base;
5. only then derive an architecture hypothesis, train it on dev, and freeze an
   independent confirmation.

The scientific process is
[doc/31](doc/31_problem_discovery_and_architecture_iteration_protocol.md); its
continuous autonomous execution and end states are defined by
[doc/32](doc/32_autonomous_pipeline_controller.md).
[doc/15](doc/15_proposed_system_design_principles.md) defines architecture
entry and eligibility. [doc/24](doc/24_parallel_llm4rec_design_protocol.md) is
historical and does not authorize current work. Test remains locked.

The C01--C80 causal record is
[the terminal retrospective](doc/dev_log/20260712_c01_c80_terminal_retrospective.md),
and the candidate ledger remains under [systems/](systems/).

## Scenario

```text
User issues a real product search query. Given the user's strictly prior
behavioral history, candidate text/attributes, and a fixed exposed candidate
set, rank the candidates for that request.
```

KuaiSearch remains the nominal main track and Amazon-C4 the text-rich secondary
track. JDsearch is a conditional robustness anchor only for claims supported by
its no-plaintext information object; it is not automatically a gate for a
token-semantic mechanism.

## Repository Contract

Track:

- source under `src/myrec/`;
- historical and future hypothesis-local system source under `systems/`;
- runnable scripts under `scripts/` and reusable configs under `configs/`;
- tests and tiny fixtures under `tests/`;
- upstream baseline code, adapters, and notes under `baselines/`;
- protocols and concise reasoning under `doc/`;
- experiment manifests under `experiments/`;
- curated results under `reports/` and manuscript files under `paper/`.

Do not track:

- downloaded, standardized, or processed datasets;
- checkpoints, model weights, embeddings, indexes, or caches;
- raw logs, score dumps, sweeps, or scratch state;
- credentials, tokens, private paths, or machine-local settings.

## Directory Layout

| Path | Git policy | Purpose |
|---|---|---|
| `doc/` | tracked | constraints, protocols, decisions, and concise dev logs |
| `src/myrec/` | tracked | reviewed shared data, evaluation, baseline, and analysis code |
| `systems/` | tracked | C01--C80 history and future hypothesis-local source after doc/31 authorization |
| `scripts/` | tracked | runnable download, preparation, training, scoring, and audit commands |
| `configs/` | tracked | reusable dataset, baseline, analysis, and experiment configs |
| `tests/` | tracked | unit/integration tests and tiny fixtures |
| `baselines/` | tracked | upstream baseline trees, adapters, manifests, and patches |
| `experiments/` | tracked | plans, Failure Cards, proposals, and short run manifests |
| `reports/` | selective | audit JSON, curated metrics, and paper-ready summaries |
| `paper/` | tracked | manuscript source and small selected assets |
| `data/` | ignored | raw, interim, processed, and standardized data |
| `models/` | ignored | downloaded and trained model state |
| `runs/` | ignored | raw training, scoring, evaluation, sweep, and log output |
| `artifacts/` | ignored | generated analysis and exported predictions |
| `tmp/` | ignored | disposable scratch state and evaluator locks |

## Local Workflow

1. Put real datasets under `data/raw/<dataset>/`.
2. Convert them through `src/myrec/data/` and `scripts/` into the unified,
   label-isolated interface under `data/standardized/<dataset>/<version>/`.
3. Run the common audits, candidate-hash checks, and metric tests.
4. Train/evaluate baselines through the shared evaluator and log every dev call.
5. Execute doc/31 R0 observability, strong-baseline, and failure-discovery work.
6. Create `systems/<hypothesis>/` only after a Failure Card passes. Keep
   hypothesis, implementation, dev-trial, and confirmation IDs separate.
7. Keep checkpoints in `models/`, run state in `runs/`, and generated analysis
   in `artifacts/`.
8. Promote only concise, reproducible summaries to tracked files.

## Run Naming

```text
YYYYMMDD_<dataset_id>_<method_id>_<short_purpose>
```

Each important run records its command, config, code state, dataset/manifest
hash, checkpoint reference, random seed, environment, metric summary, and
remaining development budget.

## Evidence Boundary

- Training/scoring code never reads `qrels_dev.jsonl` or `qrels_test.jsonl`.
- All methods export `scores.jsonl` and use the same evaluator.
- Candidate-set hashes are asserted before every evaluation.
- Every dev evaluation is appended to `reports/dev_eval_log.jsonl`.
- Mechanics, learnability, utility, specificity, attribution, numerical safety,
  and novelty are separate conclusions.
- Confirmation is frozen before outcome access; test is run once only after the
  complete method/config is frozen.
