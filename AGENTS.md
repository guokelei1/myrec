# AGENTS.md

These instructions apply to the whole repository.

## Purpose

This repo is the compact working home for the **Query-conditioned
Personalized Product Ranking (PPS)** paper. It supports the full local
workflow for dataset download, baseline evaluation, motivation
experiments, proposed-system development, and paper writing, while Git
only stores the reproducible core.

The research scope is documented in [doc/](doc/). Key documents:

- `doc/07_paper_design_constraints.md` — paper design constraints (Tier 1
  rules that govern all experiments).
- `doc/10_direction_decision.md` — current bounded direction and dataset
  roles: KuaiSearch Full main, JDsearch independent functional replication,
  and Amazon-C4 English semantic positive boundary.
- `doc/11_experiment_and_dataset_plan.md` — active unified-record, split,
  label-isolation, E0–E8, and metric contract.
- `doc/12_experiment_execution_protocol.md` — run boundaries, dev-eval
  logging, determinism checks.
- `doc/13_baseline_implementation_plan.md` — per-baseline implementation,
  the fairness input matrix (§2.4), quantified tuning budgets (§2.5),
  per-baseline deliverables (§2.6), and the developer runbook (§7).
- `doc/15_proposed_system_design_principles.md` — current architecture-entry
  rules after the C80 terminal retrospective.
- `doc/31_problem_discovery_and_architecture_iteration_protocol.md` — current
  authoritative pipeline for source observability, strong-baseline development,
  failure discovery, architecture formulation, dev iteration, and confirmation.
- `doc/32_autonomous_pipeline_controller.md` — autonomous loop, persistent
  state, feedback transitions, recovery, budgets, and whole-pipeline end states.
- `doc/40_transformer_recurrence_transfer_motivation_v1_zh.md` — current
  empirical motivation, three-model result, claim boundary, and remaining
  robustness obligations.

Before adding a file, decide whether it is source/protocol evidence or
local experiment state.

## What To Track

- Track project source under `src/myrec/`.
- Track proposed-system source under `systems/`.
- Track runnable project scripts under `scripts/`.
- Track reusable configuration under `configs/`.
- Track tests and tiny fixtures under `tests/`.
- Track baseline source trees, project patches, adapters, and notes under
  `baselines/`.
- Track experiment plans, short manifests, and config templates under
  `experiments/`.
- Track concise research notes under `doc/`, especially `doc/dev_log/`.
- Track curated paper-ready results under `reports/` and manuscript files
  under `paper/`.

## What Not To Track

- Do not track downloaded datasets, standardized records, or processed
  data. Keep them under `data/`.
- Do not track generated baseline datasets, checkpoints, caches, logs,
  score dumps, or model downloads. Keep those under `data/`, `models/`,
  `runs/`, or `artifacts/`.
- Do not track model weights, checkpoints, embedding tables, vector
  indexes, or caches. Keep them under `models/` or `artifacts/`.
- Do not track raw logs, tensorboard/wandb/mlflow output, score dumps, or
  sweep directories. Keep them under `runs/`.
- Do not track scratch files. Keep them under `tmp/`.
- Do not track credentials, tokens, API keys, private paths, or
  machine-local settings.

## Baseline Policy

The active method roles are defined in
[doc/13_baseline_implementation_plan.md](doc/13_baseline_implementation_plan.md):
strong query-candidate bases, ordinary full-token encoder/decoder families,
simple/static controls, a traditional personalized control, and a train-only
signal witness. Historical B0–B9 names remain only where an existing source
adapter or provenance card still uses them.

**Self-implemented baselines** (B0a Popularity, B0b Recent-behavior, B1
BM25, B7 Static mixture) live as code under `src/myrec/baselines/`. They
read from the same unified JSONL interface as every other method.

**Upstream-code baselines** go directly in:

```text
baselines/<baseline_name>/
```

Current upstream baselines:

| Dir | Baseline | Source |
|---|---|---|
| `baselines/kuaisearch_official/` | B5: DIN / DCNv2 | KuaiSearch official repo |
| `baselines/recbole/` | B4: SASRec / BERT4Rec | RecBole framework |
| `baselines/pps_classic/` | B6: HEM / ZAM / TEM | PPS classic paper code |

For each upstream baseline, record in a tracked README or manifest:

- upstream repository URL;
- upstream commit hash;
- license notes;
- environment and setup commands;
- expected input/output format;
- any local patch summary.

Do not put checkpoints, downloaded model weights, baseline-generated
datasets, raw logs, or score dumps inside tracked baseline code. Register
each baseline's boundary card (official code / adapter-only / structural
change / zero-shot) in `experiments/pps_baseline_cards.md`.

## Proposed-System Policy

Architecture search ended at C80. There is no C81 and no C80 precision,
canonicalization, threshold, label-opening, dev, or test rescue. The C01--C80
trees under `systems/` are historical evidence, not active templates.

Current work follows the bounded Motivation V1 outcome in
`doc/40_transformer_recurrence_transfer_motivation_v1_zh.md`. V1 does not
authorize a new architecture. Any later proposed-system step must first freeze
and validate a Failure Card under
`doc/31_problem_discovery_and_architecture_iteration_protocol.md`
under the autonomous controller in `doc/32_autonomous_pipeline_controller.md`:

1. audit the cross-dataset information objects and confirmation data;
2. establish full-token observability on the main track and a comparable
   secondary track;
3. normally tune an ordinary full-token joint Transformer as the strong base;
4. produce a replicated Failure Card that localizes a ranking-relevant failure;
5. only then formulate one architecture hypothesis derived from that failure;
6. separate implementation revisions, dev trials, and frozen confirmation.

No new architecture source tree or architecture GPU training is authorized
until a Failure Card passes doc 31. A future proposed model must still be an
LLM4Rec-style Transformer/LM ranker in which the LM/Transformer is the
end-to-end ranking core. Prompt-only scoring, offline LLM features fed to an
MLP, fixed-score routing, and renamed existing attention modules are not
eligible architecture contributions.

For AI execution, keep tracked work products concise and evidence-oriented.
Each R0 iteration records hypothesis, single change, result, next action, and
budget. Keep at most three active failure ideas and probe at most the top two;
use the cheapest reversible discriminating probe first. A CPU/tiny-data
disposable prototype is allowed under `tmp/r0_prototypes/` before a Failure
Card, but it may not read evaluation labels, call the evaluator, enter
`systems/`, or support a claim.

Mechanics, learnability, utility, specificity, attribution, numerical safety,
and novelty are separate states. A mechanical failure is not negative utility;
normal dev tuning is allowed inside a frozen budget; paper-level joint gates
apply only to a frozen confirmation survivor. Controls are claim-specific:
event permutation is binding only for an order/set-invariance claim, wrong-user
history only for provenance, and no-history exactness only for a base-preserving
claim.

Track source code, configs, and design notes. Do not track
checkpoints, runs, logs, or caches — the `.gitignore` catches those under
`systems/**/checkpoints/`, `systems/**/runs/`, etc.

## Dataset Policy

Use this local layout:

```text
data/raw/<dataset_id>/
data/interim/<dataset_id>/
data/processed/<dataset_id>/
data/standardized/<dataset_id>/<version>/
```

Dataset tracks:

| Track | Dataset | Role |
|---|---|---|
| Main | KuaiSearch Full | Binding natural-query validation after E0 admission |
| Replication | KuaiSAR Full | Functional behavioral replication; no plaintext-semantic claim |
| Fallback | JDsearch | Pre-registered replacement if KuaiSAR cannot pass its source boundary |
| Non-binding | Amazon-C4 + history companion | Semantic stress test and optional legacy pilot only |

The standardized record interface is defined in
[doc/11_experiment_and_dataset_plan.md](doc/11_experiment_and_dataset_plan.md)
(Phase 1, §1.2). All methods read from this single JSONL interface — no
`if dataset == X` branches are allowed (only `if masks.history_present`
evidence conditions).

Dataset manifests, checksums, and provenance summaries may be promoted to
`doc/`, `experiments/`, or `reports/` only when they are small and useful
for reproduction.

## Experiment Policy

Use run IDs in this form:

```text
YYYYMMDD_<dataset_id>_<method_id>_<short_purpose>
```

Example:

```text
20260708_kuaisearch_bm25_motivation_m1
```

Raw run state belongs in:

```text
runs/<run_id>/
```

For important runs, promote a concise tracked summary containing:

- command;
- config path;
- git commit;
- dataset version or manifest hash;
- checkpoint or external model reference;
- key metrics;
- short conclusion and next action.

Large logs remain in `runs/`. Development reasoning and decisions belong
in `doc/dev_log/`.

### Phase Reports

Each active phase (E0–E8) promotes only a small reviewed audit or decision
report under `reports/`, for example:

```text
reports/pps_history_response_e0_data_admission.json
reports/pps_history_response_e3_activity.json
reports/pps_history_response_e8_failure_card.json
```

These are tracked (small JSON). A positive claim may advance only after its
phase decision passes. A failed premise closes or narrows the claim; a later
phase may not rescue it with a new slice.

## Engineering Rules

- Prefer existing repository patterns over new abstractions.
- Keep changes scoped to the paper workflow unless the user asks for
  broader refactoring.
- Do not rewrite or remove user changes unless explicitly requested.
- Use `rg` for search when available.
- Use structured parsers or configs instead of ad hoc text manipulation
  when practical.
- Add tests when changing shared behavior, evaluators, data conversion,
  or metric code. Metric code must have unit tests with hand-computed
  assertions (doc 11, C1 checkpoint).
- Before finalizing, check `git status --short` and ensure ignored
  large-file areas are not accidentally staged or visible as untracked
  files.
- All methods must be evaluated by the same evaluation script — no
  method ships its own evaluation code (doc 11, C2 checkpoint).
- Candidate-set hashes must be asserted before every evaluation (doc 11,
  C2 checkpoint).
- Scoring/training code must never read `qrels_dev.jsonl` /
  `qrels_test.jsonl`; dev/test records are label-free by construction
  (doc 11 §1.2). Violations invalidate all runs of that method.
- Every dev evaluation is appended to `reports/dev_eval_log.jsonl` and
  reconciled against the tuning budget in doc 13 §2.5.
- Paper-table numbers are registered only in `experiments/pps_results.md`,
  copied verbatim from evaluator `metrics.json`.
