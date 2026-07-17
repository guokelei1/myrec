# myrec

Compact working repository for the **Query-conditioned Personalized Product
Ranking (PPS)** paper.

## Current direction

The active scientific entry point is [`doc/motivation.md`](doc/motivation.md).
It records the recurrence–transfer research question, the frozen V1.2
first-round evidence, the claim boundary, and the current mechanism question.
The active work plan is
[`experiments/motivation/mechanism_analysis_plan.md`](experiments/motivation/mechanism_analysis_plan.md).

The frozen result is a preliminary motivation finding: the tested
history-conditioned Qwen rankers reliably respond on recurrence requests, but
strict transfer is not yet established. The authorized next step is mechanism
analysis across signal availability, history selection, preference abstraction,
candidate readout, training dynamics, and population stability. It is not yet
a proposed architecture.

Only the current Motivation question, mechanism plan, frozen first-round
evidence, reusable code/contracts, and runtime inputs are kept in the active
tree. Superseded execution plans and experiment controllers are removed.

## Active repository contract

Keep the reproducible core in:

- `src/myrec/`: shared data, baseline adapters, evaluation, and metrics;
- `scripts/`: current data preparation, baseline, and evaluator commands;
- `configs/`: reusable dataset, baseline, method, and environment configuration;
- `tests/`: shared contracts and hand-computed metric tests;
- `doc/`: current protocols and concise decisions;
- `experiments/motivation/`: the current mechanism plan and frozen first-round protocol;
- `reports/`: current curated results and the development ledger;
- `paper/`: manuscript source after the evidence boundary is settled.

Downloaded data, checkpoints, caches, raw runs, and generated artifacts remain
ignored under `data/`, `models/`, `runs/`, `artifacts/`, and `tmp/`.

## Evidence invariants

- training/scoring code never reads confirmation or test qrels;
- every method uses the shared evaluator and asserts candidate-set hashes;
- every dev evaluation is logged in `reports/dev_eval_log.jsonl`;
- recurrence, strict transfer, overlap, contribution, and uncertainty remain
  separate conclusions;
- source test stays locked until a new protocol explicitly authorizes it.
