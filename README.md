# myrec

Compact working repository for the **Query-conditioned Personalized Product
Ranking (PPS)** paper.

## Current direction

The active scientific entry point is [`doc/motivation.md`](doc/motivation.md).
It records the recurrence–transfer research question, the frozen V1.2
first-round evidence, and the mechanism-derived design boundary. The active
work plan is
[`experiments/motivation/candidate_contrast_architecture_plan.md`](experiments/motivation/candidate_contrast_architecture_plan.md).

The frozen result is a preliminary motivation finding: the tested
history-conditioned Qwen rankers reliably respond on recurrence requests, but
strict transfer is not yet established. The mechanism inventory is complete;
the authorized next step is implementation and matched validation of the
Candidate-Contrast Personalization architecture on the same Qwen3-0.6B and
KuaiSearch boundary.

The active tree preserves the Motivation question, historical mechanism plans,
frozen evidence, reusable code/contracts, and the new architecture plan.
Deferred deep-dive queues remain provenance and are not active prerequisites.

## Active repository contract

Keep the reproducible core in:

- `src/myrec/`: shared data, baseline adapters, evaluation, and metrics;
- `scripts/`: current data preparation, baseline, and evaluator commands;
- `configs/`: reusable dataset, baseline, method, and environment configuration;
- `tests/`: shared contracts and hand-computed metric tests;
- `doc/`: current protocols and concise decisions;
- `experiments/motivation/`: the active architecture plan, historical mechanism plans, and frozen protocol;
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
