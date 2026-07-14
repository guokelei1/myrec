# myrec

Compact working repository for the **Query-conditioned Personalized Product
Ranking (PPS)** paper.

## Current direction

The active scientific plan is
[`doc/34_history_response_direction_gap_validation_plan.md`](doc/34_history_response_direction_gap_validation_plan.md).
It asks whether ordinary full-token rerankers produce candidate-relative
history responses whose direction is reliably correct. It is a validation
plan, not an architecture authorization: no new model tree, training run, or
dev/test label opening is allowed before the E0 admission protocol passes and
the plan's evidence chain supports a Failure Card.

C01–C80 and R0/round1–5 are closed historical work. Their source, reports,
logs, and generated outputs are preserved under the dated legacy archive and
are not active templates. See [`archive/README.md`](archive/README.md).

## Active repository contract

Keep the reproducible core in:

- `src/myrec/`: shared data, baseline adapters, evaluation, and metrics;
- `scripts/`: only current data preparation, baseline, and evaluator commands;
- `configs/`: reusable dataset, baseline, and environment configuration;
- `baselines/`: upstream baseline trees and adapters;
- `tests/`: shared contracts and hand-computed metric tests;
- `doc/`: current protocols and concise decisions;
- `experiments/history_response_gap/`: the new direction's locks and manifests;
- `reports/`: only new-direction audits and curated results;
- `paper/`: manuscript source after evidence exists.

Downloaded data, checkpoints, caches, raw runs, and generated artifacts remain
ignored under `data/`, `models/`, `runs/`, `artifacts/`, and `tmp/`.

## First authorized work

1. Review doc 34 and freeze a short E0 source/collision/power protocol.
2. Audit KuaiSearch Full and the pre-registered fallback data object without
   training or reading dev/test labels.
3. Only after admission, establish the E-QC/D-QC and E-FULL/D-FULL ordinary
   reranker families and the shared true/null/matched-wrong instrumentation.
4. Do not create `systems/<hypothesis>/` until a replicated Failure Card has
   passed [`doc/31_problem_discovery_and_architecture_iteration_protocol.md`](doc/31_problem_discovery_and_architecture_iteration_protocol.md).

## Evidence invariants

- training/scoring code never reads `qrels_dev.jsonl` or `qrels_test.jsonl`;
- every method uses the shared evaluator and asserts candidate-set hashes;
- every dev evaluation is logged in `reports/dev_eval_log.jsonl`;
- response, direction, utility, specificity, and data sufficiency are separate
  conclusions;
- confirmation/test remain locked until model, cohort, endpoint, and analysis
  rules are frozen.
