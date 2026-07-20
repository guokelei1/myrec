# Shared source

`src/myrec/` contains the project-owned data adapters, frozen Q0--Q3 Qwen
baseline harness, W0 witness, shared evaluator, metrics, hashing, and JSONL
utilities used by the current Motivation.

All methods use the standardized record interface and shared evaluator.
Training and scoring code must not read confirmation or test qrels. New
Candidate-Contrast method code is authorized under
[`../experiments/motivation/candidate_contrast_architecture_plan.md`](../experiments/motivation/candidate_contrast_architecture_plan.md),
must use a distinct package/output identity, and must leave the frozen baseline
and mechanism implementations unchanged.
