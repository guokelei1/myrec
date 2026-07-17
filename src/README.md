# Shared source

`src/myrec/` contains the project-owned data adapters, frozen Q0--Q3 Qwen
baseline harness, W0 witness, shared evaluator, metrics, hashing, and JSONL
utilities used by the current Motivation.

All methods use the standardized record interface and shared evaluator.
Training and scoring code must not read confirmation or test qrels. The current
mechanism plan authorizes isolated diagnostic probes and matched controls, not
a new transfer architecture.
