# Active research documents

The current empirical entry point is
[`40_transformer_recurrence_transfer_motivation_v1_zh.md`](40_transformer_recurrence_transfer_motivation_v1_zh.md).
It contains the three-model result, interpretation boundary, and remaining
robustness obligations. Superseded motivation plans and duplicate audits were
removed after V1 consolidation.

## Governing contracts

| Document | Role |
|---|---|
| `07_paper_design_constraints.md` | evidence hygiene and claim boundaries |
| `10_direction_decision.md` | current scope, dataset roles, and exclusions |
| `11_experiment_and_dataset_plan.md` | record, split, metric, and phase contract |
| `12_experiment_execution_protocol.md` | run, logging, determinism, and label boundaries |
| `13_baseline_implementation_plan.md` | baseline roles, fairness, and tuning budgets |
| `15_proposed_system_design_principles.md` | architecture-entry constraints |
| `31_problem_discovery_and_architecture_iteration_protocol.md` | Failure Card and hypothesis gate |
| `32_autonomous_pipeline_controller.md` | persistent execution-state contract |
| `40_transformer_recurrence_transfer_motivation_v1_zh.md` | current Motivation V1 claim and evidence index |

Only the latest consolidation note is retained in `dev_log/`. Raw runs,
checkpoints, scores, and standardized data remain in their ignored local
directories.
