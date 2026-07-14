# Active research documents

The current entry point is
[`34_history_response_direction_gap_validation_plan.md`](34_history_response_direction_gap_validation_plan.md).
It is the scientific plan for the history-response direction-gap validation.

The short operational contracts are:

| Document | Role |
|---|---|
| `07_paper_design_constraints.md` | evidence hygiene and claim boundaries |
| `10_direction_decision.md` | current scope and exclusions |
| `11_experiment_and_dataset_plan.md` | record, split, metric, and phase contract |
| `12_experiment_execution_protocol.md` | run, logging, determinism, and label boundaries |
| `13_baseline_implementation_plan.md` | E1/E2 baseline roles and bounded tuning |
| `15_proposed_system_design_principles.md` | architecture authorization after a Failure Card |
| `31_problem_discovery_and_architecture_iteration_protocol.md` | Failure Card and later hypothesis gate |
| `32_autonomous_pipeline_controller.md` | persistent execution state after protocol freeze |

The former C01–C80, R0, and round1–5 documents are historical evidence. They
are outside the active document set and live under `archive/`.

New implementation/audit decisions belong in `dev_log/`; keep raw experiment
state outside `doc/`.
