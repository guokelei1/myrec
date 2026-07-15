# Active research documents

The current result entry point is
[`35_controlled_history_composition_motivation.md`](35_controlled_history_composition_motivation.md).
It synthesizes the exploratory motivation evidence after the direction-gap
validation. For a detailed Chinese explanation of the terminology, accounting,
and interpretation, read
[`36_controlled_history_composition_reader_guide_zh.md`](36_controlled_history_composition_reader_guide_zh.md).

The pre-outcome experiment plan remains
[`34_history_response_direction_gap_validation_plan.md`](34_history_response_direction_gap_validation_plan.md).

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
| `34_history_response_direction_gap_validation_plan.md` | pre-outcome motivation validation plan |
| `35_controlled_history_composition_motivation.md` | current exploratory evidence synthesis and claim boundary |
| `36_controlled_history_composition_reader_guide_zh.md` | detailed Chinese reader guide |
| `37_representative_architecture_validation.md` | Qwen/HSTU/LLM-SRec representative-baseline matrix and execution boundary |

The former C01–C80, R0, and round1–5 documents are historical evidence. They
are outside the active document set and live under `archive/`.

New implementation/audit decisions belong in `dev_log/`; keep raw experiment
state outside `doc/`.
