# doc

Tracked research constraints, protocols, decisions, and concise development
logs for the PPS project.

## Current Authority

C01--C80 architecture search ended without a validated proposed architecture.
C80 closed at a pre-label mechanical gate; its utility is unknown, its fresh
labels remain unopened, and there is no C81.

Current execution order:

```text
scope/data-object audit
  -> full-token observability parity
  -> normally tuned ordinary full-token baseline
  -> replicated Failure Card
  -> one architecture hypothesis
  -> bounded dev iteration
  -> frozen independent confirmation
  -> one-shot test
```

`31_problem_discovery_and_architecture_iteration_protocol.md` is the current
authority for this workflow. `24_parallel_llm4rec_design_protocol.md` and all
C01--C80 candidate-local protocols are historical records only.

## Key Documents

| File | Role |
|---|---|
| `07_paper_design_constraints.md` | Tier-1/Tier-2 evidence and paper-shape constraints |
| `10_direction_decision.md` | PPS scenario and conditional dataset roles |
| `11_experiment_and_dataset_plan.md` | Dataset, metric, split, checkpoint, and label-isolation plan |
| `12_experiment_execution_protocol.md` | Environment, run, resource, metadata, evaluator, and dev-log rules |
| `13_baseline_implementation_plan.md` | Baseline fairness matrix, tuning budgets, and developer runbook |
| `14_official_baseline_plan.md` | Official-code alignment, budgets, and stop-loss rules |
| `15_proposed_system_design_principles.md` | Current Failure-Card-to-architecture entry rules |
| `24_parallel_llm4rec_design_protocol.md` | Historical C01--C04 isolation/GPU protocol; no current authorization |
| `25_history_signal_observability_protocol.md` | Pooled history-source observability protocol |
| `27_amazon_history_signal_observability_protocol.md` | Amazon pooled-history observability protocol |
| `28_amazon_token_history_observability_protocol.md` | Ordinary full-token positive-control protocol |
| `30_amazon_token_edge_attribution_protocol.md` | Full-token Q/H/C edge attribution protocol |
| `31_problem_discovery_and_architecture_iteration_protocol.md` | Current discovery, training, feedback, and confirmation pipeline |
| `dev_log/20260712_c01_c80_terminal_retrospective.md` | C01--C80 causal retrospective and terminal boundary |

Documents 16--23 and candidate-specific documents remain useful historical
protocol evidence, but their old launch/authorization wording is superseded by
doc 31.

## Subdirectories

- `dev_log/` - chronological reasoning, incidents, outcomes, and decisions.
- `baseline_notes/` - baseline setup notes, provenance, and reproduction details.
- `review_prompts/` - bounded prompts for independent audits.
- `design_prompts/` - historical independent-agent design prompts; not current
  architecture authorization.
