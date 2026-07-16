# Curated Motivation V1 reports

Only current V1 evidence and the direct frozen-confirmation audit chain are
kept here. Superseded exploratory reports and old figures were removed; their
raw scores, logs, checkpoints, code, and reusable configs remain outside this
directory.

## Current result

- `pps_three_transformer_history_surface_audit.json`: canonical Qwen3/TEM/
  InstructRec target-aware result;
- `history_response_gap_motivation_status.json`: concise current claim and
  remaining limitations;
- `pps_query_conditioned_baseline_comparison.json`: model and input boundaries.

## Direct frozen Qwen evidence

- `pps_motivation_confirmation_decision.json`: authoritative five-gate result;
- `pps_history_response_confirmation_score_integrity.json`: pre-label identity
  and completeness audit;
- `pps_history_response_confirmation_target_aware_surfaces.json`: target-aware
  surface statistics;
- `pps_history_response_confirmation_assignments.json`: frozen counterfactual
  assignments;
- `pps_history_response_confirmation_qwen3_preprocess_train.json` and
  `pps_history_response_confirmation_qwen3_preprocess_confirmation.json`:
  preprocessing and token-boundary evidence;
- `pps_history_response_e0_confirmation_data_admission.json`: frozen population
  admission.

`dev_eval_log.jsonl` and `confirmation_eval_log.jsonl` remain the evaluation
ledgers. No report in this directory authorizes test access or architecture
development.
