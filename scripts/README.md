# Active scripts

Active shared entry points:

- `validate_standardized_records.py`: validates method-visible records without
  opening qrels;
- `evaluate_scores.py`: standard ranking evaluator;
- `analyze_history_response.py`: qrels-reading true/null/wrong direction
  evaluator with counterfactual identity checks;
- `compare_runs.py` and `eval_trec_ranklist.py`: shared comparison/adaptation
  utilities.

New source-specific data preparation and model execution commands should be
added only after the corresponding E0/model-family card is reviewed. They must
write the shared record/score contracts and may not read qrels.

The old C01–C80 and R0/round1–5 scripts are archived. Do not revive them by
copying their locks or output paths.
