# Active scripts

Active shared entry points:

- `validate_standardized_records.py`: validates method-visible records without
  opening qrels;
- `evaluate_scores.py`: standard ranking evaluator;
- `analyze_history_response.py`: qrels-reading true/null/wrong direction
  evaluator with counterfactual identity checks;
- `audit_kuaisearch_source.py`: exploratory outcome-free KuaiSearch Lite/Full source,
  reconstructed-history, and exact-query collision opportunity audit;
- `prepare_kuaisearch_lite_scout.py`: source-train-only, time-split,
  label-isolated KuaiSearch source scout with recall-log causal history;
- `prepare_amazon_c4_standardized.py`: fresh Amazon-C4/history BM25-candidate
  stress-test materialization with current causal and label-isolation checks;
- `run_simple_baseline.py`: source-order, train-popularity, recent-history,
  and request-local BM25 exploratory controls;
- `materialize_request_surfaces.py`: label-free history, recurrence, and
  repeated-query request-ID surfaces and their pre-defined intersections;
- `materialize_request_manifest.py`: dataset-independent hashes for visible
  query and candidate identity used by shared scorers;
- `materialize_history_assignments.py`: causal true/null and deterministic
  query/length/action/time-matched wrong-user assignments;
- `materialize_assignment_surfaces.py`: label-free exact-query/global donor
  request-ID surfaces for provenance-specific reporting;
- `analyze_response_direction_intervention.py`: label-oracle diagnostic that
  preserves each request's observed score-delta multiset while changing only
  candidate attribution;
- `summarize_response_direction_intervention_surfaces.py`: clustered surface
  uncertainty for that diagnostic;
- `run_full_token_cross_encoder.py`: one fixed ordinary BGE cross-encoder
  condition using the counterfactual score-bundle metadata contract;
- `train_full_token_cross_encoder.py`: train-only ordinary QC/FULL pairwise
  or pointwise encoder reranker with matched examples and optimizer budgets;
- `train_decoder_cross_encoder.py`: train-only ordinary QC/FULL pointwise
  adaptation for the decoder-only Qwen3 reranker using its official prompt and
  raw yes-minus-no scoring interface;
- `train_recoverability_witness.py` and `run_recoverability_witness.py`:
  train-only frozen-embedding diagnostic used only to test whether strict
  nonrepeat direction is cheaply recoverable;
- `summarize_history_response_surfaces.py`: request/user/query-cluster
  uncertainty summaries over an already evaluated counterfactual bundle;
- `audit_full_token_coverage.py`: label-free tokenizer budget and truncation
  risk audit for query+history+candidate serialization;
- `fit_history_assignments_to_context.py`: label-free deterministic guard for
  `only_second` scoring; it materializes the effective recent-history suffix
  and drops the minimum number of oldest assigned events needed to preserve at
  least one candidate token;
- `compare_runs.py` and `eval_trec_ranklist.py`: shared comparison/adaptation
  utilities.
- `train_sequence_ranker.py` and `run_sequence_ranker.py`: matched official-core
  HSTU/SASRec QC/FULL training and label-free true/null/wrong scoring;
- `materialize_frozen_text_features.py`: qrels-free frozen text features shared
  by representative sequence baselines;
- `materialize_sequence_teacher_features.py`: frozen train-only SASRec item/user
  representations for the LLM-SRec mechanism, without dev qrels;
- `train_llm_srec.py` and `run_llm_srec.py`: independent paper-mechanism
  LLM-SRec lightweight training and fixed-checkpoint counterfactual scoring.

Source-specific exploratory audits and ordinary-model pilots may be added under
the active exploration manifest. They must log their scope, write shared
record/score contracts where applicable, and may not read qrels. Confirmation
commands still require a separately frozen admission/model-family card.

The old C01–C80 and R0/round1–5 scripts are archived. Do not revive them by
copying their locks or output paths.
