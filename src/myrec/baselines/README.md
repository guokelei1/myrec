# Active baseline source boundary

The existing modules are retained shared controls and upstream adapters. Some
still carry historical method IDs or `v0_lite` metadata defaults; they must not
be relabeled as E-QC/E-FULL or D-QC/D-FULL without a reviewed E0 data version,
new config, current metadata, and a source-specific test.

The active ordinary encoder/decoder family implementation should use the
counterfactual score contract under `experiments/history_response_gap/` and
keep one checkpoint fixed across true/null/wrong scoring. Proposed
architecture code does not belong here or under `systems/` before a replicated
Failure Card.

`representative_sequence_adapter.py` is the shared label-free record-to-sequence
boundary for the HSTU and LLM-SRec representative baselines. It fits categorical
vocabularies on training records only and appends the current query as the final
causal token, including for null-history requests.

Representative baseline modules:

- `hstu_pps_adapter.py` and `sequence_ranker_training.py`: official-core HSTU
  and same-tree SASRec under one matched fixed-slate PPS interface;
- `frozen_text_features.py`: qrels-free frozen content features needed for
  cold/new item coverage;
- `sequence_teacher_features.py`: frozen SASRec teacher item/user stores for
  LLM-SRec;
- `llm_srec_adapter.py` and `llm_srec_training.py`: independent implementation
  of the KDD 2025 paper mechanism and its explicit PPS task adaptation.

These are baseline/failure-localization sources. They are not proposed-system
code and do not unlock `systems/`.
