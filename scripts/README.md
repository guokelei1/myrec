# Motivation baseline and shared scripts

These are the frozen V1.2 baseline and reusable shared entry points:

- `download_kuaisearch_full.py`: acquire the permitted KuaiSearch source;
- `validate_standardized_records.py`: validate label-free standardized records;
- `materialize_request_manifest.py`: build request identity manifests;
- `materialize_history_assignments.py`: build true/null/matched-wrong history assignments;
- `build_motivation_v12_release_lock.py`: freeze method/checkpoint identities;
- `materialize_motivation_v12_kuaisearch_holdout.py`: build the recipe-locked 4k holdout;
- `train_motivation_v12_ranker.py` / `score_motivation_v12_ranker.py`: train and score Q0--Q3;
- `materialize_frozen_text_features.py`: materialize the qrels-free W0 feature store;
- `train_copps_transfer_witness.py` / `score_copps_transfer_witness.py`: train and score W0;
- `evaluate_motivation_v12_evidence.py`: run the shared audit, evaluator, and summary path.

First-round result-producing commands bind the frozen
[`protocol.yaml`](../experiments/motivation/protocol.yaml). Historical mechanism
probes remain governed by the frozen
[`mechanism analysis plan`](../experiments/motivation/mechanism_analysis_plan.md).
New method scripts must follow the active
[`Candidate-Contrast architecture plan`](../experiments/motivation/candidate_contrast_architecture_plan.md),
use new run identities, and leave frozen outputs unchanged. No script opens
source-test qrels during training or scoring.
