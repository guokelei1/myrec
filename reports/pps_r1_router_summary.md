# PPS R1 Router Summary

Status: complete. R1 is a cheap control, not the proposed system.

## Results

- R1b LR dev run: `20260710_kuaisearch_r1b_router_lr_dev`
- R1b NDCG@10: `0.3072`
- Recovery ratio: `-0.2521`
- R1a CV dev run: `20260710_kuaisearch_r1a_router_cv_dev`
- R1a NDCG@10: `0.3106`
- R1a relative delta vs R1b: `0.0112`

## Comparisons

- R1b vs b7_bge: delta `-0.0234`, 95% CI `[-0.0266, -0.0201]`; R1b is significantly below the reference.
- R1b vs b0b: delta `-0.0067`, 95% CI `[-0.0123, -0.0011]`; R1b is significantly below the reference.
- R1b vs b2z: delta `0.0015`, 95% CI `[0.0006, 0.0025]`; R1b is significantly above the reference.

## Anti-Cheat Checks

- r1b_fit_read_qrels_dev: `False`
- dev_qrels_read_only_by_shared_evaluator: `True`
- features_same_generation_as_m4: `True`
- candidate_hash_asserted_by_shared_evaluator: `True`
- channel_score_configs_match_m3_inputs: `True`

## Low-Recovery Diagnostic

- status: `completed`
- conclusion: No deterministic feature/metric mismatch was found. The official R1b LR argmax router collapses mostly to query_b2z, does not select the static channel, and remains below B7-bge. R1 is therefore retained as a weak cheap control rather than repaired by post-hoc thresholding.
- train oracle label rates: `{'history_b0b': 0.3389, 'query_b2z': 0.6189, 'static_b7_bge': 0.0422}`
- dev oracle label rates: `{'history_b0b': 0.3505601439201897, 'query_b2z': 0.6062637991659171, 'static_b7_bge': 0.0431760569138932}`
- R1b selected counts: `{'history_b0b': 259, 'query_b2z': 11970}`
- R1a selected counts: `{'history_b0b': 782, 'query_b2z': 11447}`

## Dev-Eval Reconciliation

R1b and R1a each have two retained evaluator entries: the initial evaluation
that triggered the frozen low-recovery branch and the single doc/16 §5.2
recheck after diagnostic reporting was added. Both NDCG@10 values are identical
to full precision, one hyperparameter configuration was used, and no third
evaluation occurred. The first score snapshots were overwritten, so byte-level
identity cannot be claimed. See `reports/pps_r1_dev_eval_reconciliation.json`.
