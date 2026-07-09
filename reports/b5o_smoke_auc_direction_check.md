# B5o Smoke AUC Direction Check

Date: 2026-07-09

Scope: cheap implementation check before any full B5o Stage A proxy run. This
does not count against the six-run official-alignment stop-loss budget.

## Trigger

The 2000-row materializer smoke produced official DNN test AUC 0.377851 after
one epoch. Because this is below 0.5, we checked score direction, label
semantics, and materializer label preservation before running a full proxy
alignment.

## Evidence

Artifact root:

- `artifacts/batch2b/b5o_auc_probe/auc_probe_summary.json`
- `artifacts/batch2b/b5o_auc_probe/auc_probe_scores.csv`

Official eval semantics at locked commit:

- `ranking/trainer.py` computes `roc_auc_score(labels, sigmoid(logits))`.
- Higher model score is treated as more positive.
- Official labels come from `ranking/datasets.py` as
  `is_clicked == 1 or is_purchased == 1`.

Manual check on the same smoke checkpoint/test rows:

| Quantity | Value |
|---|---:|
| Test rows | 320 |
| Positives | 18 |
| Negatives | 302 |
| Official test AUC | 0.377851 |
| Manual AUC | 0.377851 |
| Manual AUC with score reversed | 0.622149 |
| Manual AUC with label inverted | 0.622149 |
| Manual LogLoss | 0.643912 |
| Official LogLoss | 0.643912 |
| Mean score, positives | 0.467042 |
| Mean score, negatives | 0.470991 |

Materializer label preservation check against the first 2000 raw ranking rows:

| Quantity | Value |
|---|---:|
| Rows compared | 2000 |
| Key-field mismatches | 0 |
| Raw positive labels | 56 |
| Materialized positive labels | 56 |
| Clicked rows | 56 |
| Purchased rows | 1 |
| Clicked and purchased rows | 1 |
| Test split rows | 320 |
| Test positive labels | 18 |

Checked key fields:
`user_id`, `session_id`, `time_index`, `target_item_id`, `is_clicked`,
`is_purchased`.

## Conclusion

The smoke AUC is not caused by an official-evaluator score-direction reversal
or a materializer label inversion. Manual AUC exactly matches official AUC, and
the materializer preserves the raw click/purchase fields used by the official
label rule.

The low smoke AUC remains a warning signal, but the most likely explanation is
the tiny one-epoch smoke setup: only 320 test rows and 18 positive labels after
the provisional last-time split. Proceeding to DNN/DCNv2 proxy full runs is
acceptable, with full-run AUC direction monitored explicitly.
