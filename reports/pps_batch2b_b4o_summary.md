# B4o RecBole SASRec Summary

Date: 2026-07-09

Status: accepted official RecBole B4o run for Batch 2b.

## Boundary

- Implementation: RecBole `SASRec`, package `recbole==1.2.1`.
- Environment: `pps-recbole`, frozen in `configs/env/recbole.txt`.
- Config: `configs/baselines/b4o_sasrec_recbole.yaml`.
- Training input: `artifacts/batch2b/interactions_train.jsonl`, SHA256
  `dcc91391c303e6e8c69506a837620be2601bdd6df249cda624f4d1caf417ad32`.
- Scoring input: blind `records_dev.jsonl`; inference history is each record's
  frozen history, capped at 50 events.
- qrels read by method: false.
- Shared evaluator candidate manifest:
  `94eb667000e0d0f389d0a2a4d4730683b71c129043edbfcf627590376e9c123e`.

## External Sanity

`reports/b4o_env_sanity.md` passed on RecBole's bundled ml-100k example. SASRec
loss decreased from 336.5817 to 299.0741 over 3 epochs, and the example-split
test NDCG@10 was 0.0358.

## Vocab Coverage

`reports/b4o_vocab_coverage.md` records a high cold-start rate:

| Quantity | Value |
|---|---:|
| Dev candidate row in-vocab rate | 22.2043% |
| Dev candidate cold rate | 77.7957% |
| Zero-in-vocab candidate request rate | 13.3453% |

B4o keeps the documented cold-start policy rather than adding untrained dev
candidate embeddings or hybrid text/popularity fallback.

## Tuning Runs

| Run | Change | NDCG@10 |
|---|---|---:|
| `20260709_kuaisearch_b4o_sasrec_recbole_dev_s20260708` | RecBole SASRec defaults | 0.2967 |
| `20260709_kuaisearch_b4o_sasrec_recbole_t01_len20_dev_s20260708` | max sequence length 20 | 0.2967 |
| `20260709_kuaisearch_b4o_sasrec_recbole_t02_drop02_dev_s20260708` | dropout 0.2 | 0.2963 |
| `20260709_kuaisearch_b4o_sasrec_recbole_t04_lr0005_dev_s20260708` | learning rate 0.0005 | 0.2965 |
| `20260709_kuaisearch_b4o_sasrec_recbole_t03_h128_dev_s20260708` | hidden size 128, heads 4 | 0.2976 |
| `20260709_kuaisearch_b4o_sasrec_recbole_t05_l1_dev_s20260708` | one transformer layer | 0.2975 |

Frozen config: hidden size 128, heads 4, max sequence length 50, RecBole CE loss.

## Frozen Seeds

| Seed | Run | NDCG@10 | MRR | Recall@10 | pNDCG@10 |
|---:|---|---:|---:|---:|---:|
| 20260708 | `20260709_kuaisearch_b4o_sasrec_recbole_t03_h128_dev_s20260708` | 0.2976 | 0.2788 | 0.5169 | 0.3236 |
| 20260709 | `20260709_kuaisearch_b4o_sasrec_recbole_h128_dev_s20260709` | 0.2967 | 0.2769 | 0.5163 | 0.3200 |
| 20260710 | `20260709_kuaisearch_b4o_sasrec_recbole_h128_dev_s20260710` | 0.2972 | 0.2766 | 0.5174 | 0.3222 |

Mean NDCG@10 = 0.2972; sample std = 0.0004.

## Determinism

`reports/b4o_determinism_check.json` reran the selected h128 configuration with
the same seed and scored the first 1000 dev requests without reading qrels. The
request sequence matched and 42,968/42,968 score rows were exact
(`max_abs_score_diff = 0.0`).

## Comparisons

Best seed: `20260709_kuaisearch_b4o_sasrec_recbole_t03_h128_dev_s20260708`.

| Comparison | Delta NDCG@10 | 95% CI | Interpretation |
|---|---:|---|---|
| vs Random | +0.0165 | [0.0113, 0.0217] | significant sanity pass |
| vs B0b | -0.0163 | [-0.0201, -0.0125] | significantly below recent-behavior |
| vs B7-bge | -0.0329 | [-0.0382, -0.0276] | significantly below B7-best |

## Conclusion

B4o is now an official RecBole SASRec baseline, not the old placeholder adapter.
It is protocol-valid and significantly above Random, but it remains below B0b
and B7-best under this fixed-candidate KuaiSearch setting. The main caveat is
the high fixed-candidate cold-start rate for pure item-ID sequence models.
