# D2 Fine-tuned Non-personalized Control: Train-only Calibration

Status: frozen before any D2 dev scoring or evaluation.

The public `BAAI/bge-small-zh-v1.5` initialization reproduces the official
frozen query artifact on 128 aligned requests (mean cosine 1.000000, minimum
0.9999999). Query tokens cover all 96,939 retained train requests and 12,229
label-free dev requests in exact packed-record order.

## D2t epoch selection

| Epoch | Internal NDCG@10 | Train loss |
|---:|---:|---:|
| 1 | 0.299673 | 2.678801 |
| 2 | 0.304323 | 2.619090 |
| 3 | 0.305032 | 2.593185 |
| 4 | 0.305085 | 2.570627 |

Epoch 4 is only +0.000053 above epoch 3, below the frozen
`min_delta=0.0001`; epoch 3 is selected for all final seeds.

## D2p alpha selection

The first alpha calculation incorrectly used full-train click counts, which
contain internal-validation clicks. Its implausible 0.6067 result was
invalidated before any D2 dev run and remains recorded in
`alpha_recalibration.json`. The corrected calculation uses only
`item_log_click_internal_train.npy`.

| Alpha on D2t | Internal NDCG@10 |
|---:|---:|
| 0.0 | 0.294145 |
| 0.3 | 0.308687 |
| 0.5 | 0.311585 |
| **0.6** | **0.313677** |
| 0.7 | 0.312349 |
| 1.0 | 0.305032 |

Final D2p alpha is 0.6. For final dev scoring, popularity is recomputed from
all train requests, which is the legal scope once dev is held out.

## Frozen artifacts

- Base config SHA-256: `89cd643133385d54a02454158b834d722488a2ba5b167d5172ead6e17bd03885`
- Token manifest SHA-256: `18e285a88388dabbad0c09273de086415250c0bcc6201f5cf34c35f45bf260f6`
- Calibration checkpoint SHA-256: `247f27e2cbeef90e3fa5a5f5c2b8488db26e1542aa1617cec19e4b8316373572`
- Final config: `configs/analysis/finetuned_nonpersonalized_control_final.yaml`

No D2 training, calibration, or scoring has read dev/test qrels. Test remains
untouched.
