# Supervised Motivation Diagnostics: Train-Only Calibration

Status: frozen before any supervised diagnostic dev evaluation on 2026-07-10.

## Scope and integrity

Calibration used only the final 9,694 retained train requests; the preceding
87,245 requests supplied optimization data. Dev records remained label-free,
no dev evaluator was called, and test data were not read. The materialized data
contain 96,939 train requests with at least one clicked candidate and preserve
all candidates and causal histories without truncation.

The following pre-training implementation failures were invalidated rather
than interpreted as experiments:

- The first materialization attempt found 66,778 exposure requests without a
  clicked candidate. They are ineligible for the frozen positive-listwise loss
  and were excluded from optimization arrays. This was recorded before any
  training or dev metric.
- Initial residual calibration attempts made no valid optimizer updates because
  FP16 normalization produced non-finite gradients. The implementation now uses
  an explicit `1e-6` normalization epsilon; a finite-gradient and parameter-
  update test was added. The affected outputs were overwritten and are not
  counted as calibration evidence or dev evaluations.

## Frozen references

On the same internal validation split, frozen non-personalized references were:

| Train-only reference | NDCG@10 |
|---|---:|
| BGE cosine | 0.294987 |
| Train-only popularity | 0.294145 |
| Best fixed cosine + popularity mix | 0.308705 |
| B0b history | 0.300671 |
| Best fixed cosine + B0b mix | 0.315090 |

## Calibration outcome

| Variant | Selected epoch | Internal NDCG@10 | Delta vs D1q |
|---|---:|---:|---:|
| D1q supervised query base | 1 | 0.307873 | 0.000000 |
| D1m mean-history residual | 3 | 0.308092 | +0.000219 |
| D1a query-attentive residual | 1 | 0.308275 | +0.000402 |

D1q exceeded each individual frozen query input, satisfying the frozen retry
rule, although it did not exceed their best fixed mixture. Both history
residuals made only small internal gains and remained below the fixed
query/history mixture. This is calibration evidence, not a dev conclusion.

D1a epoch 3 reached 0.308310, only 0.000035 above epoch 1. Because the locked
selection rule requires an improvement greater than `min_delta=0.0001`, epoch
1 remains selected. Final epochs are therefore D1q=1, D1m=3, and D1a=1 for all
three seeds, with no dev-selected retry.

## Frozen artifacts

- Base config SHA-256: `509fe62ad6b6fdc33a38a1e7857ccb98338a0829ded1fb42770489ee72da3f79`
- Data manifest SHA-256: `bb8e5984b8f41190193e9a84914aa369715bd3ab46261558836a6be11cb19d86`
- Final config: `configs/analysis/supervised_motivation_diagnostics_final.yaml`
- Final config is the sole authority for epoch counts and final seeds.
