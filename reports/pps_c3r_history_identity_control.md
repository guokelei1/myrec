# C3-R History Identity Control

Status: **historical pass; identity interpretation superseded by failed C5-R2**

This is the locked replacement for the invalid M3/M4 positive claim. It
supports the historical aggregate comparison only. Its train-frozen donors do
not isolate identity from freshness; see
`reports/pps_c5r2_temporal_symmetric_identity.md`.

## Aggregate Complementarity

| Comparison | Delta NDCG@10 | 95% CI |
|---|---:|---:|
| B7 vs B0b | +0.0166 | [+0.0121, +0.0211] |
| B7 vs B2z | +0.0249 | [+0.0216, +0.0282] |

## Matched Wrong-User Control

| Seed | Wrong B7 full NDCG@10 | True-minus-wrong, history present | 95% CI | True-minus-wrong, same query | 95% CI |
|---:|---:|---:|---:|---:|---:|
| 20260708 | 0.3022 | +0.0427 | [+0.0375, +0.0478] | +0.0309 | [+0.0227, +0.0394] |
| 20260709 | 0.3017 | +0.0434 | [+0.0382, +0.0485] | +0.0335 | [+0.0256, +0.0418] |
| 20260710 | 0.3018 | +0.0432 | [+0.0380, +0.0484] | +0.0319 | [+0.0237, +0.0401] |

## Evidence Structure

- History is absent for 4,110/12,229 requests (33.6%).
- Among history-present requests, median history length is 6.
- Deepest-category history/candidate Jaccard is zero for 38.4% and has median 0.111.
- Same-query wrong-history subset: 2,709 requests.
- B7 and B2z are identical on all history-absent requests: True.

## Decision

Correct-user history is identity-specific and query/history evidence is complementary in aggregate. This supports interaction-aware fusion, not per-request oracle routing.

The permitted architecture hypothesis is a query-anchored personalized
residual with exact masking for absent history. Whether it can identify
irrelevant events remains a system-level falsification, not a result of
this control. The failed M3/M4 oracle evidence is not restored.
