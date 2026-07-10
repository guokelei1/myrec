# C5-R2 Temporal-Symmetric Identity Repair

Status: **FAILED**.

The temporal-staleness implementation defect is repaired: both true and wrong histories now follow a strictly-prior prequential policy, and the extreme train-versus-dev age mismatch is bounded per request before comparison. Residual balance is reported rather than hidden. The scientific gate is adjudicated separately below.

## Frozen Counts

| Subset | Requests |
|---|---:|
| History present | 8,119 |
| Freshness-balanced, all seeds | 7,614 |
| Same-query + freshness-balanced, all seeds | 1,063 |
| History absent | 4,110 |

## Paired NDCG@10 Results

| Subset | Seed | Delta | 95% CI | Significant |
|---|---:|---:|---:|---|
| freshness-balanced | 20260708 | +0.037417 | [+0.032377, +0.042448] | yes |
| freshness-balanced | 20260709 | +0.037890 | [+0.032822, +0.042975] | yes |
| freshness-balanced | 20260710 | +0.036155 | [+0.031096, +0.041274] | yes |
| same-query + freshness-balanced | 20260708 | +0.009213 | [-0.000630, +0.019066] | no |
| same-query + freshness-balanced | 20260709 | +0.009062 | [-0.000806, +0.018852] | no |
| same-query + freshness-balanced | 20260710 | +0.010265 | [+0.000231, +0.020094] | yes |

## Freshness Balance

On the 7,614-request balanced subset, true-history median age is 367.5 and seed-20260708 donor median age is 734.5; the log-age SMD is 0.141. For the same-query subset the corresponding SMD is 0.256. The per-request factor-four bound removes the orders-of-magnitude mismatch in the original control, but does not imply perfect balance.

## Decision

Same-query mean delta: **+0.009513**; significant seeds: **1/3**.

C5-R2 fails because the frozen same-query significance requirement is not met. Aggregate correct-history value remains positive, but identity specificity is not sufficiently established for formal system authorization.

The original D2s performance waterline remains valid. What is not restored is the stronger statement that same-query identity specificity has already cleared its cheap falsifier. No model was trained and test was not read.

## Integrity

Assignment/no-history audit: **passed**. All candidate and qrels hashes are recorded in the evaluator artifacts.
