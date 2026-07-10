# D2s Complete Static Waterline Summary

> Identity-control interpretation superseded by C5-R2. Numeric-waterline status
> is subsequently superseded by C5-R3 item-only (mean 0.3453755); D2s remains a
> valid bundled-history reference, but is no longer the strongest static
> control. See `reports/pps_c5r3_candidate_history_alignment.md`.

Status: complete historical bundled-history reference; item-only is now binding.

| Control | Mean NDCG@10 | Sample SD |
|---|---:|---:|
| D2s D2p + true history | 0.3416 | 0.0004 |
| D2s D2p + matched wrong history | 0.3181 | 0.0005 |

At seed 20260708, D2s exceeds D2h by +0.0064 (95% CI [+0.0037, +0.0090]).

True-minus-wrong history remains significant for every seed on both the history-present and same-query subsets. On all 4,110 no-history requests, D2s and seed-matched D2p have identical NDCG@10, MRR, and Recall@10.

D2s is a post-result fairness repair discovered after D2h: D2h omitted the popularity term already validated in D2p. Beta was selected on train only before D2s dev scoring; no model was retrained and test remains untouched.
