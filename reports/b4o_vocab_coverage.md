# B4o RecBole Vocab Coverage Check

Date: 2026-07-09

Status: reviewed; continue with the documented cold-start policy.

Scope: train-interaction vocabulary versus blind dev candidate item IDs. This
check reads `records_dev.jsonl` candidate IDs only. It does not read qrels or
labels and does not run the evaluator.

## Evidence

| Quantity | Value |
|---|---:|
| Train interaction vocab items | 188,300 |
| Dev requests | 12,229 |
| Dev candidate rows | 575,609 |
| Dev candidate rows in vocab | 127,810 |
| Row-level in-vocab rate | 22.2043% |
| Row-level cold candidate rate | 77.7957% |
| Dev unique candidate items | 396,822 |
| Dev unique candidate items in vocab | 47,080 |
| Unique-item in-vocab rate | 11.8643% |
| Requests with zero in-vocab candidates | 1,632 |
| Zero-in-vocab request rate | 13.3453% |

## Decision

The cold candidate rate is above the doc 14 review threshold of 30%.

No vocabulary expansion is applied before the first B4o run. Adding dev/test
candidate IDs with no train interactions would create item embeddings with no
training signal; their scores would be random model state rather than official
SASRec evidence. Using popularity or text fallback for cold items would also
change B4o from a query-blind sequence baseline into a hybrid adapter.

Therefore B4o keeps the frozen doc 14 cold-start policy:

- candidates in RecBole vocab receive the SASRec next-item score;
- candidates outside vocab receive the request's minimum in-vocab score minus a
  fixed margin;
- requests with no in-vocab history or no in-vocab candidates receive tied zero
  scores with diagnostics.

Interpretation caveat: B4o quality on KuaiSearch fixed candidates will partly
measure the interaction-vocab coverage limitation of pure sequence baselines in
this sparse fixed-candidate setting. The cold-start rates must be reported in
the B4o card and any Batch 2b summary.
