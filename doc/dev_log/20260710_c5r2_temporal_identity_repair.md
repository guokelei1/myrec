# C5-R2 Temporal Identity Repair

Date: 2026-07-10

## Trigger

External audit found that the original wrong-history control changed user
identity and time support together: true dev history rolled forward, while all
wrong donors were frozen in train. The old numerical result is retained, but
its identity-specific interpretation is superseded.

## Ordering

Before any repaired outcome was read, the following were frozen:

- protocol: `doc/22_c5r_temporal_symmetric_identity_protocol.md`;
- config: `configs/analysis/c5r_temporal_symmetric_identity.yaml`;
- config SHA256:
  `029d765d080627d0bfd046b8dfb1f8cb22fb3c4054041e796a748ba7c077e675`;
- per-request freshness bound: factor four in `(age + 1)`;
- minimum balanced counts: 6,000 overall and 1,000 same-query;
- decision: all overall seed CIs positive, same-query mean positive, and at
  least two of three same-query seed CIs positive.

Label-free feasibility inspection was permitted before this lock. No repaired
per-request metric, qrels, or test artifact was read during feasibility work.

## Execution

```bash
python scripts/run_temporal_identity_control.py
python scripts/evaluate_scores.py --run-id 20260710_kuaisearch_c5r2_d2s_temporal_wrong_dev_s<seed> --split dev
python scripts/compare_runs.py ...
python scripts/finalize_temporal_identity_control.py
```

Materialization/scoring read no qrels. The shared evaluator added exactly three
dev entries. No model was trained and test was not read.

## Result

- 7,614 requests are freshness-balanced for all seeds; true-minus-wrong D2s is
  +0.037417 / +0.037890 / +0.036155 and every CI is positive.
- 1,063 requests are additionally same-query matched; mean delta is +0.009513.
- Same-query CIs are non-significant for seeds 20260708 and 20260709 and
  significant for 20260710: 1/3 versus the frozen 2/3 requirement.
- Different-user, strict-prior, freshness threshold, candidate coverage,
  same-query text equality, subset hashes, and no-history fallback all pass.

## Decision

The implementation defect is repaired, but the scientific gate **fails**.
D2s remains the numeric static waterline and aggregate correct-history value is
retained. Same-query identity specificity and formal Phase-5 system
authorization are not restored. No post-hoc threshold relaxation or alternative
donor policy will be used to convert this run into a pass.

Authoritative report:
`reports/pps_c5r2_temporal_symmetric_identity.{json,md}`.
