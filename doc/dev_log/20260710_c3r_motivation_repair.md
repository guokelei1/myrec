# 2026-07-10 - C3-R Motivation Repair

## Decision

Do not repair the failed M3/M4 oracle by reinterpretation. Replace its positive
claim with a narrower construct: aggregate query/history complementarity plus
identity-specific value of the correct user's history.

## Ordering

1. `doc/17` and the executable YAML were finalized locally at 14:25:45 +08:00.
2. Wrong-history scores were generated without qrels from earlier train donors.
3. The first of six fixed dev control evaluations ran at 14:28:29 +08:00.
4. Shared paired comparisons were run only after all six metrics existed.

The YAML `locked_at` midnight value is a date marker; exact local provenance is
recorded in `reports/pps_c3r_protocol_lock_manifest.json`.

## Result

- true B7 minus wrong-history B7 on 8,119 history-present requests: mean
  +0.0431 over three seeds; all paired CI lower bounds > 0;
- same-query donor subset: 2,709 requests, mean +0.0321; all CI lower bounds > 0;
- B7 and B2z identical on all 4,110 no-history requests;
- all candidate, score, config, evaluator, and dev-log reconciliation checks
  passed.

C3-R and C5-R pass. The authorized insight is **query-anchored personalized
residual**. M3/M4, Consensus Law, and oracle routing remain failed/retired and
are not restored by this decision.

The matched-history observation, construction, and decision rule were locked
before evaluation. The residual name is the post-gate architecture mapping of
that evidence boundary, not a pre-registered model-success claim.
