# 22 - C5-R Temporal-Symmetric Identity Repair Protocol

Status: locked before outcome evaluation on 2026-07-10 20:51 +08:00.

This protocol repairs the temporal-staleness confound found in the original
matched wrong-user control. It does not reinterpret or delete the original
result. The original control compared rolling true history with train-frozen
wrong history and therefore changed user identity, freshness, and catalog
period together. The replacement below changes user identity while bounding
history freshness under the same rolling, strictly-prior information policy.

The executable configuration is
`configs/analysis/c5r_temporal_symmetric_identity.yaml`. Its SHA256 is recorded
in every generated run and in the final repair report.

## 1. Claim and Scope

The repaired question is:

> Among dev requests for which another user's strictly-prior history can be
> matched on freshness and observable context, does the correct user's rolling
> history provide higher ranking value?

A pass supports predictive identity specificity on the matched KuaiSearch dev
population. It is not a randomized causal claim and does not establish dynamic
routing or query attention.

## 2. Symmetric Information Policy

Both sides use histories available before the target request:

- true history is the frozen standardized target-user history, whose events all
  satisfy `event.ts < target.ts`;
- wrong history is a snapshot attached to a different user's train or
  earlier-dev request, with `donor_request.ts < target.ts` and every donor event
  strictly before that donor request;
- dev donor snapshots are inserted only after all targets at the same timestamp
  have been scored, so same-time outcomes cannot enter the control;
- current-request qrels, future dev behavior, test records, and test qrels are
  never read by materialization or scoring.

This is an explicit prequential/as-of amendment to the broad train-statistics
sentence in `doc/13`: strictly prior behavior already present in the unified
record may be used as request-time history evidence. Train-fitted parameters
and aggregate statistics remain train-only. The same policy must be frozen
before any final test evaluation.

## 3. Matching and Freshness Balance

History age is defined against the target request:

```text
age(history, target) = target.ts - latest(history.event.ts)
```

Donors must have a different `user_id`. The fixed context priority remains:

1. same normalized query and history-length bin;
2. same normalized query;
3. same majority candidate top-level category and history-length bin;
4. same majority candidate top-level category;
5. same history-length bin;
6. global.

At each tier, a donor is freshness-balanced when

```text
abs(log2((donor_age + 1) / (true_age + 1))) <= 2
```

so the two ages differ by at most a factor of four after the `+1` guard. The
first context tier containing a balanced donor is used. The seed selects among
at most the eight best balanced candidates; if none exists at any tier, the
closest fallback is retained for coverage diagnostics but excluded from the
balanced gate subsets. Pools retain the 64 most recent eligible snapshots per
key. These constants were fixed after label-free feasibility inspection and
before reading any repaired-control metric.

## 4. Frozen Scoring and Evaluation

- True score: existing seed-matched D2s rolling true-history runs.
- Wrong score: the same frozen D2p run and `beta=0.3`, replacing only B0b with
  the temporally matched wrong history.
- Seeds: 20260708, 20260709, 20260710.
- Metric: NDCG@10 using the shared evaluator and fixed candidate manifest.
- Comparisons: paired bootstrap, 10,000 samples, seed 20260708.
- No model training, calibration, or hyperparameter selection is performed.

The following views are reported:

- all original history-present requests, descriptive only;
- requests freshness-balanced for all three assignment seeds;
- the subset additionally matched on normalized query for all three seeds;
- history-absent requests, which remain empty and rank-equivalent to D2p.

## 5. Locked Decision Rule

C5-R2 passes only if all of the following hold:

1. at least 6,000 requests are freshness-balanced for all three seeds;
2. at least 1,000 requests are both same-query and freshness-balanced for all
   three seeds;
3. true D2s beats wrong D2s on the freshness-balanced subset with CI lower
   bound above zero for every seed;
4. the same-query/freshness-balanced mean delta is positive and at least two of
   three seed-level CI lower bounds are above zero;
5. every assignment is different-user and strictly prior, no balanced
   assignment exceeds the frozen age-gap threshold, candidate coverage is
   exact, and history-absent requests remain empty.

If the gate passes, C5-R may be restored with the explicitly bounded claim
"predictive identity specificity under a freshness-matched prequential
control." If it fails, identity specificity is not established and formal
personalized-system development remains blocked. The old train-frozen control
stays visible as a confounded historical result in either case.

