# 17 - Introduction and Motivation Repair Protocol

Status: locked before evaluating the controls below on 2026-07-10.

This protocol repairs the motivation chain after the Random-channel audit showed
that M3/M4 measure same-label argmax noise. It does not reinterpret the failed
oracle. It defines a narrower claim that can be tested without model training.

The executable configuration is
`configs/analysis/c3_history_identity_controls.yaml`. Its SHA256 is recorded in
every generated control run and in the final report.

## 1. Revised Claim

The repaired motivation asks two questions:

1. Do query evidence and history evidence provide complementary aggregate
   ranking signal inside the fixed candidate pool?
2. Is the history contribution specific to the correct user, rather than a
   generic category or popularity prior?

If both answers are positive, the paper may motivate interaction-aware use of
query, candidate, and history evidence. It may not claim that dynamic channel
routing or per-request oracle headroom has been established.

## 2. Frozen Inputs

- Query channel: B2z `20260708_kuaisearch_b2z_bge_small_zh_dev`.
- History channel: B0b `20260708_kuaisearch_b0b_recent_behavior_dev`.
- Static mixture: B7-bge `20260708_kuaisearch_b7_bge_dev_a02`, alpha 0.2.
- Seeds: 20260708, 20260709, and 20260710.
- Candidate manifest, evaluator, metric, tie-break, and qrels remain unchanged.
- No model is trained and no hyperparameter is selected.

The existing B7 comparisons provide the aggregate-complementarity test. This
protocol adds an identity control only.

## 3. Wrong-History Construction

For every dev request with non-empty history, replace its history with a
history from a different user in the earlier train split. Empty-history targets
remain empty. The target query and candidate list never change.

Queries are normalized with Unicode NFKC, lowercasing, and removal of all
whitespace. Donors are selected without qrels in this fixed priority order:

1. same normalized query and same history-length bin;
2. same normalized query;
3. same majority candidate top-level category and history-length bin;
4. same majority candidate top-level category;
5. same history-length bin;
6. any eligible train donor.

The bins are `[1], [2], [3-4], [5-8], [9-16], [17-32], [33-50]`. A bounded,
deterministic hash reservoir retains at most 32 donors per key. The seed only
chooses among eligible retained donors. Every selected donor must have a
different `user_id`, a non-empty history, and a request timestamp earlier than
the target. Every donor event timestamp must also precede the target request.

This matching makes the control conservative: same-query matches hold the
current intent fixed, while category/length matching prevents an unrelated
history length or product domain from becoming the main perturbation.

## 4. Evaluation

For each seed, generate and evaluate:

- B0b with matched wrong-user history;
- B7-bge with the same frozen query scores, alpha, and wrong-history scores.

All six runs use the shared evaluator and are appended to the dev-evaluation
log as claim controls, not tuning evaluations. Paired bootstrap comparisons use
the shared comparison implementation with 10,000 samples and seed 20260708.

Report three disjointly interpretable views:

- all history-present dev requests;
- the subset whose wrong history is matched on normalized query for all seeds;
- history-absent requests, on which B7 must be rank-equivalent to B2z.

## 5. Locked Decision Rule

The repaired motivation passes only if all conditions hold:

1. existing B7-vs-B0b and B7-vs-B2z paired CIs have lower bounds above zero;
2. on history-present requests, true B7 beats every wrong-history B7 seed with
   paired CI lower bound above zero;
3. the same-query subset contains at least 1,000 requests, its mean delta is
   positive, and at least two of three seed-level CIs have lower bounds above
   zero;
4. on history-absent requests, B7 and B2z have identical per-request rankings
   as witnessed by identical per-request metrics.

B0b comparisons are supporting diagnostics and do not independently determine
the gate.

If the gate passes, the permitted design transition is **query-conditioned
history interaction**, not oracle routing. If it fails, the paper contracts to
a benchmark and baseline analysis; no personalization mechanism is inferred.

## 6. Paper Boundary

Even after a pass, the evidence is limited to the filtered, click-observed
KuaiSearch dev population. The control establishes predictive identity
specificity under a matched permutation, not a randomized causal effect in a
deployed system. M3/M4 remain failed diagnostics and stay visible in the
repository.
