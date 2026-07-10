# 20 - D2h Static History Waterline Protocol

Status: locked after D2 evaluation and before D2h calibration, scoring, or dev
evaluation on 2026-07-10.

## 1. Why this extension is required

D2t is a stronger supervised text scorer than zero-shot B2z, while D2p shows
that a train-only popularity prior substantially strengthens a
non-personalized ranker. The existing B7 waterline combines B2z with B0b. It
would be unfair to enter system design without also combining the stronger D2t
query score with the same causal B0b history score.

D2h is therefore a post-D2 baseline repair, not a proposed method and not a
new adaptive mechanism. Its only purpose is to establish the strongest static
waterline supported by the current evidence.

## 2. Frozen score

For each request and candidate:

`D2h = alpha * z(D2t) + (1 - alpha) * z(B0b)`.

The z-score is computed within request. D2t is the seed-matched final score from
doc 19; B0b is the frozen causal recent-behavior score. D2h has one global
alpha, no learned router, and no dev-fitted parameter.

## 3. Train-only alpha

Use the saved D2t epoch-3 calibration checkpoint on the final 10% retained
train segment and the already materialized causal B0b feature for those same
requests. Select alpha from `{0.0, 0.1, ..., 1.0}` by internal NDCG@10. Ties
select the largest alpha. Freeze the selected alpha for all three final seeds
before any D2h dev score is generated.

## 4. Fixed controls and evaluation

Generate three true-history D2h score files using D2t seeds
20260708/20260709/20260710. Also generate three matched wrong-history files by
replacing B0b with the already frozen C3-R wrong-history score for the same
seed. No model is retrained.

Six dev evaluations are authorized. Required comparisons use the shared paired
bootstrap:

- D2h versus B7, D2p, D2t, and B0b at seed 20260708;
- true D2h versus matched wrong D2h on all history-present requests and on the
  same-query donor subset, for all seeds;
- exact per-request metric equality between D2h and D2t when history is absent.

## 5. Locked interpretation

- If D2h exceeds B7, D2h becomes the baseline-to-beat. The motivation becomes
  more precise: stronger query learning raises the static waterline but does
  not itself solve identity-sensitive history use.
- If D2h does not exceed B7, retain B7 and report that query-tower fine-tuning
  did not improve the static history mixture.
- A true-over-wrong D2h effect supports identity specificity of the stronger
  static control; it does not establish learned event selection.
- D2h cannot support routing, query-attentive history, or proposed-system
  claims because alpha is global and B0b is fixed.

Training/scoring cannot read dev/test qrels. Test remains untouched.
