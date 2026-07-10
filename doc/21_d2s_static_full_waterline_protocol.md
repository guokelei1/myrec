# 21 - D2s Static Full-Evidence Waterline Protocol

Status: locked after the repository fairness audit and before D2s calibration,
score generation, or dev evaluation on 2026-07-10.

## 1. Why this repair is required

D2p establishes that the frozen non-personalized combination of fine-tuned
text and train-only popularity is stronger than D2t alone. D2h subsequently
combines D2t with causal B0b history, but omits the already validated popularity
term. Calling D2h the strongest static waterline would therefore understate the
baseline available to the proposed system.

This omission was discovered after D2h dev results during the final repository
audit. D2s is an explicit post-result fairness repair. It introduces no learned
adaptive mechanism and does not retrain any model.

## 2. Frozen score

For every request and candidate:

`D2s = beta * z(D2p) + (1 - beta) * z(B0b)`.

`D2p` remains exactly the frozen score from doc 19:

`D2p = 0.6 * z(D2t) + 0.4 * z(train-only popularity)`.

All z-scores are within request. B0b is the same strictly prior correct-user
history score used by D2h. D2s has one global beta and cannot condition its
weight on the request, user, query, candidates, or history.

## 3. Train-only beta

Use the saved D2t epoch-3 calibration checkpoint on the final 10% retained
train segment. Reconstruct D2p with the already frozen alpha 0.6 and popularity
counts from only the first 90% internal-train segment. Combine it with the
causal B0b feature for the same requests.

Select beta from `{0.0, 0.1, ..., 1.0}` by internal NDCG@10. Ties select the
largest beta, preferring the stronger non-personalized control. Freeze beta for
all final seeds before generating any D2s dev score.

## 4. Fixed controls and evaluation

Generate three true-history D2s score files from the frozen seed-matched D2p
files and three matched wrong-history files using the frozen C3-R donor scores.
No model is retrained.

Six dev evaluations are authorized. Use the shared evaluator and frozen
candidate manifest. Required comparisons are:

- D2s versus D2h, D2p, and B0b at preselected seed 20260708;
- true D2s versus matched wrong D2s on all history-present requests and on the
  same-query donor subset for all three seeds;
- exact per-request NDCG@10/MRR/Recall@10 equality between D2s and D2p when
  history is absent.

## 5. Locked interpretation

- If D2s significantly exceeds D2h, D2s replaces D2h as the binding static
  baseline and all paper/design thresholds must be reissued.
- If D2s does not significantly exceed D2h, D2h remains the binding baseline,
  while D2s is retained as the complete static-feature control.
- A true-over-wrong effect supports identity specificity of the full static
  control; it does not establish adaptive event selection.
- No-history requests must reduce to the seed-matched D2p ranking because B0b
  standardizes to zero.

Training and scoring cannot read dev/test qrels. Test remains untouched.
