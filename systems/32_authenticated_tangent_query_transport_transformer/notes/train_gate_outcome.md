# C32 train-gate outcome

Status: closed at A1 on 2026-07-11, with the strongest positive result so far.

G0 passed 5/5 on untouched internal-A: true authentication coverage was 59.67%,
true authenticity averaged 0.42365 with true-minus-wrong 95% CI
[0.39188, 0.45533], and wrong authenticity was zero.  Three new seeds trained
183 full-request steps each.  A0 passed all 19 checks; the ensemble changed
48.17% of orders and 15.17% of top-10 sets, and every fallback, determinism, and
candidate-permutation check was exact.

A1 passed four of five conditions.  Primary-minus-D2p was +0.0042684 with a
strictly positive 95% CI [+0.0000707, +0.0085493].  All seeds were positive:
+0.0043247 / +0.0040383 / +0.0044421.  True-over-wrong also had a positive CI.
The only failure was every-fold positivity: fixed folds were +0.0049125,
-0.0002877, and +0.0081912.  C32 therefore remains terminal; controls and
delayed-B were not authorized, and escrow/dev/test stayed closed.

The authoritative tracked report is `reports/pps_c32_train_gate.json`, SHA-256
`f525d4173a230a5130af79da430042463b8652d69f1abf55b0e6b6772acd153e`.

Two audits are important.  First, recomputing D2p with a different item batch
layout changed 49/10,000 complete fit orders and three top-10 sets at <=2.85e-5
numeric scale, but changed fit NDCG on zero requests; execution was kept frozen.
Second, the first post-terminal variant script varied the comparison seed and
therefore changed fold membership.  With the formal fixed fold seed, adapted
attention tangent scored +0.004054 but crossed zero and retained the same
-0.0002877 fold.  No tested simple attention/magnitude reduction repairs the
remaining instability.  The formal C32 numbers were never affected by this
diagnostic bug.
