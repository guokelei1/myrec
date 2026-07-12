# C31 train-gate outcome

Status: closed at A1 on 2026-07-11.

G0 passed all five causal-authentication checks on the untouched 600-request
internal-A cohort.  True authentication was nonempty on 58.0% of requests,
true authenticity averaged 0.41205, true-minus-wrong had a 95% interval of
[0.38107, 0.44433], and wrong authenticity was exactly zero.

Three new GPU seeds trained for 183 steps on the unchanged C29 fit cohort.  A0
passed all 19 checks: the ensemble changed 45.83% of complete orders and 18.0%
of top-10 sets; wrong history changed 46.0%--46.17% of orders and
16.83%--18.83% of top-10 sets; repeat, no-history, no-auth, query-mask,
determinism, and candidate permutation contracts all passed exactly.

A1 produced the first consistent positive formal estimate in the architecture
series but failed the frozen stability gate.  Ensemble NDCG@10 improved by
0.0017455 over D2p and every seed was positive (+0.0025424, +0.0012590,
+0.0014351).  The 95% interval [-0.0010277, 0.0044739] crossed zero, one of
three fixed folds was negative (-0.0013161), and true-over-wrong also crossed
zero.  C31 therefore closed; delayed-B, escrow, dev, and test stayed unopened.

The authoritative tracked report is `reports/pps_c31_train_gate.json`, SHA-256
`66a42bb71abfc0e7b8ee608d8e6c6116311e803ef37ca626783968e7bd9446fc`.

A fixed post-terminal geometric diagnostic found that recomputing attention in
the adapted space and removing the profile component parallel to the query
raised the already-open A point estimate to +0.002506, with all three seeds
positive but a still-zero-crossing interval.  A diagnostic implementation first
varied the `compare` seed by variant, unintentionally changing hash-fold
membership; after correction to the formal fixed fold seed, one fold remained
negative (-0.001357).  This is formulation evidence only and is not a C31 rescue.
