# C29 train-gate outcome

Status: terminal at label-free A0; internal-A labels, delayed-B, escrow, dev,
and test remain closed.

G0 passed all five preregistered authentication checks.  On the untouched 600
request A role, true history authentication was 0.39294 with 95% CI
[0.36202, 0.42400], 55.67% of requests had at least one authenticated event,
and wrong-history authentication was exactly zero.

Three fixed GPU seeds each processed all 455,144 fit candidates for 9,483
steps.  The seed-averaged model changed 230/600 complete orders (38.33%) and
54/600 top-10 sets (9.0%).  Wrong history changed 32.83%--43.33% of complete
orders and 5.33%--12.83% of top-10 sets.  Deterministic rescore, no-auth,
query-absent, no-history, repeat fallback, initialization, gradient, candidate
hash, and isolation checks all passed.

The sole failed check was candidate-permutation numerical tolerance.  Maximum
absolute score differences were 9.83e-7, 1.37e-6, and 8.34e-7 against a frozen
1e-6 threshold.  C29 therefore failed 1/19 A0 checks and did not open A labels.
The failure does not authorize relaxing the threshold or editing C29.

Post-terminal label-free score diagnostics found flat cross-seed correction
correlations 0.595/0.614/0.732.  Across 334 active requests, pairwise per-request
correlation medians were 0.538/0.591/0.736 and positive on 87.4%--94.9% of
eligible requests.  This is materially more stable than C28's near-zero
correlations and justifies a separately locked mechanical continuation that
canonicalizes candidate computation order without retraining or label access.

Authoritative raw report SHA-256:
`9204b1aba37111c6ba763812ca48b5605042710c6a05b4f31c6d23d6e7f9eb60`.
