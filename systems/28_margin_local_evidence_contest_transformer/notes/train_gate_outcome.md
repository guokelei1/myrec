# C28 train-gate outcome

Status: **terminal failure at internal-A utility gate A1**.

The locked run trained three seeds, five equal-parameter/equal-compute modes,
and two epochs per mode in 781.68 seconds.  The label-free A0 passed all 18
checks.  The primary changed 420/600 complete orders (70.0%) and 75/600 top-10
sets (12.5%) relative to D2p.  Wrong histories changed 185/218/131 complete
orders and 8/8/4 top-10 sets across seeds, exceeding every frozen threshold.
Antisymmetry, probability complementarity, neutral base identity,
determinism, candidate permutation, and all fallback contracts also passed.

Passing A0 authorized opening only the 600 internal-A labels.  A1 then failed
all 11 utility checks.  Seed-averaged NDCG@10 was 0.583317 for the primary and
0.583369 for D2p, a difference of -0.000052 with 95% CI
[-0.001993, 0.001834].  The primary also failed to beat every matched control.
Wrong history was slightly better than clean history (true-minus-wrong
-0.000121, CI crossing zero), and clicked-minus-unclicked correction was
-0.001509 (CI crossing zero).  Delayed-B, C28 escrow, dev, and test remain
unopened.

C28 therefore validates a narrower architectural fact but not a useful
ranker: fixed continuous margin-local competition solves the prior
rank-activity/corruption-sensitivity failure, while a freely learned odd
comparator does not give stable ranking-aligned evidence direction.  A
post-terminal sign audit further found near-zero correction correlations
between seeds (0.011, -0.005, 0.012), with median within-request correction
ranges 0.0638, 0.4496, and 0.000343.  Exact sign reversal helped seeds 1 and 3
but badly hurt seed 2, so the failure is not a globally inverted head; it is
seed-dependent direction/scale non-identifiability.

C28 must not be rescued by choosing seed 2, flipping a sign, tuning the kernel
scale, shrinking corrections, or reopening delayed-B.  A successor must use a
new outcome-isolated role and remove the free comparator gauge through a new
Transformer-internal evidence law, while retaining C28 locality and its exact
uniform/local controls where relevant.

Raw report SHA-256:
`d0f854f5ab6200f54e00a89012bb66e738b11d2e0cb226db91ff4f99f721d9e2`.
