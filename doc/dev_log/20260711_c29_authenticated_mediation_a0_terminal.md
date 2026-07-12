# 2026-07-11 C29 causal-authentication gate

C29 introduced strict prequential user-memory authentication as an admission
mask on a full pretrained factual/null Transformer ranker.  Its label-free G0
strongly separated true from distinct-user wrong history, and all three fixed
GPU runs completed on the full frozen fit set.

The architecture was load-bearing before labels: its ensemble changed 38.33%
of A orders and 9.0% of top-10 sets, while wrong-history substitution changed
5.33%--12.83% of top-10 sets per seed.  Eighteen of nineteen A0 checks passed.
One seed exceeded the candidate-permutation fp32 tolerance by 3.71e-7
(1.3709e-6 observed versus 1e-6 allowed), so C29 closed without opening A
labels.  Delayed-B, escrow, dev, and test stayed closed.

This is not evidence of ranking utility.  It is stronger structural evidence
than C28 because correction directions correlate across seeds.  The next legal
step is a separately frozen, weights-preserving canonical-order continuation;
it may not retrain, change the threshold, inspect A labels before A0, or claim a
new architecture primitive.
