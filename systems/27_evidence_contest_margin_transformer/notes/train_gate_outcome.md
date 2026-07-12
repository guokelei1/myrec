# C27 train-gate outcome

Status: **terminal failure at label-free A0 (17/18 checks passed)**.

The locked run completed three seeds, four 139,648-trainable-parameter modes,
and two epochs per mode in 611.33 seconds.  Training, gradients, matched
initialization/capacity, determinism, permutation equivariance, exact neutral
base order, pair antisymmetry/complementarity, query/no-history D2p, and repeat
item-only contracts all passed.

The evidence contest materially fixed C26's rank-inertness: it changed
253/600 complete orders (42.17%) and 33/600 top-10 sets (5.5%) relative to
D2p.  Wrong histories changed every numerical correction and changed
74/86/133 complete orders (12.33%/14.33%/22.17%) across seeds, all above the
5% requirement.

The sole failure was the preregistered wrong-history top-10 contract.  It
required at least 0.5% (3/600) in every seed; observed counts were 2/600,
1/600, and 7/600.  The first two seeds therefore failed even though the third
passed.  The threshold cannot be relaxed and the third seed cannot be selected
post hoc.  Internal-A, delayed-B, escrow, dev, and test labels remain unopened,
so C27 has no ranking-utility verdict.

C27 closes uniform all-pair soft-Borda aggregation for this evidence contest,
not antisymmetric pairwise evidence as a family.  The failure pattern supports
one new hypothesis: uniform averaging over many far-margin candidate pairs
dilutes history at decision boundaries.  A successor may test a generic
base-margin-local contest graph on a previously untouched label-free role.  It
must keep C27 uniform aggregation as a matched control and may not target rank
10, tune a temperature on this A0 role, or merely increase `pair_delta_max`.

Raw report SHA-256:
`51b67b8245bac6018e2c36760ea77d8163d60c81cabf641d9ebefc7d0dd51492`.
