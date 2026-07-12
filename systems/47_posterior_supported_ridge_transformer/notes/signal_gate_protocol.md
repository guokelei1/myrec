# C47 staged signal-gate protocol (prelock draft)

No outcome role is selected and no C47 label is open. Exact hashes, counts,
seeds, thresholds, and GPU assignments become immutable in a later execution
lock before any C47 feature or score is materialized.

## D0 — operator gate

All must pass on CPU:

- finite forward/backward and nonzero query/history/candidate gradients;
- exact no-history zero correction;
- candidate and history permutation equivariance;
- support in `[0,1]` within tolerance;
- posterior correction magnitude never exceeds plain-ridge magnitude;
- orthogonal candidate support/correction exactly zero;
- duplicate aligned evidence monotonically raises support;
- fixed hand-computed solve matches NumPy/explicit inverse coordinates.

## G0 — fresh two-domain lock

- KuaiSearch A is hash-selected as 600 requests from the union of label-closed
  C34-A/C36-A after excluding every known label-opened target and every incident
  index. Prior label-free features/scores may exist, but no label or utility
  outcome exists and selection may not inspect those scores;
- the 2,370 prelock label-exposed indices registered in
  `reports/pps_c47_prelock_label_scope_incident.json` are fit-only and forbidden
  from every C47 outcome role;
- Amazon-C4 A must come from the still-unmaterialized C39 reserve or another
  mechanically proven untouched train role;
- fit roles may reuse labels already lawfully opened by C38/C46, but A labels
  remain closed through all training and A0 scoring;
- true/wrong donors are distinct-user and matched on frozen history-length
  bins; candidate hashes are frozen;
- dev/test records, labels, and qrels remain closed.

Exact frozen counts are 6,000 fit / 600 A / 395 reserve for KuaiSearch and
6,000 fit / 300 A / 99 reserve for Amazon-C4. Kuai fit is a hash subset of the
already-open C34 fit; Amazon fit exactly reuses C38 fit. No fit result selects
an A request.

## S0 — fresh fixed-operator signal gate

After proposal and selection locks, encode only the two A roles without labels.
Score `posterior_supported`, `plain_ridge`, fixed-temperature softmax history
attention, query base, and matched wrong history. A labels may open only after
candidate hashes, deterministic rescore, candidate/history permutation,
finite-state, no-history-zero, and support-bound checks all pass.

On **each** domain all conditions are binding:

- posterior-supported minus query base `>=0.002`, CI lower bound `>0`, and all
  three hash folds positive;
- posterior-supported minus plain ridge and fixed softmax each `>=0.0005`, CI
  lower bound `>0`, and all folds positive;
- true minus wrong `>=0.002`, CI lower bound `>0`, and all folds positive;
- clicked correction direction and clicked true-minus-wrong CI lower bounds
  are positive.

Any failure closes C47 before trainable implementation. Passing S0 authorizes
only the A0/A1 trained gate below; it is not a proposed-system result.

## A0 — load-bearing trainable architecture (only after S0)

Before A labels open, the trained posterior-supported mode and every matched
control must be finite, deterministic, permutation-equivariant, updated, and
parameter/compute audited. On both domains the primary must change at least 5%
of complete orders and 1% of top-10 sets versus plain ridge; replacing
`rho_c` by one in the same checkpoint must change at least 5% of orders and
0.5% of top-10 sets. True/wrong histories must change at least 5% of orders.
No-history returns the registered base exactly; Kuai repeat returns registered
item-only exactly.

## A1 — fresh utility

All conditions are conjunctive on both domains:

- primary minus registered base `>=0.002`, paired CI lower bound `>0`, every
  seed and hash fold positive;
- primary minus plain ridge, softmax attention, and free scalar gate
  `>=0.0005`, paired CI lower bound `>0`, every seed positive;
- true minus wrong history CI lower bound `>0` and every fold positive;
- clicked correction direction and clicked true-minus-wrong CI lower bounds
  `>0`;
- no post-outcome lambda, width, support exponent, scale, epoch, subset, or
  dataset-specific adjustment.

Failure in either domain closes C47. Passing only authorizes a separately
frozen full implementation; it does not authorize dev/test.
