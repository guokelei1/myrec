# C58 outcome — structural signal present, numerical A0 terminal

C58 stopped before every fit label.  Its fixed semantic candidate-budget
operator was deterministic, finite, parameter-free, and exactly preserved the
registered no-history base and repeat item-only anchors.  It was also strongly
load-bearing on the 1,200 untouched holdout requests:

- primary versus base: 1,193 complete-order and 857 Top-10 changes;
- primary versus wrong history: 1,187 complete-order and 863 Top-10 changes;
- primary versus history-axis: 577 complete-order and 99 Top-10 changes.

The sole failed contract was the frozen candidate-permutation tolerance.
Changing candidate storage order produced maximum score differences of
`3.0756e-5`, `2.3842e-6`, `4.7684e-7`, and `0` across the four shard audits;
the first two exceed `2e-6`.  The source is order-dependent floating-point
reduction in candidate softmax/request z-scoring, not a semantic or label
outcome.  Consequently C58 is terminal and its utility is unknown; changing
its tolerance or implementation in place is forbidden.

One separately frozen numeric-equivalence successor is authorized: retain the
identical formula, data, controls, score coefficient, and thresholds, but use
sorted float64 symmetric candidate-set reductions.  It must rerun all of A0
under a new lock.  This is an implementation falsifier, not a new mechanism or
a favorable interpretation of C58.
