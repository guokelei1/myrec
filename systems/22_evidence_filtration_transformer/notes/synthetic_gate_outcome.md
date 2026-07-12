# C22 synthetic gate outcome

Status: **failed; terminal stop before repository data**.

The single hash-locked run completed three seeds × four parameter-identical
modes on physical A40 GPU 2 in 327.46 seconds.  All 10,800 optimizer steps were
finite.  The filtration Jacobian contract held exactly in every seed:
anchor-from-recurrence, anchor-from-transfer and recurrence-from-transfer were
all zero, while recurrence-to-transfer norms were `2.29`, `1.49` and `3.49`.
Dense controls had nonzero forbidden Jacobians.  Thus the intended architecture
was implemented and load-bearing.

## Result

| seed | filtration repeat | filtration supported | dense supported | parallel supported | final-projection supported |
|---:|---:|---:|---:|---:|---:|
| 20260730 | 1.0000 | 0.9570 | 0.9729 | 0.9809 | 0.9625 |
| 20260731 | 1.0000 | 0.9638 | 0.9860 | 0.9663 | 0.9565 |
| 20260732 | 1.0000 | 0.9552 | 0.9454 | 0.8973 | 0.9324 |

Filtration passed all absolute utility and evidence contracts: no-history and
repeat accuracy were 1.0, supported accuracy exceeded 0.95 in every seed,
supported margin gain was `1.75`--`1.82`, corruptions retained at most 1.1% of
clean gain, identity removal reduced repeat margin by `5.54`--`5.59`, and
no-history/query-mask were bitwise base.  Candidate permutation, deterministic
rescore, matched initialization/parameters and nontrivial order changes passed.

It nevertheless failed the only test that could justify the extra architecture:
control advantage.  Repeat accuracy was exactly tied at 1.0 for all modes.
Filtration-minus-dense supported accuracy was `-0.0159`, `-0.0222`, `+0.0098`;
versus parallel it was `-0.0239`, `-0.0025`, `+0.0579`; versus the strong final
projection it was `-0.0056`, `+0.0074`, `+0.0228`.  No comparison met the frozen
`+0.03` per-stratum and `+0.05` worst-stratum margins across seeds.

## Decision

C22 is closed.  The causal filtration is mathematically real and safe, but
ordinary dense/parallel Transformers and a late recurrence floor already solve
the same task as well or better.  This confirms the nearest-neighbour warning:
block-triangular evidence ordering does not pay empirical rent beyond simpler
known structures.  Do not change block widths, control thresholds, generator,
steps or normalization and rerun.  No real train, dev, test, standardized
record, qrel or evaluator was accessed.

Raw report SHA-256:
`63cf89f84d3ee1c636b1175af5c369dcffad92ab3cb5ab3d21ff1ac4227391fe`.
Proposal-lock SHA-256:
`c4dd6ca1fecd92c6cb22d2d474f49cc49d1aea4acba9c29cd8da9a5e7822473c`.
