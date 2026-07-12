# Post-terminal fit-only readout audit

This descriptive audit was run only after C26 had already failed A0.  It used
the first 1,000 requests in the already-open fit role and the three frozen
primary checkpoints.  Internal-A, delayed-B, escrow, dev, and test remained
closed.  No C26 threshold, model, or decision was changed.

| Seed | median candidate correction range | clicked-minus-unclicked mean (95% bootstrap CI) | fit order/top-10 changes vs D2p | fit wrong-history order/top-10 changes |
|---|---:|---:|---:|---:|
| 20260755 | `5.32e-6` | `+1.90e-7` [`+0.93e-7`, `+2.88e-7`] | 1 / 0 of 1,000 | 1 / 0 |
| 20260756 | `2.19e-6` | `+0.79e-7` [`-1.06e-7`, `+2.74e-7`] | 0 / 0 | 3 / 0 |
| 20260757 | `8.36e-4` | `-1.38e-5` [`-3.63e-5`, `+0.72e-5`] | 105 / 2 | 107 / 3 |

The scalar head did not exhibit a stable “right direction, merely too small”
failure.  Two seeds collapsed the candidate-relative scale; the seed with a
larger scale did not have positive clicked direction.  This supports changing
the readout constraint itself: a successor should make candidate-relative
antisymmetry structural and aggregate pairwise margins, rather than tune the
size of another independent candidate scalar.
