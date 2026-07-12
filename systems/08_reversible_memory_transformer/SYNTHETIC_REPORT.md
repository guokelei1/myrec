# C08 structural synthetic report

Date: 2026-07-11. Device binding: `CUDA_VISIBLE_DEVICES=""`. Data access: none.
Label/qrels access: none. This report contains structural CPU evidence only.

Command:

```bash
CUDA_VISIBLE_DEVICES="" pytest -q test_reversible_memory.py --tb=short
```

Result: **8 passed in 2.49 seconds**.

Covered contracts:

- reversible write/read information flow;
- exact write undo and unit Jacobian determinant;
- aligned versus disjoint evidence support;
- constructive endpoint-state non-reduction witness;
- identical parameterization of loop and ordinary controls;
- exact empty-history/query-only fallback;
- candidate permutation equivariance;
- candidate-common residual and score-common-mode invariance;
- history-order sensitivity for overlapping evidence;
- two finite optimizer steps with nonzero gradients through write, probe,
  readout, and both Transformer blocks.

## Decision

**GO only to G1 learned synthetic. NO-GO to real data at present.**

The structural witness is strong enough to show that RWPU is not merely the
same-width ordinary terminal state under renamed variables. It does not show a
learned advantage over generic history attention, DeltaProduct-like recurrence,
or a parameter-matched FFN. G1 is therefore the cheapest honest falsifier. If
G1 fails any frozen control or corruption condition, the final decision becomes
STOP and this directory remains a negative design result.
