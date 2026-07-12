# C48 formulation execution abort before labels or metrics

The first locked formulation command stopped on the first request while
checking reversed candidate order.  NumPy `[::-1]` produced a negative-stride
view and `torch.from_numpy` rejected it.  The primary/duplicate/wrong in-memory
operator calls for that first request had completed, but no C48 label loader,
metric, bootstrap, report, score artifact, fresh reserve, dev/test record, or
qrels was opened or written.

This is a serialization/layout defect only.  The authorized repair may add
`np.ascontiguousarray` at the NumPy-to-Torch boundary and must leave the
operator, cohort, controls, thresholds, seeds, folds, and all scientific
settings unchanged.
