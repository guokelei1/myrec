# C67 pre-implementation review

Decision: authorize one locked, data-free, three-seed GPU falsifier.

- Plain TTT/TTT4Rec/GradMem is rejected as the novelty claim and retained only
  as nearest controls.
- The primitive changes the internal state update rule, not a dataset feature,
  output threshold, loss coefficient, or fixed-score router.
- Query and candidates are excluded from writing fast weights; all
  personalization reads a history-derived state.
- Exact held-out validation, first-order agreement, self-fit, and ordinary TTT
  share parameters and differ only in the frozen write law.
- The synthetic task tests conditional capability under shared law plus
  nuisance and an unsupported-law abstention regime. It cannot establish that
  product histories contain this structure.
- No repository data, fit label, fresh role, dev, test, or qrels is authorized.
- Failure is terminal. A pass requires a second review before any real-data
  implementation.
