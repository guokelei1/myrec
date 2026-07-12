# C40 Metric-Coupled Transport Transformer

C40 tests one architecture hypothesis: identity specificity is lost when
independent `Q/K/V/O/FFN` maps select history in one coordinate system, rewrite
it in another, and rank it in a third. Each C40 head instead uses one residual
semantic map for query, history, candidate, attention, transported value, and
final readout.

This is an LM/Transformer ranker, not a dataset rule. It has no category,
query-type, user-ID score, fixed-score router, pair MLP, scalar candidate gate,
tangent projection, or halfspace projection. Cached frozen-BGE states are an
exact execution optimization of the in-model LM encoder.

Only the data-free design gate is currently authorized. A real train-internal
gate requires a separate frozen proposal, untouched cohort, label barrier,
matched controls, and execution lock after the design gate passes.
