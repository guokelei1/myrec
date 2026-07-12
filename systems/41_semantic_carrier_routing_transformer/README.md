# C41 Semantic-Carrier Routing Transformer

C41 is an architecture-boundary candidate, not yet a novelty claim. It learns
multi-head query/history routing but carries every selected history value and
the candidate readout in the immutable pretrained-LM semantic coordinate.

The design is the consistently winning `selection_only` reduction from C40,
registered separately before any untouched real-data score or label is opened.
Identity values/projections and QKV simplification have direct precedents, so a
positive C41 result can establish a strong backbone but cannot by itself be
called the paper's architectural innovation.

The design gate is data-free. Real training requires a new proposal lock,
label-unopened Amazon cohort, feature-only stage, G0 label barrier, execution
lock, matched controls, and C38 unprojected strong-control comparison.
