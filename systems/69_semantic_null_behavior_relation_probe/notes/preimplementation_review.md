# C69 preimplementation review

Decision: `authorize_exposed_formulation_gate_only`.

- The audit corrects the problem target from history sensitivity to behavioral
  relevance alignment.
- The model generalizes by LM content rather than item-ID lookup and uses the
  same graph/hyperparameters in both domains.
- Semantic-matched versus random-negative training is the sole tested
  difference.
- Anchoring removes unary shortcuts exactly.
- C69 is explicitly ineligible as the proposed architecture even if positive.

Authorized: C47 fit histories, six fixed GPU seeds, label-free scoring of the
already-open C47-A roles, and one post-A0 aggregation of their already-open
train labels.

Forbidden: C47 reserve, any dev/test record or qrel, alternative negative
cost, query temperature, model width/depth, aggregation, label-selected
checkpoint, or second attempt after a scientific result.
