# C64 nearest-neighbor boundary

C64 is deliberately a known-architecture probe, so it makes no novelty claim.

- Full cross-encoders and late-layer fine-tuning are standard neural ranking
  techniques.
- C53 is the binding frozen-state architecture control; C64 keeps its joint
  candidate/history information graph and changes only whether pretrained token
  layers adapt.
- `adaptive_query_candidate_lm` distinguishes generic extra LM tuning from
  history value.
- `frozen_history_lm` distinguishes representation learning from the joint
  context head.
- A future proposal may cite C64 only as a signal prerequisite.  It may not
  rename late-layer fine-tuning or concatenation as the paper's innovation.
