# Pre-outcome reduction audit

## Rejected predecessor

The initial C23 sketch used
`m_t=m_{t-1}+alpha_t(v_t-m_{t-1})`.  This is a scalar/vector delta rule and is
covered by DeltaNet/Gated DeltaNet; 2026 SinkRec already applies a
memory-conditioned Gated DeltaNet to recommendation.  Candidate/query inputs do
not prevent that algebraic reduction.  The predecessor was rejected before
code or outcome.

## RRST reductions

1. If the reset boundary is removed, every candidate sees the full history.
   The model is ordinary target-aware sequence attention.  This is the locked
   `unreset_history` control with identical parameters.
2. If suffix positions are removed, the encoder is permutation invariant over
   its event multiset.  This is `orderless_suffix`.
3. If query projections are structurally zero, the model is a candidate-aware
   recurrence calibrator, not a query-conditioned search architecture.  This
   is `query_independent`.
4. If the learned correction is zero, the exact score is registered static
   item-only on repeat requests and D2p otherwise.
5. A learned scalar over count/recency cannot reproduce suffix-token
   substitution sensitivity while remaining invariant to the masked prefix.
   The paired prefix/suffix intervention is therefore the mechanism witness.

## What a pass would and would not mean

A pass establishes that the candidate-specific reset graph pays empirical rent
over known reductions on repeat-present train-internal data.  It does not prove
global novelty, dev utility, non-repeat transfer or cross-dataset validity.
Any tie with `unreset_history` or `orderless_suffix` closes the architecture
claim even if RRST beats D2p.
