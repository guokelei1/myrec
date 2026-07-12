# C65 counterfactual residual state: G0 terminal

C65 responded to C64's generic-reranker shortcut by exposing only the internal
factual-minus-NULL candidate-state residual and making wrong-history residuals
rank-neutral during training.  Four equal-parameter modes and a stopped NULL
reference were frozen before labels.

Real-token G0 showed that all intended paths were active and differentiable,
but the state subtraction plus LayerNorm magnified candidate serialization
roundoff to `3.64e-5`.  Because the candidate-permutation contract was binding,
C65 stopped before training labels.  No utility or wrong-neutral outcome exists.

A successor is justified only as a C30-style numerical continuation: stable
item-ID canonicalization around all factual/NULL/wrong branches, with every
scientific setting unchanged.  Any other change would be an outcome-informed
rescue.
