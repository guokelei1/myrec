# C17 algebraic reduction audit

## 1. A learned persistent ledger is an Edge Transformer

Let the node partitions be candidates `c_i` and history events `h_j`.  A tensor
`L_ij` that stores a learned vector for every pair is exactly a vector-valued
state on the bipartite edge `(c_i,h_j)`.  Any update

```text
L_ij' = F(L_ij, x_i, h_j, aggregate_k G(L_ik), aggregate_r R(L_rj))
```

is edge-conditioned message passing.  Adding triangular interactions through
another candidate or event is the defining higher-order update of an Edge
Transformer.  Persisting the state for several LM layers changes depth, not
the operator class.

## 2. A chain-rule-tied ledger is attribution, not a new ranker

Introduce one scalar source gate `alpha_j` per event and define

```text
L_ij^l = partial x_i^l / partial alpha_j.
```

Propagating `L` through attention, residuals, normalization and FFNs by their
exact Jacobians is forward-mode automatic differentiation.  The final ledger
is `partial s_i / partial alpha_j`.  It is faithful local provenance, but if the
ordinary score `s_i` remains the output, the forward function and ranking are
unchanged.  Attention rollout/flow and gradient-based attribution are direct
neighbours of this branch.

Integrated or pathwise variants do not rescue the primitive.  With a complete
path decomposition, completeness reconstructs the factual-minus-null output.
Without modifying the reconstructed sum, this is still only an explanation.

## 3. Letting provenance control the score becomes a gate

Any rule of the form

```text
Delta s_i = sum_j g(L_ij, x_i, h_j) v_ij
```

is an eventwise attention/message value with an attribution-conditioned gate.
If `g` is scalar it is directly an event gate; if `g` emits a vector it is the
generic joint nonlinear pair message already identified in C15.  Conservation,
NULL handling, or a bounded scale constrain the gate but do not alter this
factorization.

## 4. A margin ledger returns to closed flow or pairwise ranking

For antisymmetric entries `M_ik=-M_ki`, the canonical scalar readout is

```text
Delta s_i = sum_k M_ik.
```

This is graph divergence, the same candidate-relative score geometry tested by
C06.  If the divergence is not used and cyclic margins directly determine a
tournament ranking, the model becomes an ordinary pairwise/edge ranking
network and requires a separate rank-aggregation primitive.  Persisting the
cycle state does not make the history ledger distinct from an Edge Transformer.

## Binding consequence

The free ledger is already a known edge-state architecture.  The only strict
structural escape—tying it to the content Transformer's exact Jacobian—removes
its ability to change predictions.  Every way of restoring predictive effect
reintroduces a registered gate, pair message, or candidate flow.  C17 therefore
has no implementation-worthy fingerprint.
