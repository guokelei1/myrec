# C18 mechanism fingerprint

## Operator and intervention

- candidate ID: `c18`
- name: Evidence-Constrained Order Transformer (`ECOT`)
- intervention locus: final candidate-logit operator inside the Transformer
  ranking core, after semantic history interaction and before listwise loss;
- persistent state: ordinary token states plus a request-local convex order
  set; there is no free candidate×event ledger;
- training signal: listwise target loss through an unrolled Euclidean
  projection; the soft-penalty control receives the same inequalities;
- inference inputs: query, candidate set, strictly prior history, masks and
  exact item equality already present in the unified interface;
- online external LLM calls: zero.

## Collision-resistant properties

For fixed anchor `a` and repeat relation `r`, the exact projection `P_K` has
four jointly load-bearing properties:

1. **feasibility:** every protected margin is satisfied;
2. **idempotence:** `P_K(P_K(y))=P_K(y)`;
3. **minimality:** no other feasible score vector is closer to `y` in L2;
4. **active-set coupling:** when a constraint activates, its two candidates
   receive equal/opposite corrections and later active constraints alter the
   joint Jacobian.

An elementwise gate, scalar mixture, FiLM/hyperadapter, ordinary attention
value, or one-pass candidate flow does not satisfy this quartet for all `y`.
The closest generic class is a differentiable quadratic-program/optimization
layer; C18 does not claim to invent that class.  Its falsifiable architecture
claim is the PPS-specific recurrence-anchored order set and the separation of
hard identity evidence from learned semantic transfer.

## Degenerations and matched controls

| mode | degeneration | purpose |
|---|---|---|
| `projection` | full primitive | hard order-cone projection |
| `direct` | `s=y` | same Transformer/capacity, no order constraint |
| `soft_penalty` | `s=y` plus hinge loss | tests whether an ordinary training penalty is sufficient |
| `anchor_only` | `s=a` | exact-recurrence control without semantic transfer |
| `base_only` | `s=b` | query/candidate control |

The three trainable modes instantiate identical parameters and initialization;
only their ranking operator/loss differs.  `anchor_only` and `base_only` are
deterministic same-forward counterfactuals of the trained model.

## Complexity

For this two-group partial order, exact Euclidean projection reduces to a
one-dimensional isotonic threshold: clamp repeated deltas below the threshold
upward and non-repeated deltas above it downward.  Fixed bisection costs
`O(T C)` time and `O(C)` activation memory per request after the shared
Transformer.  A real-data gate, if ever authorized, must still profile packed
candidate counts and reject the candidate if projection plus corruption
diagnostics cannot fit the registered budget.
