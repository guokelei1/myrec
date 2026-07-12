# C73 locked data-free design gate

## Purpose

Test whether the counterfactual query relay supplies a useful inductive bias
that its closest internal reductions do not.  This is not a benchmark result.

## Synthetic information contract

Each example contains four current-query facet tokens, eight ordered history
events, and twelve candidate tokens.  Relevant events bind a query facet key
to a preference value; the correct non-repeat candidate requires the two-hop
composition `query facet -> historical value -> candidate`.  A nuisance event
outside the query facets points to the correct candidate during fitting and to
a distractor only in validation.  This makes direct history shortcuts fail
under a frozen shift while preserving the same causal rule.

Twenty percent of examples have no history and twenty percent contain an exact
repeat.  Their supplied base/item-only rankings are exact targets and must be
preserved.  The remaining examples are supported non-repeat cases.

## Fixed modes

- `counterfactual_query_relay` (primary)
- `late_state_difference`
- `pooled_query_relay`
- `factual_query_relay`

Every mode instantiates the same parameters, initialization scheme, optimizer,
batch order, steps, and candidate set.  Only the information-flow formula
changes.

## Corruptions

- wrong history: deterministic cross-example history rotation;
- shuffled history: deterministic nontrivial event permutation;
- coarse history: preference-value coordinates removed while keys remain;
- query mask: personalized path disabled;
- no history and repeat: exact fallback checks.

## Binding decision

All three seeds must satisfy every threshold in `configs/design_gate.yaml`.
In particular, primary must beat base and all three reductions on shifted
supported non-repeat examples; corruption gain retention must remain below its
frozen maximum; repeat/no-history must be essentially perfect; exact fallback,
determinism, candidate permutation, finite loss, and active-gradient contracts
must pass.

Any failure is terminal for C73.  The generator, nuisance shift, modes, steps,
width, seeds, and thresholds may not be changed after outcome.
