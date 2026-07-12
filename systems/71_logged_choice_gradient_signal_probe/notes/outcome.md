# C71 outcome — invalid zero-positive target surface

Decision: `invalid_zero_positive_target_surface_terminal`.

C71's label-free computation was valid and strongly active. All A0 checks
passed under execution lock
`07d7c7290099c3ce5a7866cd532e479c429635c8fb921c933c04813f84f53c60`:
the 9,349 cached episode values were finite, every choice gradient was nonzero,
primary correction RMS was `0.054678`, true/wrong difference RMS was
`0.068095`, and 90.33% of requests changed complete order. Determinism,
candidate permutation, and no-history differences were exactly zero.

Only after A0 did the aggregator open the 600 target rows' registered
`clicked` labels. All 600 had zero clicked positives. A subsequent scoped audit
also found zero purchased positives: every target was a `(click=0,purchase=0)`
row. The shared clicked-direction statistic correctly refused an empty surface.

This happened because the 96,939-request historical packed pool is precisely
the label-bearing subset used by prior diagnostics; the 66,778 standardized
train requests outside it do not provide a positive outcome surface. C71 is
therefore neither a positive nor negative result for logged-choice gradients.
Changing the target label, selecting on hidden positivity, or moving to another
fresh-unpacked cohort after opening these labels would violate the frozen gate.

The only valid next diagnostic may use an explicitly exposed fit-label role
and must be described as formulation evidence, never independent confirmation.
Dev, test, qrels, and source-episode labels remain closed.
