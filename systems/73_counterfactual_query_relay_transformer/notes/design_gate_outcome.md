# C73 data-free design-gate outcome

Decision: `failed_design_gate_terminal`.

All mechanical and evidence contracts passed in all three fixed GPU seeds.
Every parameter group received gradients, every loss decreased, parameter
counts matched, candidate permutation and deterministic rescoring were exact,
no-history returned the base exactly, repeat returned item-only exactly, and
wrong, shuffled, coarse-value, and query-masked histories all destroyed the
primary's clean gain.

The query-relay primitive learned useful structure, but missed two binding
architecture-rent conditions:

| Seed | Base | Primary | Gain | Primary - late | Primary - pooled | Primary - factual |
|---:|---:|---:|---:|---:|---:|---:|
| 20265001 | 0.719691 | 0.770160 | +0.050469 | +0.055686 | +0.021760 | +0.116287 |
| 20265002 | 0.719691 | 0.720899 | +0.001208 | +0.034842 | +0.018373 | +0.142803 |
| 20265003 | 0.719691 | 0.769730 | +0.050039 | +0.040472 | +0.020419 | +0.147229 |

The frozen absolute gain minimum was `+0.10`, missed by every seed.  More
importantly, token-resolved relay beat the pooled query-relay reduction by only
`+0.0184--+0.0218`, below the locked `+0.025` requirement in every seed.  One
seed also generalized almost exactly at the base despite fitting cleanly.

Thus routing history through the current query is a better inductive path than
late direct history attention or factual-only query relay on the constructed
shift, but token-resolved counterfactual relay has not paid stable incremental
rent over a much simpler pooled query transport.  C73 closes before any
repository record, label, pretrained-LM fit, dev, test, qrel, or shared
evaluator call.

No threshold, seed, generator, width, step, nuisance, or real-data rescue is
authorized.  A successor must introduce a genuinely new constraint that makes
the two-hop relay direction identifiable across its two attention stages; it
may not merely tune C73 or relabel the uniformly positive but sub-threshold
pooled margins.

Authoritative report:
`reports/pps_c73_counterfactual_query_relay_design_gate.json`, SHA-256
`072612f150cd0f88df6a11d90ca73a4b3bab61cfbba32f9b214b6fcc2b7fd233`.
