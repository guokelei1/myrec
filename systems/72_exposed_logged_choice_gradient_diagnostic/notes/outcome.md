# C72 exposed formulation outcome

Decision: `failed_exposed_formulation_terminal`.

All label-free mechanics passed. The logged-choice correction was strongly
rank-active: 96.33% of requests changed complete order, every episode gradient
was nonzero, primary correction RMS was `0.050954`, and true/wrong correction
difference RMS was `0.064756`. Determinism, candidate permutation, and
no-history differences were exactly zero.

On 600 C47 fit requests whose labels were already exposed, primary NDCG@10 was
`0.287066`, versus query base `0.276478`, positive-only `0.287712`, uniform
slate `0.286938`, semantic history `0.285356`, and wrong history `0.282233`.
Primary-minus-base was `+0.010588`, but its interval
`[-0.000689,+0.022254]` crossed zero. Primary was nominally below
positive-only, essentially tied with uniform-slate, and had no stable
true-over-wrong interval. Clicked correction direction was positive but its
lower interval endpoint was slightly negative.

Thus actual historical-slate subtraction does not pay incremental rent over a
positive-only history value. The activity proves the formula is operational;
the control result closes its role as C70's missing signed preference object.
No temperature, normalization, scale, donor, subset, label, or second exposed
diagnostic is authorized. C70 remains unimplemented; dev/test/qrels remain
closed.
