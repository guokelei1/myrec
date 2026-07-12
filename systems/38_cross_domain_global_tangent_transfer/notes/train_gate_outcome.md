# C38 cross-domain global tangent transfer terminal outcome

C38 terminated at train-internal A1.  The authoritative report SHA-256 is
`c8352c8cfc22dc7ab35fdf5c70fdd7f9e1a12b120cc97bd616ca54f9b0e2157e`.
The proposal and execution lock SHA-256 values are respectively
`bba8d1c25c230b4562057e886610e4f4bf73b4d4cc550ab13f765cc2bdabe717`
and `63962da0cc513a998126422cd5345ff93f86da3d5358c21d658d082279b0e638`.

Amazon-C4 C0 and C1 passed before the proposal lock.  C0 exposes only
nonblank-title history events to every method: source coverage is 93.07%, the
drop fraction is 6.93%, consumed coverage is 100%, no train request becomes
empty, and retained train history length has median 35.  The label-free cohort
contains 6,000 fit, 1,200 internal-A, 1,200 delayed-B, and 1,200 escrow
requests.  Wrong histories have 100% same-length-bin coverage and zero
same-user assignments.  G0 passed all 11 checks while opening only fit labels.

Three physical A40 GPUs trained the three 12,288-parameter modes under paired
initialization.  All 22 A0 checks passed.  Both adapter matrices received
nonzero gradients and changed; deterministic, candidate-permutation,
no-history, no-query, and repeat-correction errors were exactly zero.  Primary
tangent orthogonality error was at most `1.12e-7`.  The primary changed 99.25%
of complete orders and top-10 sets versus the frozen base.  Both reductions
were structurally distinct on more than 91% of top-10 sets.

A1 strongly validates cross-domain personalized history signal but rejects the
defining tangent projection.  Seed-averaged NDCG@10 is 0.243638 for the frozen
BGE base, 0.307682 for mean-history tangent, 0.317952 for query-attended
tangent primary, and 0.329332 for query-attended unprojected transport.  The
primary exceeds base by `+0.074314`, with 95% paired-bootstrap CI
`[0.047307, 0.101719]`; every seed and request-hash fold is positive.  It
exceeds mean-history tangent by `+0.010269`, CI `[0.000390, 0.019900]`, so
query-conditioned history selection pays independent rent.  True history
exceeds the matched wrong-user history by `+0.041688`, CI
`[0.030953, 0.052623]`, and the clicked-versus-negative correction direction
CI is strictly positive.

However, the primary is worse than the equal-capacity unprojected reduction by
`-0.011381`, CI `[-0.020650, -0.002271]`, and all three per-seed differences
are negative.  Therefore C38 has status `failed_A1_terminal`: removing the
query-parallel component is not load-bearing and is actively harmful in this
independent regime.  No temperature, scale, loss, threshold, encoder, cohort,
or candidate-pool rescue is authorized.  Delayed-B, escrow, upstream dev, and
upstream test remain unopened.

The transferable facts are narrower than a C38 architecture claim: strict-past
history carries causal ranking value across language and candidate
construction; query-conditioned history aggregation beats uniform history;
and tangent geometry should be removed.  The next candidate must operate at
the Transformer's representation/value-learning interface and use the simple
unprojected shared write as a strong control.  It must not rename that control
or continue geometric projection variants.
