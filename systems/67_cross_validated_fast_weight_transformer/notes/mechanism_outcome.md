# C67 data-free outcome

Status: **failed terminal before repository data**.

All three seeds and four equal-parameter modes trained for 500 steps with
finite loss, decreasing loss, and active gradients through the pair
Transformer, projections, fast-weight initialization, and inner step. Candidate
permutation, determinism, history-only writing, no-history fallback, and repeat
fallback were bit-exact.

Absolute synthetic accuracy was high but invalid as mechanism evidence. The
primary reached `0.958--0.967` noisy accuracy, while wrong history retained
`0.951--0.965`; the true-minus-wrong drop was only
`+0.0078/+0.0078/-0.0013` against a locked `0.20` minimum. Useful and nuisance
events both received approximately uniform `0.125` write mass. Unsupported
histories still produced correction RMS `1.69--1.74`, far above `0.20`.
Ordinary TTT, self-validation, and first-order gradient agreement all tied or
beat the exact held-out writer within the frozen margins.

The pair Transformer learned a generic query-candidate relation and made the
request-local learner similar across unrelated histories. Exact held-out
reconstruction therefore did not identify user-specific evidence. C67 is
closed; its high accuracy must not be cited as a positive result. No generator,
readout, nuisance ratio, step, dimension, threshold, seed, or real-data rescue
is authorized.
