# C51 exposed formulation outcome

The fixed cross-event affinity covariance passed every numerical contract on
both exposed domains: it was finite, deterministic, candidate-equivariant,
history-permutation invariant, and exactly zero with fewer than two history
events.  Its utility gate nevertheless failed.

| domain | covariance | base | uncentered moment | plain KRR | posterior | softmax |
|---|---:|---:|---:|---:|---:|---:|
| KuaiSearch | 0.303545 | 0.300870 | 0.307338 | 0.310162 | 0.307291 | 0.306967 |
| Amazon-C4 | 0.254753 | 0.253202 | 0.254502 | 0.268905 | 0.277001 | 0.276888 |

On KuaiSearch the primary beat base by `+0.002675` (95% bootstrap interval
`[+0.000908,+0.004888]`) with every hash fold positive, but it lost to the
uncentered moment by `-0.003793` and to plain KRR by `-0.006617`.  Its clicked
direction and clicked true-minus-wrong intervals crossed zero.  On Amazon its
`+0.001551` over base crossed zero, one fold was exactly zero, and it lost to
posterior and softmax by about `-0.0222`; true-minus-wrong also crossed zero.

Decision: `failed_formulation_terminal`.  Close C51 before training or fresh
reserve.  Do not tune covariance scale, centering strength, normalization,
history-length thresholds, or domain mixtures.  Together with C47--C50, this
ends the current search over fixed pooled-state readout statistics.  A
successor must intervene while token representations and candidate competition
are formed inside the Transformer, with the corresponding simpler scalar
readouts retained as controls.
