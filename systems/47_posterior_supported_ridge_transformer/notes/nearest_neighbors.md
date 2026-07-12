# C47 nearest-neighbor audit

| Neighbour | Direct overlap | Binding C47 boundary |
|---|---|---|
| Cubit | closed-form KRR replaces Transformer attention | plain-ridge mode is an exact mandatory control; KRR itself is not C47 |
| Sparse attention as kernel regression | interprets attention as nonparametric kernel regression | kernel choice/sparsity alone cannot carry C47 |
| DeltaNet / Gated DeltaNet | associative fast weights updated by prediction error, with gating/erasure | sequential fast-weight controls; C47 uses one batch normal equation and candidate posterior support |
| Titans / TTT / GradMem | loss-driven or gradient-updated test-time neural memory | C47 performs no test-time optimizer step and claims no memory novelty |
| Sparse GP Attention / CGP Transformer | GP posterior/uncertainty inside attention | GP uncertainty is prior art; only self-supported conservative write remains testable |
| Transformer posterior-predictive constructions | Transformers can compute GP mean and variance | computing support is not new; multiplying the mean write by its own support must pay rent |
| DIN / target attention | candidate-conditioned history selection | ordinary softmax attention is a mandatory trained control |
| C31/C38/C43 | query-attended history transports a shared query state | C47 replaces normalized mean transport with a normal-equation geometry and candidate self-support |

Primary sources:

- Cubit: https://arxiv.org/abs/2605.06501
- Sparse attention as compact kernel regression: https://arxiv.org/abs/2601.22766
- DeltaNet: https://arxiv.org/abs/2406.06484
- Gated DeltaNet: https://arxiv.org/abs/2412.06464
- Titans: https://arxiv.org/abs/2501.00663
- Sparse GP attention: https://openreview.net/forum?id=jPVAFXHlbL
- Correlated GP Transformer: https://openreview.net/forum?id=xlIK0vu3MW
- Transformer posterior predictive distributions:
  https://openreview.net/forum?id=jfKliftgkl
- DIN: https://arxiv.org/abs/1706.06978
