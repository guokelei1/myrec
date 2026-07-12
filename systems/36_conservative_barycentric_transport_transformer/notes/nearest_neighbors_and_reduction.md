# C36 nearest neighbours and reduction boundary

| Neighbour | Covered mechanism | C36 boundary / control |
|---|---|---|
| [Centered Self-Attention](https://arxiv.org/abs/2306.01610) | shifts token-axis attention rows to zero sum to address oversmoothing | C36 centers admitted candidate-specific **value writes** over the ranking candidate set while preserving a separate shared write |
| [ZeroS](https://arxiv.org/abs/2602.05230) | zero-sum temporal attention residuals for contrastive expressivity and linear attention | C36 is not a linear-attention efficiency method; it enforces a candidate-axis barycenter and a request-wise non-reversal bound |
| [Differential Transformer](https://arxiv.org/abs/2410.05258) | subtracts two softmax attention maps to cancel noisy context | C36 has one authenticated history map, an explicit global anchor, and algebraic candidate-set conservation |
| [Slot Attention](https://arxiv.org/abs/2006.15055) | exchangeable slots compete for inputs | candidate competition alone was C35's control/failure; C36 tests conservation around a shared query-history write |
| [Candidate-aware user modeling](https://arxiv.org/abs/2204.04726) | candidate-conditioned history aggregation | `relative_surplus_only` is the direct candidate-aware reduction; it lacks the shared anchor and conservation law |
| [BeliefFormer](https://openreview.net/forum?id=Ard2QzPAUK) | orthogonally projected attention residuals | query-tangent projection is inherited from C32; the new primitive is candidate-axis conservation and non-reversal |
| [RealFormer](https://arxiv.org/abs/2012.11747) | residual attention logits across Transformer layers | C36 composes value/residual writes within one request, not attention logits across depth |
| C32/C33 | candidate-shared authenticated tangent query transport | `global_tangent_transport` is the exact in-run reduction |
| C35 | candidate-relative tangent surplus without a shared anchor | `relative_surplus_only` is the exact in-run reduction |

The proposal does not claim that centering, tangent projection, trust regions,
or candidate awareness is individually new. Its falsifiable architecture claim
is the combination of an exactly preserved shared history write with a
zero-mean, per-request norm-bounded candidate deviation. A matched reduction
meeting or beating the primary means C36 paid no mechanism rent.
