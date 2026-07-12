# C37 nearest neighbours and reduction boundary

| Neighbour | Covered mechanism | C37 boundary / control |
|---|---|---|
| [Centered Self-Attention](https://arxiv.org/abs/2306.01610) | zero-sum token-axis attention to reduce oversmoothing | C37 conserves a separate shared value write while centering only admitted candidate-indexed residuals |
| [ZeroS](https://arxiv.org/abs/2602.05230) | zero-sum temporal attention residuals and negative weights | C37 operates on the ranking-candidate axis, not for linear-time attention, and has an explicit authenticated global anchor |
| [Differential Transformer](https://arxiv.org/abs/2410.05258) | difference of two softmax maps | C37 uses one history map and an active-set value-field conservation law |
| [Slot Attention](https://arxiv.org/abs/2006.15055) | competitive exchangeable slots | competition alone is the relative-only reduction; C37 additionally preserves a global history write |
| [Candidate-aware user modeling](https://arxiv.org/abs/2204.04726) | candidate-conditioned history aggregation | `relative_surplus_only` removes C37's shared anchor and barycentric decomposition |
| C32/C33 | authenticated global tangent query transport | `global_tangent_transport` is the exact in-run reduction |
| C35 | relative-only tangent surplus | `relative_surplus_only` is the exact in-run reduction |
| C36 | barycentric transport plus soft max-norm shrinkage | C37 deletes the A0-failed shrinkage and separately gates the surviving conservation operator |

C37 claims neither centering nor candidate awareness alone as novel. The tested
claim is an authenticated shared write plus an exactly mean-zero, active-set
candidate residual within an LM ranker. Any matched reduction meeting or
beating the primary closes that claim.
