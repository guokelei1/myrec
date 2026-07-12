# C06 nearest-neighbor and reduction audit

Verdict before outcomes: **uncertain, high novelty risk**. Primary
sources below establish that most ingredients are known separately.

| Neighbor | What is already known | Narrow C06 difference | Required control |
|---|---|---|---|
| [HodgeRank](https://arxiv.org/abs/0811.1067) | pairwise edge flows, gradient/cyclic decomposition and global ranking potentials | C06 derives candidate-local trust from incident gradient/cycle energy, then permits only a symmetrically trusted projected gradient into LM logits | `t=1`, global-event trust, potential and cycle-sign controls |
| [Sparse Pairwise Re-ranking](https://arxiv.org/abs/2207.04470) | Transformer pair preferences and additive/Borda-style aggregation into a ranking | history-indexed low-rank factors avoid explicit `C^2`; cycle direction is discarded and only candidate-local trust calibrates the projected flow | matched pairwise-additive Transformer |
| [DirectRanker](https://arxiv.org/abs/1909.02768) | reflexive, antisymmetric, transitive pairwise neural ranking by shared-score differences | C06 conditions comparisons on ordered history and uses non-potential energy only as local trust, never as a preference sign | potential/shared-score restriction |
| [GNNRank](https://proceedings.mlr.press/v162/he22b.html) | neural global ranking recovery from directed pairwise-comparison graphs | C06 builds a request-local candidate graph from LM history evidence rather than an observed global graph | graph-ranking nearest neighbor |
| [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html) / [SetRank](https://arxiv.org/abs/1912.05891) | attention-based permutation-equivariant set representations and candidate-context ranking | C06 restricts the history contribution to skew transfer plus bounded divergence | matched generic set block |
| [Context-Aware LTR](https://arxiv.org/abs/2005.10084) / [PRM](https://arxiv.org/abs/1904.06813) | Transformer self-attention over candidate lists and multivariate scores | no common-mode or score-bound guarantee in the generic attention block | centered context-aware attention |
| [MIR](https://arxiv.org/abs/2204.09370) | candidate-set and ordered-history interactions, including set-to-list attention | closest task neighbor; C06's only differentiator is the candidate-local Hodge-trust restriction on the final history path | MIR-style matched block |
| [STARank](https://arxiv.org/abs/2308.02860) | history-contextual set-to-arrangement ranking and Plackett-Luce consistency | C06 retains scalar scores and explicitly does not claim subset/internal consistency | nested candidate-pool audit |
| [DIN](https://arxiv.org/abs/1706.06978) / [Denoising Attention](https://arxiv.org/abs/2308.15968) | target-conditioned history attention and filtering | potential-flow C06 degenerates to centered target attention and must beat it | centered unary/attention control |
| [Rankformer](https://arxiv.org/abs/2503.16927) | ranking-objective inductive bias inside a graph Transformer | ranking-aware architecture alone is not novel; the testable restriction is conservative history-to-logit flow | matched graph/Transformer capacity |

## Claims explicitly forbidden

- “first pairwise Transformer ranker”;
- “first antisymmetric ranking network”;
- “first graph-flow/Hodge ranking method”;
- “first permutation-equivariant set reranker”;
- “zero-sum logits are globally novel.”

The only defensible pre-outcome statement is that the **combination is an
architecture hypothesis**, not that its ingredients are new. It becomes a
credible contribution only if local Hodge trust beats the same flow with
`t=1`, a direct learned candidate gate, the rejected global-event gate,
centered-attention, pairwise-additive and MIR/SetRank controls on a fresh locked
cohort. Otherwise C06 is a useful structural repair of C05, but not a new
ranking architecture contribution.
