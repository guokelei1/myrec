# C27 nearest neighbours and reduction audit

| Neighbour | Overlap | Required distinction/control |
|---|---|---|
| PRM / SetRank / self-attentive document interaction / CMC | candidate-set self-attention and listwise context | `generic_contest` and ordinary candidate interaction are explicit controls; generic listwise gain is not C27 evidence |
| RAISE dynamic Transformer | user intention dynamically changes candidate interactions and Q/K/V transforms | C27 does not claim dynamic personalization as new; its only eligible distinction is an odd evidence-difference comparator with base-order-preserving contest readout |
| DirectRanker / RankNet family | learned pairwise preference, including antisymmetry | `candidate_contest` isolates history-free pairwise ranking; antisymmetry alone pays no novelty rent |
| differentiable sorting networks / SoftSort | smooth pairwise comparisons and differentiable ranking | C27 uses no sorting network and claims no new relaxation; soft-Borda is only the registered load-bearing readout |
| C24 set-attention probe | candidate competition | C24 edges were rank-inert; C27 forbids independent node scores in the primary and tests the pair margin directly |
| C26 token bridge | shared-query-token history evidence | `additive_node` is the exact readout-neighbour control; primary must improve because of contest placement, not tokenization |

Primary sources checked before implementation:

- https://arxiv.org/abs/1904.06813
- https://arxiv.org/abs/1912.05891
- https://arxiv.org/abs/1910.09676
- https://arxiv.org/abs/2201.05333
- https://arxiv.org/abs/1909.02768
- https://proceedings.mlr.press/v139/petersen21a.html
- https://proceedings.mlr.press/v119/prillo20a.html
- https://aclanthology.org/2024.emnlp-main.1242/

Verdict: candidate-set interaction, user-dynamic attention, antisymmetric
pairwise ranking, and differentiable comparison are all established.  C27 is
only a matched-control signal gate for the evidence-difference + odd-contest
placement.  Global novelty is uncertain and cannot be claimed before a positive
mechanism result and a deeper review.
