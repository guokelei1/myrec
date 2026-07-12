# C53 nearest neighbours and claim boundary

| neighbour | relationship |
|---|---|
| [Personalized Re-ranking Model](https://arxiv.org/abs/1904.06813) | directly establishes Transformer self-attention over a candidate list; C53 cannot claim this as new |
| [PEAR](https://arxiv.org/abs/2203.12267) | directly models initial-list and clicked-history contexts with a contextual Transformer; strongest architecture neighbour |
| [Set Transformer](https://openreview.net/forum?id=Hkgnii09Ym) | establishes permutation-equivariant attention over sets |
| [TEM](https://arxiv.org/abs/2005.08936) | establishes Transformer query/history personalization in product search |
| C02 | strong-base internal history update, but each candidate is processed independently |
| C24 | candidate-set attention only on multi-recurrence candidates; cross edges were not load-bearing |
| C27/C28 | token evidence was pooled into pairwise scalar contests before listwise aggregation |

Novelty status: `known-foundation-control`.  C53 cannot become the paper
innovation regardless of outcome.  Its purpose is to prevent another novel
mechanism from being credited for capacity that an ordinary joint-context
Transformer already has—or from being built where that foundation has no
stable signal.
