# C41 nearest-neighbor audit

| Neighbor | Covered idea | Consequence for C41 |
|---|---|---|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Keys route while values carry content | Routing/content separation is not novel |
| [Simplifying Transformer Blocks](https://openreview.net/forum?id=RtDok9eS3s) | Removing or fixing value/projection parameters, including identity V/O | Immutable values/projections cannot carry the claim |
| [Do Transformers Need Three Projections?](https://arxiv.org/abs/2606.04032) | QKV projection-sharing variants and K=V | Fewer/shared projections are not the claim |
| [QKVAE](https://aclanthology.org/2022.naacl-main.423/) | Explicit interpretation of keys as selection and values as conveyed information | The semantic-carrier interpretation has precedent |
| [DIN](https://arxiv.org/abs/1706.06978) | Candidate-aware history attention for recommendation | A later candidate-conditioned router needs DIN/target-attention controls |
| C05 target-attention probe | Learned target attention failed through common-mode output collapse | C41 changes the carrier/readout boundary but cannot rewrite C05 as positive |
| C38 unprojected transport | Query-conditioned adapted semantic history is strongly useful on Amazon | Mandatory strong control on the untouched cohort |
| C40 selection-only | Conditional synthetic winner with the exact C41 primary function | Provides design-gate evidence, not real-data validation |

Pre-outcome novelty verdict: **boundary only**. Real utility is still worth
testing because it determines the strongest architecture foundation, but a
positive result requires a later non-reducible primitive before paper promotion.
