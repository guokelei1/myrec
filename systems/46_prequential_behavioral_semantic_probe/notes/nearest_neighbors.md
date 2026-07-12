# C46 nearest-neighbor audit

| Neighbor | Overlap | C46 boundary |
|---|---|---|
| SASRec | self-attentive next-item prediction | C46 replaces item-ID lookup with frozen LM content; SASRec remains the sequence backbone, not a novelty claim. |
| S3-Rec | self-supervised Transformer sequence/item representation | C46 uses one next-item objective only and claims no new pretraining method. |
| UniSRec | text-based transferable item adapter and contrastive sequence learning | Directly close; a positive C46 result supports using UniSRec/Recformer-class states as a control, not claiming the family as new. |
| Recformer | language item representations and Transformer next-item retrieval | Directly close; C46 uses already pooled local LM states for a cheaper signal gate. |
| C26 | token-level query/history/candidate bridge | C46 changes the learned item relation before query conditioning; C26's additive token bridge remains a negative local result. |

Primary sources:

- SASRec: https://arxiv.org/abs/1808.09781
- S3-Rec: https://arxiv.org/abs/2008.07873
- UniSRec: https://arxiv.org/abs/2206.05941
- Recformer: https://arxiv.org/abs/2305.13731

Verdict: `known representation family; signal probe only`.
