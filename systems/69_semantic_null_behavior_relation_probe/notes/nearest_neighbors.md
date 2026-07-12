# C69 nearest-neighbor boundary

C69 makes no paper-level novelty claim. Behavioral language representation is
an established family.

| Neighbor | Coverage | C69 consequence |
|---|---|---|
| Recformer | language representations trained for next-item sequence retrieval | direct strong family boundary; C69 is only a signal probe |
| BLaIR | recommendation-specialized item/language representation | mandatory external baseline context on Amazon; C69 cannot claim behavior-language pretraining itself |
| LMRec / language-space CF | maps LM item states into a behavior space with graph/contrastive learning | direct representation neighbor; any successor must identify a different internal ranking primitive |
| RecoBERT | Transformer item-pair compatibility from catalog text | two-item Transformer scoring is not novel |
| random-negative C46/C69 control | behavioral positives versus arbitrary negatives | primary must show that semantic matching, not extra capacity, removes the shortcut |
| C25 anchored interaction | anchored finite differences remove unary paths | C69's anchoring is a correctness device, not claimed novelty |
| hard-negative contrastive learning | semantically close negatives sharpen relations | established training method; this gate only asks whether the information exists here |

Primary sources:

- Recformer: https://arxiv.org/abs/2305.13731
- BLaIR: https://arxiv.org/abs/2403.03952
- Language representations for recommendation: https://arxiv.org/abs/2407.05441
- RecoBERT: https://arxiv.org/abs/2009.13292

If the probe passes, the architecture contribution still remains unsolved.
