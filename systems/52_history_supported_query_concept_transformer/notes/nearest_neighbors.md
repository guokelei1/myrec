# C52 nearest-neighbour audit

| neighbour | covered mechanism | C52 boundary / binding test |
|---|---|---|
| [Cubit](https://arxiv.org/abs/2605.06501) | replaces attention token mixing with kernel ridge regression | C52 does not claim KRR; pooled Cubit-style ridge is binding, while C52 uses the projection only as an internal query-concept logit bias |
| [ColBERT](https://arxiv.org/abs/2004.12832) | query/document token late interaction | candidate token matching is inherited machinery, not the claim |
| [Personalized Re-ranking Model](https://arxiv.org/abs/1904.06813) | Transformer self-attention across a candidate list | generic listwise interaction is a future matched backbone/control, not C52's primitive |
| [PEAR](https://arxiv.org/abs/2203.12267) | contextual Transformer over initial candidates and clicked history | directly covers generic joint history/candidate contextualization; C52 must beat a no-evidence/generic-attention control |
| [TEM](https://arxiv.org/abs/2005.08936) | query/history Transformer personalization for product search | covers target-aware semantic history attention, represented by the fixed/token-softmax controls |
| [Slot Attention](https://proceedings.neurips.cc/paper/2020/hash/8511df98c02ab60aea1b2356c013bc0f-Abstract.html) | inputs compete over exchangeable slots | C52 has no history-to-candidate allocation or candidate-axis softmax |
| [Differential Transformer](https://proceedings.iclr.cc/paper_files/paper/2025/hash/00b67df24009747e8bbed4c2c6f9c825-Abstract-Conference.html) | subtracts two softmax attention maps | C52 adds a KRR evidence coordinate before one query-token softmax; it does not subtract learned maps |
| C26 | query-token-pivoted candidate/history value bridge followed by bounded scalar residual | C52 forbids history values and intervenes before semantic query-token allocation |
| C47 | pooled history KRR and posterior candidate support | exact pooled plain/posterior scores are binding controls |
| C49/C50 | learned behavioral/prediction-error values | C52 keeps values in frozen LM semantic coordinates and changes only routing |

Novelty status before outcome: `distinct-composition-with-high-uncertainty`.
Even a positive exposed formulation would authorize only a trainable matched-
control gate, not a global novelty claim.
