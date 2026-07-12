# C25 nearest neighbours and reduction audit

| Neighbour | Overlap | Non-reduction / required control |
|---|---|---|
| Tri-Attention (Yu et al., 2022) | explicit query-key-context relevance and contextual values | generic tri-attention retains arbitrary lower-order effects; C25 is the shared-potential third difference. Direct trilinear is a mandatory control. |
| HyperAttention (Ustaomeroglu & Qu, ICML 2025) | multi-entity higher-order attention | covers generic n-way dependence; it does not impose anchored lower-order annihilation. C25 claims a restricted inductive bias, not greater expressivity. |
| functional ANOVA / purified interactions | separates main and interaction effects | primarily output/function decomposition, often distribution-dependent or post hoc; C25 computes a null-anchored Boolean-lattice coefficient per event before Transformer aggregation. This remains a close conceptual neighbour. |
| Möbius interaction recovery (Kang et al., NeurIPS 2024) | Möbius coefficients identify higher-order variable sets | recovery/interpretation rather than a ranking attention block; the algebra itself is prior art. |
| C01 counterfactual contract | triadic event Transformer and corruption twins | C01 outcome was invalid because its D2p anchor was wrong; C25 uses registered D2p and removes lower-order states structurally instead of thresholding a certificate. |
| C04 paired prefix delta | factual-minus-null LM state/logit | a two-branch history delta keeps query-history and candidate-history terms; `joint_delta` is the explicit C04-like control. |

Primary sources checked before implementation:

- https://arxiv.org/abs/2211.02899
- https://proceedings.mlr.press/v267/ustaomeroglu25a.html
- https://proceedings.mlr.press/v108/lengerich20a.html
- https://proceedings.neurips.cc/paper_files/paper/2024/hash/520b379123d16e41f85472e766846486-Abstract-Conference.html

Reduction verdict: an unrestricted `M(q,c,h)` can be simulated by a generic
MLP/tri-attention, so C25 has no expressivity novelty claim.  Its testable
difference is the exact shared-potential cancellation and absence of every
lower-order residual bypass.  Performance must beat all three registered
controls; otherwise the restriction pays no architectural rent.
