# C29 nearest neighbors and reduction audit

No global novelty claim is made before empirical and broader literature review.

| Neighbor | Shared mechanism | C29-specific falsifiable difference |
|---|---|---|
| CLLM4Rec / item-embedding soft prompts | catalog/user information enters an LM token stream | C29 does not claim soft tokens; persistent memory cannot score and only sets a strict event attention mask |
| IDIOMoE (ICLR 2026) | separates catalog and language processing inside an LM | C29 has no token-type MoE; its primitive is temporal provenance admissibility and factual-null mediation |
| CoPS cognitive personalized search | persistent user memory plus LLM search ranking | C29 uses no profile generation/refinding router; the exact prior-memory membership mask is directly ablated |
| query-aware / denoising attention | filters user history for the current query | C29's admission is query-independent provenance, after which the LM remains query/candidate conditioned; query attention is a control, not authentication |
| C01 counterfactual evidence contract | true/wrong/null shared paths | C01's learned certificate could separate corruption without ranking direction; C29 has no certificate head and constrains the ranking residual directly |
| C04 paired prefix delta | factual-minus-null shared LM score | C29 keeps the registered D2p base exact and tests whether a causal event mask, not a null prefix alone, is load-bearing |
| C22 evidence filtration | block-triangular protection inside Transformer layers | C22 protected recurrence from transfer; C29 protects the candidate path from unauthenticated provenance and must beat its mask-removal control |
| D2K / memory retrieval | old user-item-context information becomes retrievable memory | C29 stores no candidate score or ternary outcome; memory contributes only an item-membership admissibility bit |

Reduction verdict: `distinct-with-uncertainty`.  If removing authentication does
not hurt true-minus-wrong utility, C29 reduces empirically to an ordinary
paired-history cross-encoder and closes.  If replacing the event mask by one
external scalar gate is equivalent, the architecture novelty claim also
closes.  Soft-token, user-ID, or pretrained-LM capacity cannot pay the rent.

Primary sources include:

- https://arxiv.org/abs/2402.10548 (CoPS)
- https://arxiv.org/abs/2308.15968 (Denoising Attention)
- https://openreview.net/forum?id=ia9vDh0Ltn (IDIOMoE)
- https://openreview.net/forum?id=TU8e2wRY89 (CLLM4Rec)
- https://openreview.net/forum?id=4AumMJKets (D2K)
