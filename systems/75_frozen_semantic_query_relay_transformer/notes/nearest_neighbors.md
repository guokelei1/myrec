# C75 nearest-neighbor audit

| Neighbor | Shared property | Binding difference / control |
|---|---|---|
| frozen-encoder retrieval/reranking | pretrained LM parameters fixed | ordinary cosine/cross-encoder readout; C75 adds two trainable personalized attention stages inside ranking |
| DIN/ZAM/TEM | query/history personalization | direct/pooled contextual values; C75 preserves token-indexed query carriers and internal factual/NULL energy |
| C31/C32 | frozen BGE query transport | one pooled vector and cosine readout; `pooled_semantic_relay` binds the reduction |
| C41 | learned routing with raw semantic values | one pooled history profile; no candidate-to-query token relay |
| C74 | same semantic-conservative relay | C74 adapts LM token coordinates; C75 makes their invariance load-bearing and verifiable |
| frozen-backbone adapters / linear probing | small trainable module over frozen representation | generic adapter/head; C75 has no learned value/head and modifies attention routing only |

The provisional claim is the combination of immutable LM semantic carriers and
counterfactual two-hop routing.  Global novelty is not claimed before utility
and a broader literature review.
