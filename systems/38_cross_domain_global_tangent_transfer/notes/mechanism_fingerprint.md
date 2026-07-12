# C38 mechanism fingerprint

| Field | Frozen value |
|---|---|
| Scientific role | Cross-domain falsifier; no novelty claim |
| Operator | Candidate-shared query-attended tangent history transport |
| Intervention | Transformer embedding/residual state before candidate dot product |
| Trainable state | Shared rank-16 down/up projection only |
| Training signal | Fixed-candidate listwise ranking plus direction loss on train-internal fit |
| Inference input | Query text, strict-past history item text, fixed candidate text |
| Exact fallback | Empty/masked history gives zero correction and base ranking |
| Equal-capacity reductions | Unprojected query-attended transport; mean-history tangent transport |
| Causal corruption | History-length-bin-matched wrong-user history; no target category |
| Nearest internal predecessor | C32/C33 global tangent mode and C37 global control |

C38 is intentionally reducible to the earlier global tangent operator.  Its
new evidence is the independent dataset and encoder regime, not a renamed
operator.  It cannot satisfy the paper's architecture-novelty requirement by
itself.
