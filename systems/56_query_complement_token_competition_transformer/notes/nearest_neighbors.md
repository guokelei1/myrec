# C56 nearest-neighbor and reduction audit

Status: pre-outcome.  Global novelty is **unestablished**.

| Neighbor | Shared mechanism | Binding distinction / reduction |
|---|---|---|
| TEM ([paper](https://arxiv.org/abs/2005.08936)) and RTM ([paper](https://arxiv.org/abs/2004.09424)) | Transformer interaction among query, user history, and item/review units | They use ordinary factual contextualization. `raw_candidate` and `unprojected_token` test whether C56 reduces to stronger generic contextualization. |
| DIN/target attention | Candidate queries user-history values | C56 rejects the query-explained token component before transport and permits the primary score only through factual/null token-state difference plus candidate-relative V transport. `unprojected_token` is the direct degeneration. |
| ColBERT ([paper](https://arxiv.org/abs/2004.12832)) | Fine-grained query/document token interaction | ColBERT has no user-history value stream or candidate-set interaction. `raw_candidate` tests whether improved candidate token matching alone explains an effect. |
| Set Transformer ([paper](https://openreview.net/forum?id=Hkgnii09Ym)) / PRM ([paper](https://arxiv.org/abs/1904.06813)) | Permutation-equivariant candidate interactions | Generic set/list attention may transport raw candidate values. C56's primary V stream contains only a history-created token delta; `raw_candidate` and `edge_ablation` expose the reduction. |
| Orthogonal Alignment analysis ([paper](https://arxiv.org/abs/2510.09435)) | Cross-attention can learn information orthogonal to its query representation | That work reports emergent orthogonal alignment in cross-domain sequential recommendation and explicitly does not impose an orthogonality constraint. Therefore query-complement rejection alone is not a novelty claim; C56 must beat `unprojected_token`, and any later claim is limited to the PPS-specific token-write plus candidate-relative transport law. |
| C26 query-pivot token bridge | Query/candidate/history token interaction and a shared token encoder | C26 pooled the bridge and wrote an independent bounded scalar after representation formation. C56 injects history before token pooling and binds a candidate-relative readout. `edge_ablation` reduces the latter difference. |
| C32/C33 tangent query transport | Removes a query-parallel history component before scoring | C32/33 operate on one pooled request-level query/profile. C56 residualizes every candidate/history token and forms a candidate-specific factual/null token state. `pooled_complement` tests whether granularity pays rent. |
| C54/C55 history-carrier competition | Candidate Q/K with history-only V and a strong exact anchor | C54/55 form the carrier after pooled LM states. `pooled_complement` is the direct reduction; the primary must beat it under identical score units and loss. |

The literature check found a particularly close 2025 orthogonal-alignment
analysis, so C56 does **not** claim that orthogonality, cross-attention,
fine-grained matching, or set reranking is new.  The only potentially distinct
fingerprint is their ordered composition as one constrained information-flow
law: query-explained token rejection, factual/null token-state formation, and
leave-one-out history-only V competition.  If any direct reduction matches the
primary, the architecture-rent claim fails even if the absolute metric rises.
