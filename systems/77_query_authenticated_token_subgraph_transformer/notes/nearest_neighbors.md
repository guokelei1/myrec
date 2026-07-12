# C77 nearest-neighbor audit

| Neighbor | Closest mechanism | Binding distinction/control |
|---|---|---|
| [HOMA](https://arxiv.org/abs/2603.11133) | explicit trainable pairwise + triadic attention | HOMA adds learned triadic capacity. C77 uses a frozen semantic triangle only to prohibit unsupported ranking edges; `ungated_full` and a trainable-capacity supermodel remain controls. |
| [TripleNet](https://aclanthology.org/K19-1069/) | concurrent symmetric context/query/response triple attention | TripleNet learns all triple interactions from task labels; C77 freezes token eligibility before the interaction Transformer. |
| [QVI](https://arxiv.org/abs/2010.03766) | query-aware attention values | QVI changes values through learned query-value gates; C77 changes the cross-role graph and leaves ineligible values unreachable. |
| [BiFormer](https://arxiv.org/abs/2303.08810) | content-aware coarse-to-fine token routing | BiFormer learns efficiency routing; C77's frozen three-role provenance intersection is a ranking-identifiability constraint, not a compute-only router. |
| C03 | triadic partial transport over pooled role states | C77 operates on raw WordPieces and imposes a fixed edge graph before multi-layer Transformer contextualization. |
| C26 | query-pivot token bridge followed by an additive residual | C26 aggregates a learned late-interaction bridge after token encoding; C77 changes every interaction layer's admissible graph and excludes unsupported tokens entirely. |
| C39-C42 | frozen/raw semantic carriers with learned routing or metric coupling | Those pool history events into a transported value. C77 uses anchors only for admission; the ranking core remains a bidirectional token Transformer. |
| C73-C75 | two-hop history-to-query-to-candidate relay | C77 retains direct authenticated C-H and H-C edges, required by the frozen edge attribution. |

Novelty status: `role-constrained frozen subgraph; global novelty uncertain`.
Query filtering, sparse attention, triple attention, and frozen encoders are
known.  Only statistically load-bearing superiority over the four graph
reductions could support an architecture contribution.
