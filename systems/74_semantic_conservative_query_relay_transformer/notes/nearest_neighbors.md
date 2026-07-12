# C74 nearest-neighbor and reduction audit

| Neighbor | Shared idea | Binding difference / reduction |
|---|---|---|
| DIN / ZAM / query-aware PPS | query/candidate-aware history attention | one pooled user vector or direct history value; `pooled_semantic_relay` is the closest reduction |
| TEM | Transformer over query and purchase history | ordinary contextual values can be rewritten freely; C74 binds a raw semantic carrier and a factual/NULL energy difference |
| PIANO QDIR | current query refines historical interest | profile/list-node refinement; no candidate-specific second relay or internal NULL energy |
| hierarchical query-aware search models | two-level history modeling | hierarchical RNN/profile, not a token-level path-constrained Transformer operator |
| Attention Bottleneck Transformer | information passes through mediator tokens | generic learned bottleneck values; C74's mediator is current query and values are semantic-conservative |
| C40/C42 | learned metric couples selection/value/readout | exact opposite value rule; `coupled_value_relay` binds this reduction |
| C41 | learn routing while carrying raw LM values | one pooled transport/readout; C74 preserves token-indexed query carriers into candidate attention |
| C73 | factual/NULL query-token relay | learned MHA V/O, FFN, normalization, and scalar head; C74 removes those coordinate gauges |
| Bank of Values (2026 submission) | preserves context-independent token value information | token lookup values for general LM layers, not evidence-path routing or personalized ranking |
| Neural Data Router | attention as learned compositional routing | generic algorithmic routing with copy gates; no shared LM-semantic evidence carrier or PPS fallback contract |

Primary sources reviewed before lock:

- TEM: https://arxiv.org/abs/2005.08936
- PIANO: https://arxiv.org/abs/2606.16641
- hierarchical query-aware personalization: https://arxiv.org/abs/1908.07600
- Attention Bottlenecks: https://arxiv.org/abs/2107.00135
- Neural Data Router: https://openreview.net/forum?id=KBQP4A_J1K
- Bank of Values: https://openreview.net/forum?id=YoQ0VK3JnP

This is a provisional non-isomorphism audit, not a global novelty claim.
