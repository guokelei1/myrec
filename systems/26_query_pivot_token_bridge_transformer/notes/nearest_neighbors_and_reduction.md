# C26 nearest neighbours and reduction audit

| Neighbour | Overlap | Required distinction/control |
|---|---|---|
| ColBERT / ColBERTv2 | contextual token vectors and query-token late interaction | candidate-only late interaction is an explicit control; C26 history value must add value beyond it |
| LITE learnable late interaction | learned token-level scoring can approximate general interactions | C26 claims a restricted shared-pivot law, not superior expressivity |
| SFI fine-grained history interaction | candidate interacts with selected history documents/tokens | C26 pivots both sides through the current query token and uses no learned history selector; generic token triadic is the control |
| RTM/TEM | Transformer-based fine-grained/dynamic PPS representation | C26 operates at title WordPiece bridge level and must beat pooled history within the same code path |
| Tri-Attention / HyperAttention | explicit three-way or n-way dependence | generic triadic dependence is prior art and a matched control |
| C03 cycle-intersection transport | only agreement among query/history/candidate may write | C26 uses direct token-level soft alignment without transport, dustbins or cycle products; it is a cheaper information-granularity test |

Primary sources checked before implementation:

- https://arxiv.org/abs/2004.12832
- https://arxiv.org/abs/2112.01488
- https://arxiv.org/abs/2406.17968
- https://arxiv.org/abs/2110.06459
- https://arxiv.org/abs/2004.09424
- https://arxiv.org/abs/2211.02899

Verdict: token late interaction and fine-grained personalized matching are
known.  C26 is eligible only as a matched-control signal gate for the
same-query-token bridge restriction; novelty remains uncertain.
