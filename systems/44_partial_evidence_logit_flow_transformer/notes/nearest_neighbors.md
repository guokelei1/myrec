# C44 nearest-neighbor audit

| Neighbor | Covered idea | Mandatory C44 boundary |
|---|---|---|
| [Slot Attention](https://arxiv.org/abs/2006.15055) | inputs normalize across competing slots | candidate-axis competition is not novel; forced-flow control is required |
| [Sinkformers](https://arxiv.org/abs/2110.11773) | stochastic/transport attention normalization | C44 cannot claim transport normalization generally |
| [LOTFormer](https://openreview.net/forum?id=JFAIchgiZr), [ESPFormer](https://openreview.net/forum?id=Uq70mJuUB8) | efficient doubly-stochastic attention | C44 uses one softmax, not a doubly-stochastic or efficiency claim |
| [ReLA](https://aclanthology.org/2021.emnlp-main.523/) | sparse attention and attend-to-nothing behavior | null/abstention alone is not novel |
| DIN and candidate-aware user modeling | target-conditioned history aggregation | global/candidate-local vector-write controls are required |
| C03 | candidate-anchored partial OT with dustbins | C44 must remain one-pass `O(C H)`, without Sinkhorn/cycle/multiplicative plans |
| C35 | candidate-axis softmax followed by tangent vector write | direct logit flow must beat the exact vector-write reduction |
| C43 | request-global metric-coupled history transport | global pooling is the strong predecessor control |

Pre-outcome novelty verdict: `distinct-composition-with-uncertainty`. A
synthetic pass is not enough for a paper claim.
