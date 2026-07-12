# C76 nearest-neighbor and reduction audit

| Neighbor | Closest mechanism | Non-reduction witness / binding control |
|---|---|---|
| [RTM](https://arxiv.org/abs/2004.09424) | full contextual interaction among query, user reviews, and item reviews | RTM scores factual states directly; C76 permits personalization only through the same-LM factual/history-cut depth trajectory. `ordinary_full` is binding. |
| [TEM](https://arxiv.org/abs/2005.08936) | multi-layer query/history Transformer personalization | TEM operates on item-level states and has no candidate-specific edge intervention trajectory. Pooled/query-relay controls are binding. |
| [Differential Transformer](https://openreview.net/forum?id=OvoCm1gGhN) | subtracts two softmax attention maps at every layer | Its maps use separately projected Q/K partitions to cancel generic attention noise. C76 shares all Q/K/V and contrasts one semantic attention graph with its structural H cut, then reads hidden responses across depth. A differential-attention control is required. |
| [DeFormer](https://aclanthology.org/2020.acl-main.411/) | replaces full cross-segment attention by segment-local attention in selected layers | DeFormer uses the cut graph as the model for efficiency; C76 keeps the factual graph and uses the cut only as a paired intervention reference. |
| [Ladder Side-Tuning](https://openreview.net/forum?id=isPnnaTZaP5) | a side Transformer consumes intermediate backbone activations | LST reads factual/base activations and can learn a generic task scorer. C76's side path sees only factual-minus-cut states; `factual_trajectory` is binding. |
| [CD-T](https://arxiv.org/abs/2407.00886) | decomposes Transformer contributions at layer/head/position resolution | CD-T is a post-hoc circuit attribution method. C76 trains the intervention trajectory as the only personalized ranking path and must show ranking rent. |
| C04 | shared LM factual/null final-logit delta | `final_logit_delta` removes depth/segment tokens while keeping all weights and compute. |
| C65-C66 | shared LM factual/null final candidate-state residual | `final_hidden_delta` is the direct degeneration; C76 must depend on earlier layer responses. |
| C45 | factual/NULL state difference at each history event | C45 decomposes event-time recurrent transitions before candidate attention; C76 decomposes Transformer depth after full bidirectional raw-token contextualization. |
| C73-C75 | history modifies query tokens, candidates read the relay | C76 retains direct bidirectional C-H edges, which the frozen edge attribution found load-bearing. |

Novelty status: **distinct architecture restriction, global novelty uncertain**.
Subtraction, layerwise features, side networks, and full contextualization are
known separately.  The claim can survive only if their registered reductions
fail to match the shared-edge-cut trajectory under equal capacity.
