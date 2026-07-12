# C34 nearest neighbours and reduction boundary

Primary-source audit before C34 outcomes:

| Neighbour | Covered mechanism | C34 residual claim / mandatory control |
|---|---|---|
| [DIN](https://arxiv.org/abs/1706.06978), [NAIS](https://arxiv.org/abs/1809.07053) | target-item-aware history attention | `standard_target_attention` keeps candidate-specific reads and equal capacity; C34 must beat it |
| [TEM](https://arxiv.org/abs/2005.08936), [RTM](https://arxiv.org/abs/2004.09424) | Transformer contextualization of query/history/item | C34 isolates a fixed query-centred positive-cone write, not generic joint encoding |
| [ReLA](https://arxiv.org/abs/2104.07012) | ReLU attention and exact attend-to-nothing heads | ReLU alone is not new; C34's only testable surplus is candidate-specific tangent half-space admission versus forced softmax |
| [BeliefFormer](https://openreview.net/forum?id=Ard2QzPAUK) | orthogonal attention residuals inside Transformer blocks | generic tangent/orthogonal writes are already covered; `global_tangent_transport` is the direct local reduction |
| [Coneheads](https://openreview.net/forum?id=CzAFnfwbGd) | hyperbolic entailment-cone attention | C34 does not claim cone attention generally; its cone is the query-centred candidate/event half-space tied to evidence fidelity |

Algebraically, replacing `ReLU(k)` by forced row-softmax yields the target
control; making the resulting `t_i` candidate-shared yields the global control.
If either matches the primary, the proposed structural law is reducible and
C34 stops.  No globally novel claim is registered before this matched test and
a broader literature review.
