# C35 nearest neighbours and reduction boundary

| Neighbour | Covered mechanism | C35 boundary / control |
|---|---|---|
| [ReLA](https://arxiv.org/abs/2104.07012) | ReLU attention and attend-to-nothing behavior | ReLU is not new; C35 tests candidate-axis event-common subtraction before ReLU |
| [Slot Attention](https://arxiv.org/abs/2006.15055) | competitive normalization across exchangeable slots | `candidate_axis_softmax` is the mandatory generic competition control |
| [DIN](https://arxiv.org/abs/1706.06978), [candidate-aware user modeling](https://arxiv.org/abs/2204.04726) | target/candidate-conditioned history aggregation | absolute/candidate-axis controls test whether C35 exceeds ordinary candidate awareness |
| [ROWAN](https://openreview.net/forum?id=lDDX9QzEnI) | row-centred finite-support attention substitution | C35 does not claim centring generally; it centres each history event over ranking candidates and ties the write to query-tangent evidence |
| C34 | absolute tangent half-space with fixed null | `absolute_tangent_cone` is an exact in-run reduction |

Removing the candidate mean gives C34.  Replacing rectified surplus by
candidate-axis softmax gives generic competitive attention.  Making the write
candidate-shared gives global transport.  If any matched reduction equals or
beats the primary, C35 has paid no mechanism rent.  No global novelty claim is
registered before this gate and broader review.
