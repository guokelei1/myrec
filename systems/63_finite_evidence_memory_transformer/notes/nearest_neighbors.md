# C63 nearest-neighbor and reduction audit

Status: pre-outcome.  C63 has no novelty authorization unless it pays rent over
every binding control.

| Neighbor | Direct overlap | Binding distinction/control |
|---|---|---|
| Slot Attention | inputs compete across slots, then slot values are normalized | `slot_competition_memory` implements the one-step normalization; tying it rejects C63 |
| Inverted-attention Transformers | changing the normalization axis can recover slot-like behavior | same binding control; a normalization-axis-only win is insufficient |
| MESH / optimal-transport Slot Attention | balanced transport and entropy/tie-breaking form distinct slots | `balanced_transport_memory` is mandatory; C63 must beat it without post-hoc temperature tuning |
| Stick-Breaking Attention | sequential sigmoid breaks replace softmax attention over source tokens | C63 applies a finite stick per history event across memory destinations with a NULL remainder, not recency allocation over context tokens |
| Ordered Memory | stick-breaking controls recurrent write/erase | C63 is a two-phase Transformer set memory and must beat its simpler slot controls; it makes no claim to invent stick-breaking memory |
| Adaptive Slot Attention | NULL/masking can vary active slot count | C63 fixes slot capacity and lets events abstain; no learned slot-count router is claimed |
| C14/C44/C57 | subprobability/NULL or budgeted attention | those operate on candidate assignment/readout and paid no rent; C63 must show that conservation changes the history representation itself |
| C62 | history-only latent write followed by immutable read | `standard_slot_memory` and `single_pooled_memory` reproduce the failed write laws exactly |

Primary references:

- Slot Attention: https://arxiv.org/abs/2006.15055
- MESH: https://proceedings.mlr.press/v202/zhang23ba.html
- Stick-Breaking Attention: https://arxiv.org/abs/2410.17980
- Ordered Memory: https://openreview.net/forum?id=BJGuY4Sl8r
- Inverted-Attention Transformers: https://openreview.net/forum?id=m9s6rnYWqm
