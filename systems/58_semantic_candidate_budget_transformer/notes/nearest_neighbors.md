# C58 nearest-neighbor audit

Status: pre-outcome; global novelty **unestablished**.

| Neighbor | Shared mechanism | Binding boundary |
|---|---|---|
| ColBERT ([paper](https://arxiv.org/abs/2004.12832)) | Fixed fine-grained MaxSim token interaction | C58 composes two MaxSim relations through a history-event candidate budget. `raw_query` removes history and exposes a pure late-interaction reduction. |
| Slot Attention ([paper](https://arxiv.org/abs/2006.15055)) | Inputs normalize across exchangeable slots | `slot_budget_no_null` is the direct candidate-axis reduction. Candidate-axis softmax is not a novelty claim. |
| DIN/ZAM/TEM | Query/target-aware history attention, including a zero sink | `history_softmax` retains the same triadic logits and NULL but changes only the normalization axis. |
| C47/C52 fixed KRR/token softmax | Frozen LM semantic history reads and query-token allocation | `pooled_history` and `history_softmax` test whether candidate-budget normalization adds anything beyond fixed semantic reads. |
| C57 | Candidate+NULL event allocation with learned multi-head values/readout | C58 fixes the semantic compatibility and score direction. A failure means removing the gauge does not rescue the family; a positive result still must beat C57's nearest reductions. |

C58 is intentionally a formulation, not a global architecture-novelty claim.
Its purpose is to decide whether C57's unstable learned direction was hiding a
valid fixed semantic evidence law.  No new claim survives if the nearest
controls match it.
