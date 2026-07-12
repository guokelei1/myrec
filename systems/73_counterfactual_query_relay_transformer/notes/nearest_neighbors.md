# C73 nearest-neighbor and reduction audit

| Neighbor | Shared idea | Binding difference / reduction |
|---|---|---|
| DIN / query-aware PPS attention | candidate or current query selects history | direct history values reach the score; C73 forbids that edge and uses query-trajectory attention difference |
| PIANO QDIR (2026) | current-query cross-attention refines historical interest | QDIR refines a profile and list node; C73 exposes only factual-minus-NULL query-token relay to each candidate |
| Query-context-aware LTR for sequential recommendation (2025) | fuses query context inside attention | historical context is fused into sequence attention; C73 binds a two-stage path and a shared internal NULL operator difference |
| Multimodal Bottleneck Transformer | cross-stream information passes through bottleneck tokens | bottleneck is generic compression; C73's mediator is the actual current query and is paired with a structural NULL trajectory |
| RETR Pathway Attention | suppresses trivial behavior tokens by learned routing | binary behavior routing has direct sequence-to-prediction flow; C73 has no route and tests a counterfactual mediated write |
| Multi-Token Attention | attention weights depend on multiple neighboring tokens | richer local attention kernel, but no history/query provenance path or factual/NULL subtraction |
| Counterfactual Attentiveness Test | wrong-part replacement tests input reliance | evaluation/training augmentation, not an internal ranking operator; C73 adopts wrong-history replacement only as a falsifier |
| C31/C32 | history transports a shared query representation | pooled single-vector cosine transport; `pooled_query_relay` is the exact C73 reduction |
| C45 | factual-minus-NULL event innovation before readout | subtracts event-transition states; does not mediate through current query tokens |
| C54 | factual-minus-NULL history carrier in candidate competition | raw carrier reaches candidates; C73 permits only query-mediated attention difference |
| C65/C66 | factual-minus-NULL internal candidate state | subtraction occurs after the joint candidate trajectory; `late_state_difference` is the locked nearest control |

Primary sources reviewed before lock:

- PIANO: https://arxiv.org/abs/2606.16641
- Query-context-aware LTR: https://arxiv.org/abs/2507.03789
- Attention Bottlenecks for Multimodal Fusion: https://arxiv.org/abs/2107.00135
- RETR: https://arxiv.org/abs/2206.06804
- Multi-Token Attention: https://openreview.net/forum?id=Z3L35tQTEg
- Counterfactual Attentiveness Test: https://aclanthology.org/2024.findings-emnlp.205/

The audit supports only provisional non-isomorphism.  C73 makes no global
novelty claim before empirical rent and a broader paper-stage literature
review.
