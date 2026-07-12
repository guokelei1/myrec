# C62 nearest-neighbor and reduction audit

Status: pre-outcome.  Novelty is provisional until the matched controls pay
positive utility rent.

| Neighbor | Overlap | Binding distinction/test |
|---|---|---|
| Perceiver / Perceiver IO | learned latent bottleneck cross-attends inputs | C62 does not claim latent tokens alone; it freezes a history-only user state before a separate query-candidate read and tests that graph against direct attention |
| Set Transformer PMA | seed vectors pool a set | C62 preserves multiple slots and requires candidate-conditioned reads; `single_pooled_slot` is the exact reduction control |
| Slot Attention | competitive learned slots | C62 does not claim object-centric slot discovery; slot multiplicity must be load-bearing on ranking and survive the pooled reduction |
| Transformer memory tokens / recurrent memory | extra tokens carry state | ordinary joint memory tokens permit current query/candidates to rewrite state; `query_conditioned_writer` tests that nearest graph |
| DIN/ZAM/TEM/target attention | candidate/query directly selects history | `direct_history_attention` is the same-capacity binding control; tying it rejects the C62 primitive |
| PSMIM and hierarchical personalized search Transformers | hierarchical history/query modeling | C62's narrow claim is the immutable write-then-read attention graph, not hierarchy or auxiliary mutual-information losses |
| MemRerank | query-independent preference memory followed by reranking | MemRerank is an external two-stage memory/reranker framework; C62 is a jointly trained end-to-end Transformer ranker with no offline memory text or prompt/API boundary |
| C08 | reversible history write, query/candidate probe, and undo | C08's load-bearing object is a closed-loop commutator; C62 keeps a forward latent state and tests multi-slot reuse |
| C45/C49/C50 | counterfactual/prequential memory values | those candidates alter event values or fixed KRR/DeltaNet reads; C62 changes the attention graph and learns the latent state jointly from ranking loss |
| C53/C54/C56 | joint/direct history-candidate Transformers | those allow candidate-conditioned history interaction before a stable user state exists; the direct mode is a binding control, not a claimed novelty |

## Expressivity warning

A sufficiently large dense Transformer can approximate the same input-output
function.  C62 therefore makes no universal expressivity claim.  The proposed
innovation is an inductive-bias and state-lifecycle claim: history-only writing
must improve evidence fidelity under matched capacity.  If dense/direct or
query-conditioned controls tie it, the mechanism is reducible for this paper
regardless of representational differences.

## References

- Perceiver: https://arxiv.org/abs/2103.03206
- Set Transformer: https://proceedings.mlr.press/v97/lee19d.html
- Slot Attention: https://arxiv.org/abs/2006.15055
- PSMIM: https://aclanthology.org/2024.ccl-1.76/
- MemRerank: https://arxiv.org/abs/2603.29247
