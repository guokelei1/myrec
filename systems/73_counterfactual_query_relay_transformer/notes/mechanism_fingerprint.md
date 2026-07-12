# C73 mechanism fingerprint

| Field | Frozen fingerprint |
|---|---|
| primitive | counterfactual query-relay attention |
| mathematical operator | `Attn(C,Q_H,Q_H) - Attn(C,Q_0,Q_0)` inside the candidate residual stream |
| intervention site | token-level query-to-candidate attention block after a history-to-query block |
| allowed path | `history -> query tokens -> candidate tokens -> logit` |
| forbidden path | direct `history -> candidate` key/value edge |
| state | shared-LM WordPiece states plus candidate-local relay residual |
| training signal | one listwise ranking objective; corruptions are falsifiers, not inference inputs |
| inference input | common query/history/candidate/mask record only |
| exact identities | no-history/query-absent -> strong base; repeat-present -> item-only |
| nearest reductions | late state difference, pooled query transport, factual-only relay |
| claim boundary | architectural rent exists only if all three reductions lose under the locked gate |

The fingerprint is not equivalent to C45: C45 subtracts local event-transition
states before scoring.  It is not C65/66: they subtract final joint candidate
states.  It is not C31/32: they move one pooled query vector and use cosine
readout.  It is not C54: raw history carriers enter candidate competition.
