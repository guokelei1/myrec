# C74 mechanism fingerprint

| Field | Frozen fingerprint |
|---|---|
| primitive | semantic-conservative two-hop query relay |
| operator | nested history-to-query and candidate-to-query attention with raw-LM values and factual-minus-NULL semantic energy |
| learned coordinates | two low-rank routing maps plus history-position routing bias |
| immutable coordinate | history value, query carrier, candidate value, and energy readout all use the same LM state |
| allowed path | `history -> query-token raw carrier -> candidate semantic energy` |
| forbidden paths | direct history-to-candidate value; independent learned V/O/FFN/head |
| intervention site | attention logits and query residual inside the Transformer ranking core |
| inference inputs | common query/history/candidate/order/mask interface |
| exact identities | no-history/query-absent -> base; repeat-present -> item-only |
| reductions | coupled learned values, pre-relay pooling, factual-only energy |

C74 is not C40/C42 because it deliberately rejects a closed learned metric
loop and instead learns only routing.  It is not C41 because C41 transports one
pooled profile; C74 keeps query-token-indexed carriers through a second
candidate-conditioned attention.  It is not C73 because C73 learns MHA values,
FFN, LayerNorm, and a scalar output head.
