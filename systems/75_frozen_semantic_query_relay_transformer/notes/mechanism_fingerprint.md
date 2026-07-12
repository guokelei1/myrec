# C75 mechanism fingerprint

| Field | Frozen fingerprint |
|---|---|
| primitive | immutable-LM semantic carrier with trainable counterfactual query relay |
| pretrained LM lifecycle | frozen parameters, forced eval, bit-exact state before/after training |
| trainable state | two low-rank attention-routing maps and chronology only |
| value/readout coordinate | the same frozen LM token state at all carrier sites |
| allowed path | `LM history -> query-token relay -> candidate semantic energy` |
| forbidden paths | LM adaptation, direct history value to candidate, learned V/O/FFN/head |
| objective | one listwise ranking loss |
| inference input | common query/history/candidate/order/masks |
| exact identities | no-history/query-absent -> D2p; repeat -> item-only |
| matched reductions | coupled values, pooled carrier, factual-only energy |

C75 differs from C41 by retaining query WordPieces through a second candidate
attention rather than forming one pooled profile.  It differs from C74 by
making the pretrained semantic coordinate immutable rather than jointly
adapted.
