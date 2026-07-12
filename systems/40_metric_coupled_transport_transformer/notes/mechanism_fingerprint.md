# C40 mechanism fingerprint

| Field | Frozen identity |
|---|---|
| Primitive | One residual semantic metric closes selection, value, transport, and candidate readout per head |
| Transformer locus | Query-to-history attention plus transported-query residual and tied candidate readout |
| Base | Frozen LM query/candidate score; cached states are exact LM execution |
| Trainable map | `T_r(x)=normalize(x+U_r V_r x)` |
| Primary invariant | Head `r` uses `T_r` at all four evidence-loop locations |
| Exact fallbacks | no history, absent query, and repeat-present correction equal zero |
| Main falsifier | same-parameter cyclic reassignment of selection map to another head's value/readout map |
| Strong predecessor | C38 query-attended unprojected transport |
| Forbidden interpretations | generic QKV sharing, metric learning alone, adapter alone, or multi-head attention alone |

The fingerprint differs from C39: C39 projected already-rewritten event values
onto score halfspaces; C40 removes the independent rewriting maps. It differs
from C32--C38 tangent work because it removes no query-parallel component and
imposes no post-attention geometry. Its claim lives in parameter coupling and
information-flow identity, not in a score sign constraint.
