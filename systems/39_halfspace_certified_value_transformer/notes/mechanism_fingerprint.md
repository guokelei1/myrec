# C39 mechanism fingerprint

| Field | Frozen design value |
|---|---|
| Primitive | Candidate-relative, eventwise projection of Transformer values onto a candidate-readout halfspace |
| Intervention | `history -> candidate-local query token` cross-attention `V/W_O` path before event aggregation |
| Common backbone | Query-attended unprojected multi-head history value write |
| State | Frozen LM embeddings plus a trainable multi-head interaction block for the minimal gate |
| Trainable path | Shared `Q/K/V/W_O` and shared Transformer FFN; no pair MLP or scalar candidate head |
| Exact rejection | Nonpositive candidate-relative support gives a zero edge; empty history/query gives zero request correction |
| Direction law | Every admitted pre-aggregation value has nonnegative local candidate-readout contribution |
| Degenerations | Raw event values; post-pool projection; score-ray-only value; global-only write |
| Strong predecessor control | C38 query-attended unprojected transport |
| Scientific role | New value-interface hypothesis; no global novelty or proposed-system claim before gates |

The fingerprint is not a relabeling of tangent transport: C39 does not remove a
query-parallel component or transport on a sphere. It changes each admitted
event value before summation by solving a candidate-readout feasibility
problem. It is also not C35 relative tangent surplus: relative support only
determines whether an edge exists; the load-bearing hypothesis is the value
direction certificate that C35 lacked.
