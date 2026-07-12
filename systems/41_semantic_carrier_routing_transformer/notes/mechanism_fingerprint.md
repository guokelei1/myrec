# C41 mechanism fingerprint

| Field | Frozen value |
|---|---|
| Role | Architecture-boundary probe; not yet a novelty claim |
| Trainable path | Query/history routing metrics only |
| Immutable carrier | Raw normalized LM history values, query transport, and candidate readout |
| Primary | Four routing heads with shared query/history map per head |
| Exact fallbacks | no history, absent query, repeat-present -> zero correction |
| Closest internal reduction | C40 `selection_only`, numerically identical at matched state |
| Strong external control | C38 query-attended unprojected transport |
| Forbidden claim | Identity V/O, QKV simplification, or routing/content separation alone is novel |

The collision-resistant property is not a new attention family. It is the
absence of any learned history-content edge after routing. A router perturbation
can change only simplex weights; the profile remains a convex combination of
raw LM event states.
