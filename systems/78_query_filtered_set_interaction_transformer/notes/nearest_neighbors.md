# C78 nearest-neighbor audit

| Neighbor | Closest mechanism | Difference/control |
|---|---|---|
| [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html) | permutation-invariant attention over sets | C78 preserves ordered WordPieces inside each event and performs role-constrained Q/C/H ranking with a protected no-history base. Set symmetry itself is not claimed novel. |
| [Deep Sets](https://arxiv.org/abs/1703.06114) | invariant set functions | `pairwise_set`/future pooled controls test whether token-level cross-role interaction is necessary beyond invariant aggregation. |
| [RTM](https://arxiv.org/abs/2004.09424) | joint query/user-review/item-review Transformer | RTM uses sequence positions and direct factual scoring; C78 enforces event-exchangeability and a history-only residual path. |
| [BiFormer](https://arxiv.org/abs/2303.08810) | query-adaptive token routing | C78's frozen admission is an identifiability boundary; `ungated_set` and positional controls isolate it. |
| C26/C77 | query-pivot/filter token interaction | C78's only new hypothesis is event-set group equivariance; C77 positional behavior is the direct degeneration. |
| C57-C59 | candidate-axis semantic token budget | Those allocate pooled event evidence across candidates and lost to the strong base. C78 preserves candidate-wise raw-token cross-encoding and has no independent standardized history score. |

Novelty status: `known symmetry plus task-specific information-flow boundary`.
No CCF-A-level novelty claim is authorized by design alone.
