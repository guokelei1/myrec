# C80 nearest-neighbor audit

| Neighbor | Closest mechanism | C80 boundary/control |
|---|---|---|
| [HOMA](https://arxiv.org/abs/2603.11133) | explicit triadic attention | HOMA adds learned triadic capacity; C80 freezes a role-specific semantic admission graph to prevent label-created edges. |
| [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html) | permutation-invariant set attention | C80 preserves within-item WordPiece order and cross-role candidate ranking; `triadic_positional` isolates symmetry rent. |
| [TripleNet](https://aclanthology.org/K19-1069/) | symmetric context/query/response triple attention | TripleNet learns triple interactions end-to-end; C80 freezes edge eligibility and protects a no-history base. |
| [RTM](https://arxiv.org/abs/2004.09424) | joint query/user-review/item-review Transformer | RTM has factual-state scoring and event order; `ungated_full` is the closest functional control. |
| [BiFormer](https://arxiv.org/abs/2303.08810) | content-aware token routing | C80 routing is frozen provenance, not learned compute allocation. |
| C26/C77/C78 | query-pivot, triadic admission, event-set symmetry | C80 is exactly the preregistered C78 winning control, tested under a new fresh real lock; none of the component names alone is novel. |
| C57-C59 | candidate-budget semantic token evidence | C80 does not emit an independently standardized history score or candidate-axis budget. |

Novelty status: `uncertain composition; empirical rent required`.
