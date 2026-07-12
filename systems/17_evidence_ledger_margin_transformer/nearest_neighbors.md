# C17 nearest-neighbor audit

| Neighbour | Primary source | Collision with C17 |
|---|---|---|
| Edge Transformer | [Bergen, O'Donnell, and Bahdanau, NeurIPS 2021](https://papers.nips.cc/paper/2021/hash/0a4dc6dae338c9cb08947c07581f77a2-Abstract.html) | associates vector states with every pair of nodes and updates them with triangular attention; this covers a learned persistent candidate--event ledger |
| Attention rollout / flow | [Abnar and Zuidema, ACL 2020](https://aclanthology.org/2020.acl-main.385/) | propagates token information flow across Transformer layers; an exactly tied provenance ledger is a stricter differentiable attribution variant, not a new score function |
| AttCAT | [Qiang et al., NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/20e45668fefa793bd9f2edf19be12c4b-Abstract-Conference.html) | combines encoded values, gradients, attention and residual information for Transformer attribution; using such signals to rescale values is an attribution gate |
| C15 candidate-conditioned value | `../15_candidate_conditioned_value_write_transformer/preimplementation_decision.md` | scalar ledger readout is a gate; vector-valued joint readout is generic edge-conditioned message passing |
| C06 conservative flow | `../06_conservative_wedge_flow_transformer/notes/mechanism_fingerprint.md` | antisymmetric candidate margins followed by divergence are the same score geometry, even if their entries retain event provenance |
| C03/C08 higher-order state | `../03_triadic_transport_transformer/notes/final_report.md`; `../08_reversible_memory_transformer/README.md` | retaining cycles or triangular interactions introduces already-tested transport/closed-loop higher-order state rather than a ledger-specific law |

The literature search was performed before any C17 model or outcome.  The
collision is structural, not based on paper names: changing the pair-state
index from arbitrary nodes to candidate/history nodes does not create a new
operator.
