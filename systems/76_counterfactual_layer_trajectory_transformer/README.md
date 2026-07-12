# C76 — Counterfactual Layer-Trajectory Transformer

C76 is the first proposed-system candidate authorized by the positive Amazon
full-token history observability result.  It preserves ordinary bidirectional
query/history/candidate WordPiece contextualization, but the personalized
ranking path may read only the layerwise difference between a factual forward
and a same-token history-cut forward.  A protected query-candidate LM supplies
the no-history base.

The design is frozen in `notes/`; implementation and GPU gates follow the
staging rules in `notes/design_gate_protocol.md`.  Checkpoints, scores, and raw
runs belong under ignored `checkpoints/`, `runs/`, or `artifacts/` paths.
