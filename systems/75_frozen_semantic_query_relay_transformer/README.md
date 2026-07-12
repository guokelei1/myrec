# C75 — Frozen Semantic Query-Relay Transformer

C75 makes the pretrained LM semantic carrier an architectural invariant.  The
shared LM stays frozen and in evaluation mode; only the two-hop
history-to-query / candidate-to-query routing maps and chronology are trained.

This is a new successor to C74, not a repair or label release for C74.  It has
its own source tree, G0, execution lock, seeds, fixed-anchor training gate, and
matched controls.  Dev, test, qrels, and fresh roles remain closed.
