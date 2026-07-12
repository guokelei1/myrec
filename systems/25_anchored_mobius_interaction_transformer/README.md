# C25 — Anchored Möbius Interaction Transformer

C25 tests whether a Transformer can learn cross-item personalization after all
query-only, candidate-only, history-only and pairwise shortcut terms are
removed before the history write.  It is a train-only architecture gate, not a
dataset-specific recipe and not yet a paper-level novelty claim.

The residual path receives only anchored third-order Möbius event tokens.  It
cannot observe D2p, recurrence mass or raw candidate features through another
path.  Registered D2p and item-only remain protected score anchors for the
minimal probe.  No-history, query-absent and repeat-present behavior are exact
structural fallbacks.

All generated features, checkpoints and raw reports live under ignored
`artifacts/c25_*` and `models/c25_*` paths.  Dev and test are unauthorized.
