# C78 — Query-Filtered Set-Interaction Transformer

C78 is post-C76 architecture update 2/3.  It treats prior history events as an
exchangeable set while preserving within-item WordPiece order and direct
bidirectional query/candidate/history token interaction.  A frozen query-side
candidate-token admission boundary prevents C76's candidate-only shortcut.
