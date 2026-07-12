# C80 mechanism fingerprint

`frozen pretrained WordPiece anchors -> fixed top-budget shared-Q C-H triangle
admission -> no event-index positions -> bidirectional adaptive BGE interaction
-> same-LM H-edge cut -> protected frozen BGE base + centered bounded write`.

Binding invariants:

- anchor/base parameters have zero gradient and unchanged hashes;
- unsupported tokens have no personalized path;
- C-H and H-C are both present for admitted tokens;
- complete history-event permutation preserves scores;
- candidate storage permutation only permutes scores;
- no history/query returns base and recurrence retains item-only;
- removing query authentication, set symmetry, or token admission realizes a
  named equal-capacity control.
