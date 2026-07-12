# C78 mechanism fingerprint

`frozen query support on candidate WordPieces -> admitted Q/C/H token graph ->
history-event shared within-item positions with no event index -> multi-layer
bidirectional Transformer -> paired H-cut correction -> protected base`.

Contracts:

- complete-event permutation changes no score;
- within-event token permutation may change the score;
- unsupported candidate-token Jacobian is zero;
- direct C-H and H-C paths are active;
- no history/query and repeat behavior are exact;
- the protected base receives no personalization gradient.

Removing set symmetry yields C77's winning reduction.  Removing query filtering
yields an ungated Set Transformer.  Replacing it with pairwise/triadic admission
yields the two nearest graph controls.
