# C77 mechanism fingerprint

`frozen pretrained WordPiece anchors -> positive Q-C-H triangle support ->
fixed request-specific eligible token subgraph -> trainable bidirectional
interaction Transformer -> paired H-edge-cut logit -> protected base`.

Load-bearing contracts:

1. ranking gradients cannot change anchors or token admission;
2. a candidate/history token with zero shared-query triangle has zero Jacobian
   to the personalized score;
3. admitted C-H and H-C edges are both present at every interaction layer;
4. query absence, history absence, and no positive triangle give exact base;
5. candidate permutation only permutes scores;
6. removing the query factor, candidate filter, history filter, or subgraph
   mask produces a registered control, not the primary.

The primitive is not generic triple attention: it is a frozen provenance
restriction on which raw-token edges a ranking-trained Transformer may create.
Global novelty remains uncertain until both literature and matched controls
survive.
