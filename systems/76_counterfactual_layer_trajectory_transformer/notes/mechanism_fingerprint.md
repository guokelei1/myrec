# C76 mechanism fingerprint

## Load-bearing fingerprint

`same raw Q/H/C WordPieces and positions -> shared adaptive LM under factual
and history-cut attention graphs -> carrier-scaled hidden difference at every
depth -> layer/segment trajectory Transformer -> protected Q-C base plus
candidate-centred bounded write`.

The intervention changes only attention edges.  It does not delete history
tokens, shift candidate positions, substitute a learned NULL vector, or run a
second independently parameterized scorer.

## Required functional contracts

1. `H absent => F^l == N^l` and the final score equals the protected base.
2. Isolating H from Q/C on a factual sequence gives the same Q/C states as the
   corresponding null sequence within the registered tolerance.
3. Gradients from the personalized loss reach the shared adaptive LM and
   trajectory Transformer, but never the protected base.
4. Earlier layer tokens are load-bearing: replacing them with zero or
   retaining only the final state changes the primary output.
5. Candidate storage permutation only permutes scores.
6. There is no raw factual-state or raw query-candidate feature bypass into the
   personalized head.

## Degenerations

- final logit only: C04/classifier-free-guidance family;
- final hidden state only: C65-C66;
- factual layer trajectory: ladder/side network with a generic factual path;
- one learned attention-map difference: Differential Transformer family;
- pooled history before the LM: C25/C31-C43 and DIN/TEM-like reductions;
- query-only two-hop relay: C73-C75.

Deleting the depth ledger, structural edge cut, raw-token factual graph, or
protected base changes the hypothesis.  Global novelty remains provisional
until the matched controls pay empirical rent.
