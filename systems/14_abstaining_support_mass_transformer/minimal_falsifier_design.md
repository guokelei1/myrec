# Minimal falsifier if C14 is ever reopened

Status: design only; no execution is authorized.

## S0 exact-equivalence gate

Before training, sample FP64 support logits, allocation logits, masks, values,
multihead projections, candidate permutations, and LayerScale values.  Compare
C14 with the transformed zero-NULL softmax control.

Require pointwise agreement of:

- real and NULL weights;
- pre/post-`W_O` hidden outputs;
- candidate-centred/bounded writes;
- gradients with respect to support, allocation, values, `W_O`, and LayerScale;
- Hessian-vector products at interior `rho`;
- no-history and partially masked behavior.

Any agreement within numerical tolerance is a **novelty failure**, not a test
pass.  The symbolic audit predicts exact agreement, so the ladder stops here.

## Hypothetical synthetic behavior gate

Only a genuinely non-reducible successor could proceed.  A balanced generator
would separately vary:

1. identical allocation with high versus near-zero real support;
2. identical support with different correct event allocation;
3. no history, candidate-common history, wrong-user history, shuffled history,
   and adversarial high-scoring irrelevant events;
4. exact-repeat and non-repeat candidates.

Compare exact NULL, output gate, sigmoid, entmax, ZAM target attention, and
dustbin controls under matched states/capacity.  Binding requirements:

- clean support and allocation both improve candidate margin;
- wrong/shuffled/query-masked histories drive effective post-`W_O` write—not
  only reported `rho`—toward zero;
- repeat is non-inferior to the same-checkpoint item-only path;
- no-history is bitwise base; common mode is zero; write is bounded;
- radial/tangent gradients remain trainable from the frozen small non-zero
  LayerScale initialization;
- the primary beats every nearest neighbour by a predeclared minimal effect.

Tie with NULL/gated attention or dependence on LayerScale alone closes the
candidate.

## Real A0 label-free safety gate

After a synthetic pass and fit-only training, freeze a new checkpoint before
opening train-internal outcomes.  A0 may read no dev/test/qrels labels.  It must:

- assert candidate/base/config hashes and no-history pointwise parity;
- exercise max batch, empty/all history masks, duplicate candidates, reload,
  serialization, and deterministic repeats;
- assert `sum_j w=rho`, `sum_j p=1` when `rho>0`, NULL mass `1-rho`, candidate
  permutation, centring, and global bound;
- compare matched clean/wrong-user/shuffle/query-mask distributions of `rho`,
  value norm, post-`W_O` effective write, and order-change rate;
- reject compensation in which lower `rho` is offset by larger `W_V/W_O` norm;
- verify two-step gradient movement at the smallest registered LayerScale;
- run the exact transformed NULL control and stop if output/gradient parity
  persists.

A0 cannot prove better ranking without labels.  It only blocks unsafe or
collapsed execution.  C14 is rejected before S0 implementation because exact
equivalence is already proven on paper.
