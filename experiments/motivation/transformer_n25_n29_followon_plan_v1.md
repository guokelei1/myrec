# N25--N29 Transformer follow-on mechanism plan (pre-registered, inactive)

This document extends the mechanism audit only after the fixed N17--N20
boundary families have closed.  It does not authorize an architecture,
dataset, or paper-method change.  Every cell is an inference-time diagnostic
with frozen Q2/Q3 checkpoints, the existing standardized dev population,
shared score/evaluation code, and qrels opened only after complete score-bundle
integrity checks.

## Why this follow-on exists

The current inventory has direct boundaries for attention/MLP stage states,
residual composition, q/k head RMSNorm, GQA grouping, Q3 LoRA branches, and
Q1 cache phases.  Five implementation interfaces remain either only observed
geometrically or only covered by a downstream state patch:

1. the two inputs and the nonlinear product of the SwiGLU MLP;
2. final RMSNorm and the native answer-token readout;
3. causal-mask visibility and attention softmax normalization;
4. formation of the complete scaled pre-mask QK-logit tensor;
5. non-additive interaction between attention and MLP increments at one block.

These are deliberately separated because a full block-state patch cannot tell
whether a signal was erased by a nonlinearity, a visibility topology, a
normalization/readout, or composition of two increments.

## Fixed population and gates

- Models: `q2_recranker_generalqwen` and `q3_tallrec_generalqwen`.
- Fixed functional depths: blocks 13, 20, and 27; no effect-selected layer or
  head is allowed.  N25/N29 use all three blocks; N26 uses the native final
  readout; N27/N28 use all registered query/key heads at those blocks.
- Conditions: full, null, frozen wrong-user, same-prompt identity, exact
  re-add identity, zero/removal, scale half/double, sign reversal, and
  output-norm-matched random-direction controls where the operator permits it.
- Every bundle must contain all 8,000 dev requests, all candidate rows, finite
  scores, frozen candidate/request hashes, and qrels-blind metadata.  Ineligible
  rows copy the frozen baseline and are excluded from registered inference.
- Identity tolerance is `1e-5`; all operator hooks must fire exactly once per
  model forward.  A failed identity or position/mask audit is a mechanical
  non-result and blocks that family.
- Registered inference is clustered by normalized query, uses the existing
  strict-transfer surface and BH correction, and reports target-margin and
  NDCG@10 plus full/null/transfer-gap contrasts.

## N25: SwiGLU formation and nonlinearity

At each fixed block, hold the incoming residual, attention increment, and
down-projection fixed while intervening on `gate_proj`, `up_proj`, the SiLU
gate, and the complete `SiLU(gate) * up` product.  The required order is:

1. exact identity/re-add audit;
2. gate-only and up-only zero/scale/sign cells;
3. product-level removal and reverse-removal;
4. norm-matched random product direction;
5. full/null/wrong-user specificity and Q2/Q3 replication.

The claim boundary is “SwiGLU formation is implicated at a registered depth”
only if the product direction survives gate/up matched controls and the
downstream residual composition is not the sole explanation.

## N26: final RMSNorm and native readout (integration/replication)

The existing D6 Q2/Q3 native-readout bundles already capture final RMSNorm
input/output, exact tied-row/readout algebra, and Q3 teacher-forced terms.  N26
therefore does not authorize a duplicate sweep: it is a consolidation gate that
must re-audit those bundles under the same identity/causal rules and, only if a
specific operator claim remains unresolved, run the smallest predeclared
variance-vs-gain replication.  Tied-row and untied-row metadata must be
recorded; no candidate labels or qrels may enter the readout hook.  Native
full-sequence and cache-free identity are required before utility comparisons.

## N27: causal-mask and softmax topology

Use position-matched mask alternatives that change only the registered
query/history/candidate visibility edges while preserving every answer-token
causal boundary.  Separately compare native softmax with temperature and
centered-logit controls on the same pre-mask tensor.  Leakage, token-position,
and candidate-order audits are mandatory; any answer-label or future-token
visibility invalidates the cell.

## N28: complete scaled QK-logit formation

Capture the pre-mask tensor after q/k normalization, RoPE, head grouping, and
the `1/sqrt(d_head)` scale.  Intervene on the complete tensor (not a selected
edge) with identity, centered scale, sign, head-preserving random direction,
and reverse-removal controls while holding Q/K/V, mask, softmax, and `o_proj`
fixed.  N17/N18 results are covariates, not substitutes for this formation
test.

## N29: attention--MLP non-additive residual composition

At each fixed block, run a preregistered 2x2 factorial over native/removal of
the attention and MLP increments, plus matched scale/sign controls.  Estimate
the interaction term in both hidden-state geometry and native score space,
with full/null/wrong-user and reverse-removal controls.  This is the smallest
test that can distinguish “one increment erased the signal” from “the two
increments jointly rotate or cancel it”.

## Compute order and stopping point

1. Finish the current four-card wave and its shared evaluator.
2. Run N17/N18 in fixed four-lane block/model waves, then N19 in fixed two-block
   four-lane waves; N20 Q1 starts only after N19 evaluator closeout.
3. Run N25/N28/N29 in four-card waves (two models × two fixed blocks at a
   time), followed by N26/N27 readout/mask lanes.  N21--N24 training-boundary
   diagnostics use separate seeds and never preempt inference jobs.
4. Stop after the H0--H5 evidence matrix identifies supported, contradicted,
   and unresolved interfaces.  Do not add heads/layers/seeds based on observed
   effects, and do not implement a transfer architecture in this stage.
