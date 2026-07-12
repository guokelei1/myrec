# C22 nearest-neighbour audit

Status: **pre-outcome; operator-level novelty uncertain**.

| Neighbour | Overlap | Required C22 distinction |
|---|---|---|
| Fully Nested Transformers / StairFormer | block-lower-triangular attention/FFN and prefix-preserving normalization | C22 assigns the filtration to evidence reliability inside one candidate ranker and tests causal recurrence preservation, not subnet nesting; generic masks are not claimed as new |
| Dual Attention Transformer | separates sensory and relational computation | C22 imposes an ordered one-way residual algebra and protected prefix Jacobian, not two symmetric attention modules |
| Relational Attention / Abstractors | explicit edge/relational states | C22 does not maintain free learned edge tensors; exact equality only initializes the reliable quotient |
| induction heads / associative retrieval | exact/pattern retrieval through attention | C22 protects the retrieved evidence across all later blocks; it does not claim equality lookup itself |
| AC-TSR and Pathway Attention | recalibrate or sparsely select history events | C22 constrains where evidence may write, not which events receive larger scalar attention |
| C18 ECOT | recurrence-safe final order projection | C22 acts at every hidden layer and permits one-way recurrence-to-transfer conditioning before readout |
| C02 hyperadapter | candidate/history-conditioned functional update | C22 uses a fixed basis and masks; no candidate-conditioned weight or basis generator exists |

Primary sources reviewed:

- Fully Nested Transformers, OpenReview 2026:
  https://openreview.net/pdf?id=yv7Ie3UzlA
- Dual Attention Transformer, ICML 2025:
  https://openreview.net/forum?id=lbrqeIipJr
- Relational Attention, ICLR 2023:
  https://openreview.net/forum?id=cFuMmbWiN6
- In-context Learning and Induction Heads:
  https://arxiv.org/abs/2209.11895
- Attention Calibration for Transformer-based Sequential Recommendation:
  https://arxiv.org/abs/2308.09419
- Recommender Transformers with Behavior Pathways:
  https://openreview.net/forum?id=DSoFfnmUSjS

Verdict: C22 is distinct from ordinary dense/parallel/final-projection controls,
but the algebraic substrate is close to StairFormer.  It may proceed only as a
bounded empirical architecture hypothesis.  A paper-level novelty claim
requires a clear matched-control gain and broader review after the gate.
