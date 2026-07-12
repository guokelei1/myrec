# C22 pre-implementation decision

Decision: **PROCEED with exactly one locked synthetic GPU falsifier; no real
data or broad novelty claim is authorized**.

## Why implementation is justified

- C21 rules out further geometric aggregation over the same frozen states but
  does not test whether reliable evidence can be kept causally isolated through
  every Transformer layer.
- Autograd tests establish the defining asymmetry exactly: protected Jacobians
  are zero while recurrence-to-transfer coupling remains active.
- Dense, parallel and final-projection controls have identical parameters and
  initialization.  The final-projection control is deliberately strong: if a
  late C18-style safeguard is enough, filtration has no value.
- The generator has no marginal target shortcut.  On 20,000 pre-outcome audit
  rows, base accuracy is 1.0 for no-history and approximately 0.124 for repeat
  and supported strata; identity and joint `q+h_last` oracles are exactly 1.0.
- GPU smoke has finite loss/gradients and bitwise no-history fallback.  A
  120-step 64-row overfit-only engineering check reduced mean loss from 0.610
  in its first 20 steps to 0.380 in its last 20; no evaluation split or gate
  outcome was inspected.
- Twenty full-size training steps took 0.89 seconds on the assigned A40, so the
  complete one-shot falsifier fits the compute stop-loss.

## Novelty boundary

StairFormer already establishes block-triangular/prefix-preserving Transformer
machinery; Dual/Relational Attention already separates information types.  C22
therefore cannot claim those devices.  Only a consistent advantage of the
reliability-ordered causal graph over every matched control would justify a
domain-specific architecture claim and a broader literature review.

Thirteen structural tests pass.  Freeze source, config, tests, environment and
this decision before the formal optimizer.  Failure is terminal for C22.
