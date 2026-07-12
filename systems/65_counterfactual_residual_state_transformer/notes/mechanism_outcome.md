# C65 G0 outcome

Status: **failed terminal before any label access**.

The internal mechanism was active: factual-minus-NULL state RMS was `0.2345`,
wrong-history residual RMS was `0.2402`, true/wrong scores differed by up to
`0.0401`, hidden-state readout differed from ordinary and logit-difference
controls, and gradients reached both adaptive BGE layers, the joint Transformer,
residual norm, and output head.  No-history/repeat and determinism errors were
zero; every mode had 24,304,768 parameters.

The frozen candidate-permutation contract failed.  Reversing caller candidate
storage changed primary scores by `3.64e-5` against a `2e-6` tolerance.  C64's
single factual state was fp32-equivariant at G0, but C65 subtracts two
near-equal listwise states and normalizes the small residual; this amplifies
otherwise harmless reduction-order differences.

C65 is closed and may not train.  The only scientifically neutral continuation
is a separately locked candidate that canonicalizes every branch by stable
item ID before the identical computation, then restores caller order.  It may
not change the formula, initialization, modes, loss weights, LM layers,
optimizer, data split, thresholds, or precision.
