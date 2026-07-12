# C11 pre-lock rejection

Status: **REJECTED BEFORE LOCK; GPU execution forbidden.**

Independent review found an exact algebraic reduction in the proposed
predictive-gain primitive.  With tied linear decoder logits,

```text
log p_e(x_i,t) - log p_0(x_i,t)
  = E[x_i,t] dot (h_e - h_0) - (log Z_e - log Z_0).
```

For a fixed event and token position, the partition-function difference is
candidate-common.  The required candidate-centering projection removes it
exactly.  The remaining primary feature is
`center_i(E[x_i,t] dot (h_e-h_0))`, which is the registered eventwise-hidden
control up to its fixed `1/sqrt(d)` scale.  Applying `tanh` after different
fixed scales does not establish a distinct predictive-information primitive;
changing temperature would only repackage the same operator.

Consequences:

- the positive non-reduction claim in `reduction_and_control_audit.md` is false;
- the pre-lock manifest is permanently marked rejected and must never be
  promoted to `frozen_manifest.json`;
- no GPU, real, dev, test, or qrels run is authorized;
- the runner also omitted a binding `event_gain_std` decision despite logging
  it, so the executable gate did not fully implement the prose protocol;
- generator construct tests and structural contracts remain useful engineering
  artifacts, but they do not rescue the architectural hypothesis.

Any successor must use a candidate-conditioned predictive quantity whose
nonlinear normalization or interaction occurs **before** the shared
candidate-common term can be projected away, and must receive a new fingerprint
and reduction audit.  A temperature change, nonlinear map applied only after
the reduced scalar, or renamed hidden-state similarity is not eligible.
