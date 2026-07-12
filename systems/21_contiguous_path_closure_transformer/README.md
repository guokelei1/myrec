# C21 — Contiguous Path-Closure Transformer

Status: **train-only signal observability gate being frozen; no proposed
architecture or paper claim is authorized yet**.

C19 and C20 showed that synthetic orientation and positive transition
composition can be made algebraically active without producing a robust
ranking architecture.  C21 therefore starts one step earlier: it asks whether
real, label-isolated training data contains a candidate-discriminative signal
with the specific geometry needed by a temporal path primitive.

The probe tests whether the projected query-to-candidate displacement can be
closed by a short **contiguous, directed** history path whose start is compatible
with the query and whose end is compatible with the candidate.  It is not the
paper model: it consumes frozen D2p states and adds a bounded, candidate-centred
residual only to measure signal observability.  Passing the gate would authorize
a separate Transformer-internal design in which path closure controls the
history-to-candidate value write.  Failure closes this primitive before another
large architecture is built.

Hard boundaries:

- only C06's already isolated 12,000 `fit` requests may be used;
- the 9,000/3,000 fit/probe split is frozen before compact fit labels open;
- C06 `internal_A`, `internal_B`, `escrow`, dev, test, qrels and paper metrics
  are forbidden;
- one fixed recipe, three seeds, no sweep, retry, early stopping or checkpoint
  selection;
- the primary must beat the frozen D2p score and every matched operator control,
  then lose its gain under wrong-history and event-shuffle interventions;
- a pass authorizes architecture formulation only, never dev/test evaluation.

The binding protocol is in `notes/train_signal_gate_protocol.md`.
