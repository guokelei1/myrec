# C05 G2a terminal failure

Date: 2026-07-11

C05 review1 was locked before any candidate fit outcome and then executed once
on physical GPU 0.  G0 passed all manifest, selection, full-candidate, FP32 D2p
and key-alignment checks.  G1 passed the real maximum-work two-step gradient,
checkpoint-reload and same-seed zero-residual controls.

The fixed final G2a checkpoint failed its train-internal gate: D2p and the probe
both obtained NDCG@10 `0.3118520542` on 1,200 requests, for delta `0.0`, paired
bootstrap CI `[0.0, 0.0]`, and three hash-fold deltas all equal to zero.
Deterministic rescoring was bit-identical.  No dev records/qrels/evaluator or
test data were read.

A read-only diagnostic after the terminal decision showed that all 54,637
model-internal tanh deltas had saturated at exactly `+1`; the apparent
`2.48e-8` score-delta standard deviation came only from FP32 subtraction.
Zero request rankings changed. The failure is therefore consistent with a
common-mode, ranking-null collapse rather than a zero-movement no-op: final
parameter L2 movement was 4.636, although the saturated terminal gradient may
be near zero.

Per the frozen stop rule, G2b, CCEB, dev, and full training remain unauthorized,
and the exposed internal set will not be reused for a rescue attempt.  A future
successor would need a new untouched cohort and a separately locked
candidate-relative score-space primitive.  Authoritative tracked result:
`reports/pps_c05_g2a_signal_gate.json`.
