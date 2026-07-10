# 2026-07-10 - Supervised Diagnostic Dynamic Batching

Materialization showed candidate-count mean 47.8 and maximum 2,196. A fixed
512-request padded batch would let one outlier inflate every tensor in that
batch.

Before training, the loader was frozen to close a batch when any condition is
met:

- 512 requests;
- 65,536 padded candidate slots;
- 32,768 padded history-event slots.

The loader does not truncate or reorder candidates within a request. The same
rule is used for internal calibration, final training, and label-free scoring.
