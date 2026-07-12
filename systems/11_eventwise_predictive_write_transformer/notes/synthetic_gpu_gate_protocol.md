# C11 synthetic GPU gate protocol

Status: thresholds and seeds fixed for independent review; execution disabled
until an approved `frozen_manifest.json` exists.

- protocol: `c11_eventwise_predictive_write_synthetic_v1`
- proposed physical GPU: 3
- resource cap: one A40, 900 wall seconds / 0.25 A40-hours; abort after the
  first completed mode that crosses the cap
- run prefix: `20260711_kuaisearch_c11_`
- seeds: `2026071111`, `2026071112`, `2026071113`
- per seed: 4,096 fit, 2,048 untouched synthetic evaluation examples
- training: 5 epochs, batch 128, AdamW lr 0.001, weight decay 0.0001
- objective: the same listwise ranking loss only; no event-reliability role or
  generator factor is exposed to any model
- models: base, scalar-logit, pooled-C10, centred-attention,
  eventwise-hidden, eventwise-predictive
- initialization and data order: seed-matched across all modes
- candidate list: full seven candidates; one relevant candidate

Before creating an optimizer, the runner regenerates the 32,768-example
construct audit and requires every bound in `construct_audit.md`.  After
training, a history-blind base above transfer NDCG 0.80 invalidates the gate.

## Conjunctive outcome rule

The primary passes only if all conditions hold:

1. transfer NDCG gain over its own base is ≥0.03 in every seed;
2. mean advantage is ≥0.003 over each of pooled-C10, centred attention,
   eventwise-hidden, and scalar-logit, with at least two nonnegative seed wins;
3. exact-repeat NDCG is no more than 0.005 below its same-checkpoint internal
   item-only path in any seed;
4. wrong-user, shuffle, and evidence-query-mask gain retention is at most
   0.30/0.70/0.50 respectively;
5. transfer order changes on at least 10% of requests, delta standard deviation
   is at least 0.001, and the write is zero-sum/bounded;
6. no-history output is bitwise identical to the history-blind base.

One failed conjunct is a frozen failure.  This gate can establish only
conditional synthetic learnability; even a pass requires a new independent
review before any real train-internal materialization.  Dev/test/qrels access is
never authorized here.

The runner refuses execution unless (a) `frozen_manifest.json` exists, (b) its
review status is `approved_for_single_gpu_run`, (c) every hash matches, and (d)
`CUDA_VISIBLE_DEVICES=3`.
