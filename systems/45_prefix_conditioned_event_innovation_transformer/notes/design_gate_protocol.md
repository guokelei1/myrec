# C45 frozen synthetic design-gate protocol

## Scope

The gate is data-free. It may use physical GPUs 0--2 only after the proposal
lock verifies. It may not read repository datasets, labels, qrels, dev/test
records, prior candidate score arrays, or the shared evaluator.

## Synthetic task

A fixed nonlinear latent transition generates ordered behavior events. At each
step the latent factual state and its no-event transition differ by an
event-specific, prefix-dependent effect. Current relevance combines a query
direction with the accumulated latent event effects. Candidate sets contain a
positive, query-hard negatives, and random negatives. This task is not evidence
that the repository data obeys the law; it only tests whether the proposed
inductive bias is learnable and pays rent over its reductions.

The generator is frozen at 4,096 train and 1,200 validation requests, history
length 6, candidate count 12, and seed 20262900. Generator audit must establish
finite tensors, positive-varying labels, base headroom, and non-degenerate
wrong/shuffle interventions before model training.

## Training

Seeds 20262901/02/03 map to physical GPUs 0/1/2. Every mode has identical
parameters, seed-specific initialization, 360 optimizer steps, batch 64,
AdamW learning rate 0.002, weight decay 0.0001, and clipping at 1.0. Modes are
`innovation`, `ordinary_delta`, `factual_state`, and `raw_event`. All four
execute both factual and NULL transitions, so parameter and transition-call
counts match.

## D0 structural requirements

- equal parameter count and paired initialization across modes;
- finite loss, state, score, gradient, and update;
- all trainable parameter groups receive a nonzero gradient;
- deterministic rescore max absolute difference 0;
- candidate permutation max absolute difference at most 1e-6;
- no-history score equals the internal query/candidate base bit exactly;
- query-absent personalized correction is bit-exact zero;
- repeat wrapper equals supplied item-only scores bit exactly;
- event=NULL makes the primary innovation token bit-exact zero;
- repository data/label/dev/test/qrels access declarations remain false.

## D1 predictive requirements

- primary clean NDCG@10 minus its base is at least 0.035 in every seed;
- mean primary margin over each matched control is at least 0.010;
- primary beats each control in at least two of three seeds;
- wrong-user and shuffled-event gain retention are each at most 0.45 in every
  seed when clean gain is positive;
- clicked-minus-unclicked correction is positive in every seed;
- primary changes at least 10% of complete candidate orders from its base in
  every seed.

Any failure is terminal for C45. No extra steps, alternative generator, sign
flip, width, threshold, loss, seed, or mode removal is permitted. Passing D1
authorizes design of a new train-internal gate only; it does not authorize
repository access by the current runner, dev/test, or full training.

## Formal commands after lock

```bash
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /data/gkl/conda_envs/myrec-c37/bin/python \
  systems/45_prefix_conditioned_event_innovation_transformer/probe/run_design_gate.py \
  --config systems/45_prefix_conditioned_event_innovation_transformer/configs/design_gate.yaml \
  --stage seed --seed 20262901 --device cuda:0

# Repeat on GPUs 1 and 2 with their registered seeds, then:
/data/gkl/conda_envs/myrec-c37/bin/python \
  systems/45_prefix_conditioned_event_innovation_transformer/probe/run_design_gate.py \
  --config systems/45_prefix_conditioned_event_innovation_transformer/configs/design_gate.yaml \
  --stage aggregate
```
