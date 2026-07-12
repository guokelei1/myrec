# C03 — Triadic Cycle-Consistent Transport Transformer

This directory owns candidate C03.  The locked minimal probe implements a
candidate-anchored **cycle-intersection partial transport** operator inside a
compact Transformer ranker.  It is not a claim that generic optimal transport,
Sinkhorn normalization, cycle consistency, or dustbins are novel.

The probe keeps the registered D2p score as an exact no-history skip contract.
For history-present requests, a local Transformer jointly contextualizes a
query state, candidate state, and strictly-prior history-event states.  Three
differentiable partial transport plans (`q↔h`, `h↔c`, and `q↔c`) have learned
null scores.  Only their non-null cycle-intersection mass can update the
candidate state and produce a centered signed ranking residual.  Exact item
identity appears only as a protected low-cost atom in `h↔c` transport.

This is a falsifier-scale implementation.  It uses a frozen local
`BAAI/bge-small-zh-v1.5` text encoder and trains the interaction Transformer and
transport path.  It is not the authorized full system.  A screening survivor
must still wait for a separately budgeted full design gate.

## Boundaries

- Candidate directory: `systems/03_triadic_transport_transformer/`
- Environment: `myrec-c03`
- Physical GPU: 2 (`CUDA_VISIBLE_DEVICES=2`, program-visible `cuda:0`)
- Seed: `20260708`
- Run prefix: `20260710_kuaisearch_c03_`
- Candidate manifest SHA256:
  `94eb667000e0d0f389d0a2a4d4730683b71c129043edbfcf627590376e9c123e`
- Dev evaluator budget: exactly one primary screening call
- Test records, test qrels, test metrics, and all qrels in training/scoring are
  forbidden.

## Layout

```text
configs/  frozen screening and smoke settings
model/    Transformer and transport operator
train/    candidate-local feature, training, scoring, and audit entry points
tests/    hand-computed transport and contract tests
notes/    proposal, fingerprint, nearest-neighbor audit, gate, and reports
```

## Intended execution

```bash
export CONDA_ENVS_PATH=/data/gkl/conda_envs
export CUDA_VISIBLE_DEVICES=2

conda run -n myrec-c03 python -m pytest -q \
  systems/03_triadic_transport_transformer/tests

conda run -n myrec-c03 python \
  systems/03_triadic_transport_transformer/train/run_probe.py \
  --config systems/03_triadic_transport_transformer/configs/c03_screening.yaml \
  prepare

conda run -n myrec-c03 python \
  systems/03_triadic_transport_transformer/train/run_probe.py \
  --config systems/03_triadic_transport_transformer/configs/c03_screening.yaml \
  train
```

The remaining locked commands, including deterministic rescore, primary score
materialization, the single shared-evaluator call, and adjudication, are listed
in `notes/gate_protocol.md`.
