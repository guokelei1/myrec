# C02 — Candidate-Conditioned History HyperAdapter Transformer

Terminal status: **closed at the valid train-internal gate after one authorized
mechanical continuation; 2/6 checks passed, no dev score/evaluator/test.**  See
`notes/mechanical_continuation_outcome.md` and
`reports/pps_c02_mechanical_continuation_gate.json`.

This directory owns the isolated C02 design/probe.  The candidate is called
**CHHT**.  Its load-bearing primitive is a request/candidate-specific,
history-composed low-rank rotation of an internal Transformer FFN map.  It is
not a score router and it does not create a permanent user adapter.

The proposal is frozen in `notes/proposal_lock.json` before any C02 GPU model
outcome.  The one-call dev screening is provisional and cannot authorize
multi-seed or full training.  Test records and test labels are outside this
track's boundary.

## Owned resources

- environment: `myrec-c02` under `/data/gkl/conda_envs`;
- physical GPU: 1, always exposed to the program as `cuda:0`;
- run prefix: `20260710_kuaisearch_c02_`;
- source root: this directory only;
- raw state: `artifacts/c02_history_hyperadapter/`,
  `models/c02_history_hyperadapter/`, and matching `runs/` IDs.

## Reproducible command sequence

All commands run from the repository root.

```bash
CONDA_ENVS_PATH=/data/gkl/conda_envs \
  conda create -n myrec-c02 --clone pps-kuaisearch -y

CUDA_VISIBLE_DEVICES=1 \
  /data/gkl/conda_envs/myrec-c02/bin/python -m unittest discover \
  -s systems/02_history_hyperadapter/tests -v

CUDA_VISIBLE_DEVICES=1 \
  /data/gkl/conda_envs/myrec-c02/bin/python \
  systems/02_history_hyperadapter/train/prepare_features.py \
  --config systems/02_history_hyperadapter/configs/screen.yaml

CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=1 \
  /data/gkl/conda_envs/myrec-c02/bin/python \
  systems/02_history_hyperadapter/train/train_screen.py \
  --config systems/02_history_hyperadapter/configs/screen.yaml

CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=1 \
  /data/gkl/conda_envs/myrec-c02/bin/python \
  systems/02_history_hyperadapter/train/score_screen.py \
  --config systems/02_history_hyperadapter/configs/screen.yaml

flock tmp/pps_dev_evaluator.lock \
  /data/gkl/conda_envs/myrec-c02/bin/python scripts/evaluate_scores.py \
  --run-id 20260710_kuaisearch_c02_chht_screen_s20260708 \
  --candidate-manifest \
  data/standardized/kuaisearch/v0_lite/candidate_manifest.json

CUDA_VISIBLE_DEVICES=1 \
  /data/gkl/conda_envs/myrec-c02/bin/python \
  systems/02_history_hyperadapter/train/analyze_screen.py \
  --config systems/02_history_hyperadapter/configs/screen.yaml
```

`score_screen.py` asserts the frozen candidate-manifest hash before writing the
only evaluable output, `runs/<run-id>/scores.jsonl`.  The scorer and trainer do
not accept paths to separated evaluation labels.

`CUBLAS_WORKSPACE_CONFIG=:4096:8` is required because C02 enables PyTorch's
deterministic-algorithm mode; without it CUDA linear algebra stops before the
first forward pass.

## Layout

- `model/`: compact Transformer core and the Cayley HyperAdapter operator;
- `train/`: candidate-local feature preparation, training, scoring, and audit;
- `configs/`: frozen screening configuration;
- `tests/`: operator, masking, schema, and determinism unit tests;
- `notes/`: proposal, mechanism fingerprint, literature audit, gate, lock, and
  final report.
