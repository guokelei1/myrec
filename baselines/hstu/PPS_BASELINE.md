# PPS HSTU baseline boundary

This directory is a vendored upstream snapshot for the reusable rec-native
diagnostic baseline recorded in `experiments/pps_baseline_cards.md`.

## Upstream

- Repository: `https://github.com/meta-recsys/generative-recommenders`
- Locked commit: `6135bc30398f97e5786674192558d91f2ef2fa90`
- Paper: *Actions Speak Louder than Words: Trillion-Parameter Sequential
  Transducers for Generative Recommendations*, ICML 2024
- License: Apache-2.0; the upstream `LICENSE` is retained verbatim.
- Import method: `git archive` of the locked commit. No upstream source file
  has been edited.

The `generative_recommenders/ops/cpp/cutlass` submodule is intentionally not
vendored. The first PPS implementation uses the upstream PyTorch kernel path;
CUDA/Triton efficiency kernels are not required to establish model behavior.

## Environment boundary

The locked upstream declares:

```text
torch>=2.6.0
fbgemm_gpu>=1.1.0
torchrec>=1.1.0
gin_config>=0.5.0
pandas>=2.2.0
tensorboard>=2.19.0
pybind11
```

Install these in a dedicated baseline environment. Do not change the project
environment silently to make upstream code import. The local adapter must log
the exact Torch, CUDA, fbgemm-gpu, torchrec, and Triton versions.

## Official versus local code

Official, unchanged code:

- HSTU and SASRec research implementations under
  `generative_recommenders/research/modeling/sequential/`;
- official HSTU modules and PyTorch/Triton ops;
- upstream public experiment configs.

Local work, which must live under `src/myrec/baselines/` rather than editing
this tree:

- standardized JSONL reader and ID vocabulary;
- frozen query/candidate content encoder boundary;
- query-conditioned fixed-candidate scorer;
- QC/FULL and true/null/wrong materialization;
- score writer compatible with the shared PPS evaluator.

The resulting method must be called **official HSTU core + PPS task adapter**.
It is not an exact reproduction of the paper's MovieLens/Amazon next-item
numbers.

## Input and output contract

Input is the project standardized record interface only. Training may read
`qrels_train.jsonl`; scoring code reads label-free `records_dev.jsonl` or a
future authorized label-free confirmation split and never reads development or
test qrels.

The adapter must preserve and log:

- `request_id`, `query`, ordered strictly prior history, and candidate order;
- item/action/category/time field masks;
- dataset and request/candidate manifest hashes;
- HSTU versus SASRec architecture ID and all capacity/training settings.

Output is one score per request/candidate with the same candidate order and the
metadata required by `experiments/history_response_gap/score_bundle_contract.md`.

## Generated state

Do not write processed datasets, checkpoints, embeddings, caches, or logs into
this directory. Use `data/`, `models/`, `artifacts/`, and `runs/`.

## Current status

The adapter passed historical mechanics checks but did not pass the ranking
adequacy gate needed for the current motivation. Generated smoke reports were
removed from the curated report set during V1 consolidation. The reusable
production adapter remains `src/myrec/baselines/hstu_pps_adapter.py`; any future
use must rerun its tests and emit fresh evidence through the shared evaluator.
