# B5o KuaiSearch Official Alignment

Date: 2026-07-09

Status: alignment not verifiable from the official repository at the locked
commit. No KuaiSearch dev evaluation has been produced.

## Source

- Upstream repository: `https://github.com/benchen4395/KuaiSearch`
- Locked commit: `7ce0471b659112096f0aa7e892ed0aa4c972246a`
- License: MIT
- Environment: `pps-kuaisearch`, recorded in `configs/env/kuaisearch.txt`
- CLI sanity: `ranking/main.py --help` passed under `pps-kuaisearch`.

## Paper Targets

The KuaiSearch paper reports ranking-task CTR metrics on KuaiSearch-Lite
Table 7:

| Method | Logloss | ROC-AUC |
|---|---:|---:|
| DNN | 0.1588 | 0.6258 |
| Wide & Deep | 0.1598 | 0.6217 |
| DCN | 0.1611 | 0.6194 |
| DCNv2 | 0.1603 | 0.6239 |
| DIN | 0.1606 | 0.6262 |

The official `ranking/main.py` reports `LogLoss` and `AUC`, so these are the
nominal alignment targets.

## What Ran

A tiny official-code smoke run was executed to verify that the checked-out
ranking code can train and evaluate in the `pps-kuaisearch` environment.

Smoke input:

- Source: `baselines/kuaisearch_official/demo/rank.jsonl`
- Rows: 200
- Train rows: 160
- Test rows: 40
- Synthetic bridge: `age = age_bucket`, minimal `corpus.jsonl`, random 512-d
  query/item embeddings
- Command: `accelerate launch --num_processes 1 ranking/main.py --model DCNv1
  --data_dir data --batch_size 64 --num_epochs 1 --mixed_precision no`

Smoke output:

| Metric | Value |
|---|---:|
| Train loss | 0.6980 |
| Valid loss | 0.6055 |
| Valid AUC | 0.7857 |
| Test LogLoss | 0.7459 |
| Test AUC | 0.3873 |

This smoke run is environment/code-path evidence only. It uses synthetic random
embeddings and a schema bridge, so it is not an alignment result and does not
count against the KuaiSearch dev budget.

## Why Alignment Is Not Verifiable

The current official repository does not provide a self-consistent, direct
stage-A reproduction path for the paper ranking numbers:

- The official scripts expect `data/rank.jsonl`, `data/corpus.jsonl`, and
  `data/users.jsonl`, while the released raw files are organized as
  `rank_lite/train.jsonl`, `items_lite/train.jsonl`, and
  `users_lite/train.jsonl`.
- `ranking/data/process.py` writes `query_emb.npy`, `session_id2idx.json`,
  `item_title_emb.npy`, and `item_id2idx.json` into the current working
  directory, but `ranking/datasets.py` reads those files from `./data/`.
- Released user rows use `age_bucket`; `ranking/datasets.py` reads `age`.
  With a matched user row this produces `age_idx = None`; with a missing user,
  the fallback indices `gender=2` and `age=9` exceed the declared embedding
  cardinalities `(2, 8)` and `(7, 16)`.
- The demo ranking file and demo item file use incompatible item-id spaces:
  the 200 demo ranking targets are reindexed ids 38-330, while the demo item
  file contains original ids in the thousands/millions; target coverage is 0.
- The paper text describes query/title embeddings from a BERT encoder, while
  the released `ranking/data/process.py` uses `BAAI/bge-small-zh-v1.5`.

These gaps require adapter decisions or source patches before the official
ranking pipeline can be run on the public files. Under `doc/14_official_baseline_plan.md`
section 4.1, this means B5o must be downgraded to
`official-code, alignment-not-verifiable`; it must not be described as an
official reproduction in the main table.

## Verdict

B5o Stage A is downgraded: official code is present and executable, but paper
number alignment is not verifiable at the locked commit without protocol
bridges. No Stage B KuaiSearch dev evaluation is authorized from this evidence.
