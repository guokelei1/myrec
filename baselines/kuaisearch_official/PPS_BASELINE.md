# PPS B5o KuaiSearch Official Baseline

This directory vendors the ranking-code subset used for Batch 2b B5o.

## Upstream

- Repository: `https://github.com/benchen4395/KuaiSearch`
- Commit: `7ce0471b659112096f0aa7e892ed0aa4c972246a`
- License: MIT according to the upstream README badge. No separate LICENSE file
  was present at the locked commit.

## Environment

- Environment group: `pps-kuaisearch`
- Manifest: `configs/env/kuaisearch.txt`
- Sanity: `conda run -n pps-kuaisearch python baselines/kuaisearch_official/ranking/main.py --help`

## Expected Official Inputs

The upstream ranking loader expects:

- `data/rank.jsonl`
- `data/corpus.jsonl`
- `data/users.jsonl`
- `data/query_emb.npy`
- `data/session_id2idx.json`
- `data/item_title_emb.npy`
- `data/item_id2idx.json`

## Current PPS Status

B5o is downgraded to `official-code, alignment-not-verifiable`.

Evidence:

- `reports/b5o_official_alignment.md`
- `reports/b5o_protocol_diff.md`

No KuaiSearch dev run was produced. The official code path was smoke-tested
with tiny synthetic bridge data under ignored `artifacts/batch2b/`.

## Local Scope

Only ranking code needed for B5o is retained here. Upstream demo data,
relevance data, recall code, site assets, generated caches, and checkpoints are
not tracked in this repository.
