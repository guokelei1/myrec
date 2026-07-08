# RecBole Boundary Card

Purpose: B4o official SASRec/BERT4Rec baseline for Batch 2b.

## Upstream

- Repository: https://github.com/RUCAIBox/RecBole
- Package used: `recbole==1.2.1` from PyPI
- Source mode: installed package, not vendored source
- Commit hash: not applicable to the PyPI package; if a source checkout is
  later needed, record the exact upstream commit here before running it.
- License notes: PyPI package metadata has an empty `License` field. Verify the
  upstream repository license before any paper artifact release that redistributes
  code or patches.

## Environment

Environment group: `recbole`

Setup summary:

```bash
conda create -n pps-recbole python=3.10 pip -y
conda run -n pps-recbole python -m pip install \
  "torch==2.2.2" "numpy<2" "pandas<2.2" "scipy<1.12" \
  "scikit-learn<1.4" "recbole==1.2.1"
conda run -n pps-recbole python -m pip install "setuptools<81"
conda run -n pps-recbole python -m pip freeze > configs/env/recbole.txt
```

Key compatibility note: `ray==2.6.3` imports `pkg_resources`, which requires
`setuptools<81` in this environment.

## Input And Output Boundary

Input data comes from the project standardized interface only:

- `data/standardized/kuaisearch/v0_lite/records_train.jsonl`
- `artifacts/batch2b/interactions_train.jsonl`
- blind split records for scoring, with frozen per-record history only

Forbidden inputs:

- `qrels_dev.jsonl` / `qrels_test.jsonl`
- dev/test labels
- user-global history sequences at inference
- original raw KuaiSearch logs

Output must be project-standard `runs/<run_id>/scores.jsonl`, then evaluated by
`scripts/evaluate_scores.py`. RecBole's own evaluator may be used only for the
external ml-100k sanity report, not for paper KuaiSearch metrics.

## Current Status

External sanity passed on RecBole's bundled ml-100k example with SASRec:

- report: `reports/b4o_env_sanity.md`
- loss decreased from 336.5817 to 299.0741 over 3 epochs
- final example-split test NDCG@10: 0.0358

Next work: implement the KuaiSearch atomic data adapter and fixed-candidate
scoring adapter.
