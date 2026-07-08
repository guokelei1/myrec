# B4o RecBole Environment Sanity

Date: 2026-07-09

Status: passed.

Scope: external RecBole/ml-100k sanity only. This run did not read KuaiSearch
dev/test records, qrels, candidate manifest, or any project score files. It does
not count against the B4o 16-evaluation KuaiSearch dev budget.

## Environment

| Field | Value |
|---|---|
| Environment group | recbole |
| Environment name | pps-recbole |
| Python | 3.10.20 |
| RecBole | 1.2.1 |
| torch | 2.2.2+cu121 |
| CUDA available | true |
| torch CUDA | 12.1 |
| numpy | 1.26.4 |
| pandas | 2.1.4 |
| ray | 2.6.3 |
| Freeze file | `configs/env/recbole.txt` |

Compatibility note: `ray==2.6.3` imports `pkg_resources`, so the environment
pins `setuptools==80.10.2`. The import warning is expected under this old ray
dependency and is controlled by the pin.

## Command

```bash
CUDA_VISIBLE_DEVICES=0 \
conda run -n pps-recbole \
python artifacts/batch2b/recbole_sanity/run_ml100k_sasrec_sanity.py
```

The sanity script used RecBole's bundled `dataset_example/ml-100k` and
`run_recbole(model="SASRec", dataset="ml-100k", saved=False)`.

Configuration changes relative to the SASRec defaults were limited to sanity
runtime controls: `epochs=3`, `show_progress=False`, larger eval/train batches,
`checkpoint_dir` under `artifacts/batch2b/`, and
`train_neg_sample_args=None` because RecBole SASRec uses CE loss and rejects
negative sampling for that loss type.

## Evidence

| Epoch | Train loss | Valid Recall@10 | Valid NDCG@10 |
|---|---:|---:|---:|
| 0 | 336.5817 | 0.0424 | 0.0198 |
| 1 | 311.3353 | 0.0785 | 0.0332 |
| 2 | 299.0741 | 0.0986 | 0.0458 |

Final test result on the RecBole example split:

| Metric | Value |
|---|---:|
| Recall@10 | 0.0732 |
| NDCG@10 | 0.0358 |

Conclusion: RecBole SASRec trains, evaluation completes, loss decreases, and
the GPU path is usable. B4o may proceed to the KuaiSearch data/scoring adapter.
