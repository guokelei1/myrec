# B5o KuaiSearch Official Alignment

Date: 2026-07-09

Status: Stage A passed as `official-code, proxy-aligned (last-time 10% split)`.
This is not an upstream-confirmed Table 7 split guarantee. No KuaiSearch PPS dev
evaluation has been produced.

## Source

- Upstream repository: `https://github.com/benchen4395/KuaiSearch`
- Locked commit: `7ce0471b659112096f0aa7e892ed0aa4c972246a`
- License: MIT
- Environment: `pps-kuaisearch`, recorded in `configs/env/kuaisearch.txt`
- CLI sanity: `ranking/main.py --help` passed under `pps-kuaisearch`

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

## Adapter Fixes

The official repository does not expose a direct public-file reproduction path:
raw files are `rank_lite/items_lite/users_lite`, the official loader expects
`data/rank.jsonl`, `data/corpus.jsonl`, and `data/users.jsonl`, embeddings are
written to `./` but loaded from `./data/`, released users use `age_bucket` while
the loader reads `age`, and the released code uses
`BAAI/bge-small-zh-v1.5` despite the paper's BERT wording.

Repair source:

- `src/myrec/baselines/kuaisearch_materializer.py`
- `tests/test_kuaisearch_materializer.py`
- protocol diff: `reports/b5o_protocol_diff.md`

The adapter maps `age_bucket -> age`, emits legal default rows for missing
users, checks target item coverage, and moves official BGE outputs into the
loader's `./data/` path. No official source patch was required.

## Smoke AUC Direction Check

The 2000-row smoke reported DNN test AUC 0.377851. Before full runs, this was
checked in `reports/b5o_smoke_auc_direction_check.md`:

- official AUC equals manual AUC: 0.377851
- reversing scores or labels gives 0.622149
- materialized labels preserve raw `is_clicked` / `is_purchased`
- the official label rule is `is_clicked == 1 or is_purchased == 1`

Conclusion: no score-direction or label-inversion implementation bug was found.
The smoke AUC was treated as a tiny-split warning only.

## Proxy Full Materialization

Authorized proxy: `last_time_fraction`, latest 10% by `time_index`, with
threshold ties assigned to test. Decision note:
`doc/baseline_notes/20260709_b5o_stage_a_split_decision.md`.

Command:

```bash
PYTHONPATH=src python -m myrec.baselines.kuaisearch_materializer \
  --raw-dir data/raw/kuaisearch \
  --output-root artifacts/batch2b/b5o_proxy_lasttime_full \
  --split-policy last_time_fraction \
  --test-fraction 0.10 \
  --min-target-coverage 0.999
```

Manifest: `artifacts/batch2b/b5o_proxy_lasttime_full/materializer_manifest.json`

| Quantity | Value |
|---|---:|
| Ranking rows | 17,800,904 |
| Unique target items | 5,548,777 |
| Target coverage | 1.000000 |
| Corpus rows/items | 6,206,709 |
| Users | 102,086 |
| Synthetic missing users | 0 |
| Positive labels | 682,228 |
| Negative labels | 17,118,676 |
| Test threshold | `time_index >= 867165` |
| Train rows | 16,020,759 |
| Test rows | 1,780,145 |
| Actual test fraction | 0.100003 |

Official BGE encoding:

| Artifact | Shape / count |
|---|---:|
| `data/query_emb.npy` | `(555553, 512)`, float16 |
| `data/session_id2idx.json` | 555,553 entries |
| `data/item_title_emb.npy` | `(6206709, 512)`, float16 |
| `data/item_id2idx.json` | 6,206,709 entries |

BGE command log: `artifacts/batch2b/b5o_proxy_lasttime_full/bge_process.stdout`.
Elapsed wall time: 21:09.

## Proxy Alignment Runs

Both runs used the locked official `ranking/main.py` defaults:
`batch_size=4096`, `num_epochs=30`, `lr=1e-3`, `weight_decay=1e-5`,
`mixed_precision=bf16`, seed 42, and `early_stop_patience=2`.

Run count against the Stage A stop-loss cap: 2/6.

| Method | Target Logloss | Target AUC | Proxy LogLoss | Proxy AUC | Rel. Logloss Diff | Rel. AUC Diff | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| DNN | 0.1588 | 0.6258 | 0.160731 | 0.613133 | +1.22% | -2.02% | within +/-10% |
| DCNv2 | 0.1603 | 0.6239 | 0.162635 | 0.616348 | +1.46% | -1.21% | within +/-10% |

DNN details:

- log: `artifacts/batch2b/b5o_proxy_lasttime_full/train_dnn_proxy.stdout`
- early stop: epoch 8
- best checkpoint epoch: 6
- best valid LogLoss/AUC: 0.153592 / 0.689025
- final test LogLoss/AUC: 0.160731 / 0.613133
- elapsed wall time: 2:46:26

DCNv2 details:

- log: `artifacts/batch2b/b5o_proxy_lasttime_full/train_dcnv2_proxy.stdout`
- early stop: epoch 5
- best checkpoint epoch: 3
- best valid LogLoss/AUC: 0.150727 / 0.714827
- final test LogLoss/AUC: 0.162635 / 0.616348
- elapsed wall time: 1:46:00

Large checkpoints were removed after metrics were recorded; logs and materialized
data remain under ignored `artifacts/`.

## Verdict

B5o Stage A is `official-code, proxy-aligned (last-time 10% split)`. The result
is strong enough to say the locked official ranking code and adapter match the
Table 7 metric scale under the proxy split.

The claim is intentionally limited:

- it is not an upstream-confirmed Table 7 split guarantee;
- it does not authorize a main-table official claim without the caveat;
- it does not consume KuaiSearch PPS dev budget;
- it does not start Stage B on PPS standardized data.

Next decision, if desired: whether to run a scoped Stage B KuaiSearch adapter
under the same caveat, or keep B5o as an external-aligned proxy baseline only.
