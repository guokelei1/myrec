# B5o KuaiSearch Official Alignment

Date: 2026-07-09

Status: Stage A passed as `official-code, proxy-aligned (last-time 10% split)`;
Stage B PPS dev evaluation is complete under the same caveat. This is not an
upstream-confirmed Table 7 split guarantee, and B5o must always be reported with
the proxy-aligned identity.

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

## Stage B PPS Adapter

Stage B uses the locked official model/trainer classes through
`src/myrec/baselines/kuaisearch_official_adapter.py` and
`scripts/run_b5o_kuaisearch_official.py`. No official source patch was required.

Boundary:

- identity: `official-code, proxy-aligned (last-time 10% split)`;
- config: `configs/baselines/b5o_kuaisearch_din_dcnv2.yaml`;
- artifact root: `artifacts/batch2b/b5o_stageb_standardized`;
- train input: standardized train candidate rows with clicked/purchased labels;
- dev scoring input: label-free fixed candidates from `records_dev.jsonl`;
- qrels read by method: false;
- shared evaluator candidate manifest:
  `94eb667000e0d0f389d0a2a4d4730683b71c129043edbfcf627590376e9c123e`;
- dev eval budget used: 6/16.

Stage B materialization and embedding artifacts:

| Quantity | Value |
|---|---:|
| Train records | 163,717 |
| Train rank rows | 6,241,354 |
| Dev records | 12,229 |
| Dev score rows | 575,609 |
| Corpus items | 2,778,902 |
| Users | 52,133 |
| Query keys | 175,946 |
| Item title embeddings | `(2778902, 512)`, float16 |
| Query embeddings | `(175946, 512)`, float16 |

The locked loader uses `recently_clicked_item_ids` capped to 20. The adapter
preserves purchased history in the materialized JSONL for audit, but the locked
official loader ignores it.

## Stage B Dev Runs

Official defaults were used for both DNN and DCNv2. No hyperparameter grid was
run; the three seeds are frozen-seed repeats.

| Model | Seed | Run ID | Best internal valid loss/AUC | NDCG@10 | MRR | Recall@10 | pNDCG@10 |
|---|---:|---|---:|---:|---:|---:|---:|
| DNN | 20260708 | `20260709_kuaisearch_b5o_dnn_dev_s20260708` | 0.149689 / 0.702160 | 0.3088 | 0.2850 | 0.5334 | 0.3191 |
| DNN | 20260709 | `20260709_kuaisearch_b5o_dnn_dev_s20260709` | 0.150712 / 0.700120 | 0.3030 | 0.2772 | 0.5291 | 0.3024 |
| DNN | 20260710 | `20260709_kuaisearch_b5o_dnn_dev_s20260710` | 0.149567 / 0.699730 | 0.3072 | 0.2817 | 0.5321 | 0.3164 |
| DCNv2 | 20260708 | `20260709_kuaisearch_b5o_dcnv2_dev_s20260708` | 0.147644 / 0.721909 | 0.3056 | 0.2855 | 0.5261 | 0.3168 |
| DCNv2 | 20260709 | `20260709_kuaisearch_b5o_dcnv2_dev_s20260709` | 0.148659 / 0.719736 | 0.3051 | 0.2824 | 0.5287 | 0.3150 |
| DCNv2 | 20260710 | `20260709_kuaisearch_b5o_dcnv2_dev_s20260710` | 0.146007 / 0.730978 | 0.3056 | 0.2842 | 0.5269 | 0.3125 |

Frozen-seed summaries:

| Model | Mean NDCG@10 | Sample std | Mean MRR | Mean Recall@10 | Mean pNDCG@10 |
|---|---:|---:|---:|---:|---:|
| DNN | 0.3063 | 0.0030 | 0.2813 | 0.5315 | 0.3127 |
| DCNv2 | 0.3054 | 0.0002 | 0.2840 | 0.5272 | 0.3148 |

The formal B5o registry row uses the best DNN seed, matching the existing B4o
reporting convention, and records the DNN three-seed mean in the seed column.

## Stage B Determinism

`reports/b5o_determinism_check.json` reloads the frozen DNN best checkpoint and
rescored the first 1000 dev requests without training and without reading qrels.
The rerun exactly matched the base score file:

- requests compared: 1000;
- score rows compared: 42,968;
- exact score rows: 42,968;
- `max_abs_score_diff`: 0.0.

## Stage B Comparisons

Best B5o run: `20260709_kuaisearch_b5o_dnn_dev_s20260708`.

| Comparison | Delta NDCG@10 | 95% CI | Interpretation |
|---|---:|---|---|
| vs Random | +0.0277 | [0.0224, 0.0331] | significant sanity pass |
| vs B0b | -0.0051 | [-0.0105, 0.0004] | below recent-behavior; CI crosses 0 |
| vs B7-bge | -0.0217 | [-0.0272, -0.0162] | significantly below B7-best |

## Verdict

B5o is complete for Batch 2b as
`official-code, proxy-aligned (last-time 10% split)`.

Stage A is strong enough to say the locked official ranking code and adapter
match the Table 7 metric scale under the proxy split. Stage B shows that the
official DNN/DCNv2 family is a valid sanity baseline on the PPS standardized
KuaiSearch dev split, but it does not beat the current strongest static
baseline.

The claim is intentionally limited:

- it is not an upstream-confirmed Table 7 split guarantee;
- it must not be reported as a caveat-free upstream split result;
- it consumed 6/16 B5o KuaiSearch dev evaluations;
- its best dev NDCG@10 is 0.3088, with DNN mean 0.3063 +/- 0.0030.

Result implication: B5o is significant over Random, statistically tied or
slightly below B0b, and significantly below B7-bge. This preserves the current
baseline-to-beat as B7-bge and supports the fixed-candidate-pool diagnosis that
query relevance is already mostly saturated by recall.
