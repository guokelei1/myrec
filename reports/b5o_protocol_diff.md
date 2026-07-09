# B5o KuaiSearch Official Protocol Diff

Date: 2026-07-09

Scope: Batch 2b B5o, locked upstream commit
`7ce0471b659112096f0aa7e892ed0aa4c972246a`.

Status: Stage A passed under the authorized last-time proxy split. The exact
paper split remains unverified and no KuaiSearch dev run has been produced.

## Official Ranking Pipeline Requirements

| Requirement | Official code path | Current evidence |
|---|---|---|
| Ranking samples | `ranking/datasets.py` reads `data/rank.jsonl` | Public raw layout is `data/raw/kuaisearch/rank_lite/train.jsonl` |
| User features | `data/users.jsonl` with `user_id`, `gender`, `age` | Public rows use `age_bucket`; no `age` field |
| Item corpus | `data/corpus.jsonl` with `item_id`, `item_title`, category ids | Public raw layout is `items_lite/train.jsonl`; field shape is compatible after renaming to `corpus.jsonl` |
| Query embeddings | `./data/query_emb.npy` + `./data/session_id2idx.json` | `ranking/data/process.py` writes these files to `./`, not `./data/` |
| Item title embeddings | `./data/item_title_emb.npy` + `./data/item_id2idx.json` | Same path mismatch as query embeddings |
| History | `recently_clicked_item_ids` only, max 20 | Standardized records have frozen history click/purchase events; official path ignores purchased history |
| Label | `is_clicked == 1 or is_purchased == 1` | Compatible with raw ranking rows; formal PPS dev labels remain forbidden |
| Split | sample field `split == "test"` else train, then random 10% valid | Local public `rank_lite/train.jsonl` has no test rows; the authorized proxy uses last 10% by `time_index`, with threshold ties assigned to test; the exact Table 7 split boundary is still unverified |
| Metrics | test LogLoss and ROC-AUC | Paper Table 7 provides Logloss/ROC-AUC targets |

## Fairness-Matrix Mapping For Any Future Adapter

| Official feature | Allowed standardized source | B5o decision |
|---|---|---|
| Query text embedding | `records_*.query` encoded by frozen public text encoder | Allowed only after external alignment issue is resolved or downgraded scope is explicit |
| Target title embedding | candidate/item catalog title text encoded by the same frozen encoder | Allowed, cache under `artifacts/batch2b/` with hash |
| User id | `record.user_id` | Allowed |
| User gender/age | Not present in standardized records | Must use missing/default bucket; this is a protocol loss relative to official code |
| Recent clicked history | frozen record history click items, max 20/50 as config states | Allowed; must not use global user sequence |
| Recent purchased history | frozen record purchased history | Official code ignores this unless adapter is extended; extension would be structural and must be declared |
| Target category ids | standardized item/category fields | Allowed if already present in candidate/catalog record |
| Raw user statistics | Not present in standardized records | Forbidden unless first promoted into standardized records from train-only data |
| Target statistical features | Not used by current official model code despite raw rank rows containing them | No adapter dependency |
| Dev/test labels | `qrels_dev`, `qrels_test`, candidate clicked/purchased labels | Forbidden for training/scoring |

## Blocking Differences

1. **Repository path mismatch**: official preprocessing writes embeddings to
   root, while official training reads embeddings from `./data`.
2. **User schema mismatch**: public `age_bucket` is not read by the model
   loader's `age` field.
3. **Fallback index bug**: missing users map to `gender=2`, `age=9`, but the
   declared embedding cardinalities are 2 and 7.
4. **Demo data mismatch**: demo rank targets are not covered by demo item ids,
   so the demo cannot be used as-is for official ranking training.
5. **Encoder mismatch**: paper says BERT-derived embeddings; released code uses
   `BAAI/bge-small-zh-v1.5`.

## Stage A Repair Attempt: Official-Format Materializer

New adapter source:

- `src/myrec/baselines/kuaisearch_materializer.py`
- test: `tests/test_kuaisearch_materializer.py`

The materializer writes the files expected by `ranking/datasets.py` under an
official-code working root:

| Official file | Public source | Materializer decision |
|---|---|---|
| `data/rank.jsonl` | `rank_lite/train.jsonl` | Copy ranking rows, set `split` by explicit policy, preserve official labels for Stage A only |
| `data/corpus.jsonl` | `items_lite/train.jsonl` | Load all target and clicked-history item ids needed by the materialized rank rows |
| `data/users.jsonl` | `users_lite/train.jsonl` | Map `age_bucket -> age`; preserve legal `gender`; synthesize legal default rows for missing users |
| `data/query_emb.npy`, `data/session_id2idx.json` | official `ranking/data/process.py` over `rank.jsonl` | Run official encoder, then place outputs in `./data/` where the loader reads them |
| `data/item_title_emb.npy`, `data/item_id2idx.json` | official `ranking/data/process.py` over `corpus.jsonl` | Same placement decision; no source patch required for the path mismatch |

Age mapping:

| Public `age_bucket` | Official `age` |
|---|---|
| `0-11` | `0-11` |
| `12-17` | `12-17` |
| `18-23` | `18-23` |
| `24-30` | `24-30` |
| `31-40` | `31-40` |
| `41-49` | `41-49` |
| `50+` | `50+` |

Missing or invalid user features use `gender=M`, `age=31-40` so the official
fallback indices `gender=2` and `age=9` are never reached.

Smoke artifacts:

- `artifacts/batch2b/b5o_materializer_smoke/materializer_manifest.json`
- `reports/b5o_smoke_auc_direction_check.md`
- 2000 ranking rows
- target item coverage: 1997/1997 unique target ids, rate 1.0
- users: 9/9 matched, 0 synthetic missing users
- official BGE process completed with `BAAI/bge-small-zh-v1.5`
- official DNN 1-epoch smoke completed: test LogLoss 0.643912, AUC 0.377851
- the low smoke AUC was manually checked: official AUC equals manual AUC, and
  the materializer preserves the raw click/purchase label fields

This smoke validates the official loader/trainer path only. It is not a Table 7
alignment run because the subset is tiny and uses a provisional split.

## Authorized Proxy Split For Full Stage A

The materializer can produce full official-format data. The paper Table 7 split
remains unverified: the local public ranking file carries `split=train` for all
checked rows, while the official loader requires `split=test` rows.

Authorized proxy for the bounded Stage A run:

- policy: `last_time_fraction`
- test fraction: `0.10`
- time field: `time_index`
- tie handling: `time_index >= test_time_min` is assigned to test, so the
  actual test fraction may exceed exactly 10%

Conclusion rule:

- proxy +/-10% alignment can only be reported as
  `official-code, proxy-aligned (last-time 10% split)`, with the split caveat;
- proxy >10% difference leaves B5o as
  `official-code, alignment-not-verifiable`, with no tuning loop.

Decision note:
`doc/baseline_notes/20260709_b5o_stage_a_split_decision.md`.

## Proxy Stage A Outcome

Full proxy artifact root:
`artifacts/batch2b/b5o_proxy_lasttime_full`.

The full materializer passed with target coverage 1.0, 17,800,904 ranking rows,
6,206,709 corpus items, and a proxy test split of 1,780,145 rows
(`time_index >= 867165`, actual fraction 0.100003).

Official BGE encoding generated `(555553, 512)` query embeddings and
`(6206709, 512)` item-title embeddings with the locked repo encoder
`BAAI/bge-small-zh-v1.5`.

The official default DNN and DCNv2 runs both landed within +/-10% of the paper
Table 7 metric scale under this proxy split:

| Method | Proxy LogLoss | Proxy AUC | Table 7 Logloss | Table 7 AUC | Verdict |
|---|---:|---:|---:|---:|---|
| DNN | 0.160731 | 0.613133 | 0.1588 | 0.6258 | proxy-aligned |
| DCNv2 | 0.162635 | 0.616348 | 0.1603 | 0.6239 | proxy-aligned |

This does not remove the split caveat. The correct status is
`official-code, proxy-aligned (last-time 10% split)`.

## Stage B Adapter Protocol

Authorized identity: `official-code, proxy-aligned (last-time 10% split)`.
This is a scoped PPS adapter under the Stage A split caveat, not a caveat-free
official baseline claim.

Implementation boundary:

- adapter source: `src/myrec/baselines/kuaisearch_official_adapter.py`;
- runner: `scripts/run_b5o_kuaisearch_official.py`;
- artifact root: `artifacts/batch2b/b5o_stageb_standardized`;
- official source patch: none planned; if ever required, write
  `reports/b5o_patch.diff` before running.

Field mapping:

| Official field | PPS source / rule |
|---|---|
| `rank.jsonl.session_id` | `request_id`, used as the official query embedding key |
| `original_session_id` | standardized `session_id`, retained for audit only |
| `query` | `records_train/dev.query` |
| `target_item_id` | fixed candidate `item_id` |
| `is_clicked`, `is_purchased` | train candidate labels only; dev score rows use dummy zeros and are not evaluated by official metrics |
| `recently_clicked_item_ids` | current record frozen click history, <=50 before official loader truncates to 20 |
| `recently_purchased_item_ids` | current record frozen purchase history retained in JSONL; locked loader ignores it |
| `users.jsonl.gender` | default `M`, because standardized records expose no official demographic field |
| `users.jsonl.age` | default `31-40`, a legal official bucket |
| category ids | deterministic text-category mapping into official embedding ranges |
| query/title embeddings | `BAAI/bge-small-zh-v1.5`, CLS pooling, max length 32, float16 cache under artifact root |

Training and scoring:

- training `rank.jsonl` contains train candidate rows only, with labels from
  `records_train.candidates.clicked/purchased`;
- the shared train-interaction artifact remains the provenance record for
  train-only user behavior:
  `artifacts/batch2b/interactions_train.jsonl`
  (`dcc91391c303e6e8c69506a837620be2601bdd6df249cda624f4d1caf417ad32`);
- official training keeps the locked trainer/model classes and official default
  hyperparameters for the first run;
- dev rows are materialized separately as `score_dev.jsonl`; they are never put
  into official `test_loader` because official test evaluation requires labels;
- dev scoring is a label-free forward pass through the trained official model,
  writing full fixed-candidate coverage to `runs/<run_id>/scores.jsonl`;
- only the shared PPS evaluator reads `qrels_dev.jsonl` after scores exist.

Budget and search space:

- budget: 16 KuaiSearch dev evaluations for B5o;
- this prompt authorizes DNN and DCNv2 with official defaults plus frozen seeds
  `[20260708, 20260709, 20260710]`;
- no additional hyperparameter grid is authorized in this run.
