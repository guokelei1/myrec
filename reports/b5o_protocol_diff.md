# B5o KuaiSearch Official Protocol Diff

Date: 2026-07-09

Scope: Batch 2b B5o, locked upstream commit
`7ce0471b659112096f0aa7e892ed0aa4c972246a`.

Status: downgrade report. B5o is `official-code, alignment-not-verifiable`;
no KuaiSearch dev run was produced.

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
| Split | sample field `split == "test"` else train, then random 10% valid | Paper says last day is test; public file contains `split`, but exact paper split cannot be verified from code alone |
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

## Required Decision Before Any Formal B5o Run

Proceeding to a KuaiSearch dev run would require a scoped adapter, not a clean
official reproduction. The adapter would need to:

- materialize official-format `rank.jsonl`, `corpus.jsonl`, and `users.jsonl`
  from train-only standardized or raw train data;
- explicitly map `age_bucket` to `age` or default missing demographics;
- write embeddings to the exact paths consumed by `ranking/datasets.py`;
- preserve frozen per-request history at inference;
- emit per-fixed-candidate `scores.jsonl` for the shared PPS evaluator.

Until that decision is authorized, B5o remains a downgraded secondary baseline
candidate and not a formal official-aligned main-table run.
