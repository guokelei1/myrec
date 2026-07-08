# PPS Baseline Cards

状态：占位清单。每个 baseline 完成接入前，先在这里登记边界；进入论文主表前，
补齐 tuning budget、run ID 和验收结论。

| ID | Batch | Method | Role | Channels | Source | Impl type | Status |
|---|---|---|---|---|---|---|---|
| Random | 0 | Random scorer | sanity check | none | self | self-implemented | first |
| B0a | 1 | Popularity | lower bound | item popularity | self | self-implemented | first |
| B0b | 1 | Recent-behavior | control/lower bound | history | self | self-implemented | first |
| B1 | 1 | BM25 | query-only lexical | query + item text | classical IR | self/Pyserini | first |
| B2z | 1 | Dense bi-encoder zero-shot | query-only semantic | query + item text | bge/gte weights | zero-shot | first |
| B7 | 1 | Static mixture | key control | query score + history score | self | self-implemented | first |
| M3 | 1 | Per-request oracle | headroom analysis | baseline scores | self | analysis only | first |
| B3 | 2 | Cross-encoder reranker | query-only upper bound | query + item text | bge-reranker or similar | zero-shot/fine-tuned | deferred |
| B4 | 2 | SASRec/BERT4Rec | strong history baseline | history sequence | RecBole + original papers | adapter-only | deferred |
| B5 | 2 | DIN/DCNv2 | official industrial baseline | full structured features | KuaiSearch official | official code + adapter | deferred |
| B6 | 2 | HEM/ZAM/TEM | PPS classic baseline | query + history + item text | PPS classic papers/code | official/reimplementation TBD | deferred |
| B6+ | 2 | MAI/NAM-style | recent PPS/when-personalize baseline | query + history + item text | recent PPS papers | feasibility TBD | candidate |
| B8a | 2 | Raw-history LLM rerank | quality/cost upper bound | query + history + candidates | Qwen or similar | prompt baseline | deferred |
| B8b | 2 | MemRerank-style memory rerank | quality/cost upper bound | query + memory + candidates | MemRerank-style | style-adapted | deferred |

## Batch 1 Run Cards

```text
ID: Random
Method: deterministic random scorer
Implementation type: self-implemented instrumentation
Input fields used: request_id, candidate_item_id
Config path: configs/baselines/random.yaml
Run IDs: 20260708_kuaisearch_random_c1
Current status: complete
Acceptance notes: C1 random sanity run; NDCG@10 = 0.2811.

ID: B0a
Method: Popularity
Implementation type: self-implemented
Input fields used: train candidate clicked labels; dev candidate item_id only
Config path: configs/baselines/b0a_popularity.yaml
Run IDs: 20260708_kuaisearch_b0a_popularity_dev
Current status: complete
Acceptance notes: significant over Random; stats artifact hash recorded in run metadata.

ID: B0b
Method: Recent-behavior
Implementation type: self-implemented
Input fields used: history item/category/event; candidate item/category
Config path: configs/baselines/b0b_recent_behavior.yaml
Run IDs: 20260708_kuaisearch_b0b_recent_behavior_dev
Current status: complete
Acceptance notes: significant over Random; no query fields used.

ID: B1
Method: BM25
Implementation type: self-implemented
Input fields used: query; candidate title/brand/seller/category
Config path: configs/baselines/b1_bm25.yaml
Run IDs: 20260708_kuaisearch_b1_bm25_dev; 20260708_kuaisearch_b1_bm25_cjk23_dev; 20260708_kuaisearch_b1_bm25_exact_dev; 20260708_kuaisearch_b1_bm25_jieba2_dev; 20260708_kuaisearch_b1_bm25_globalidf_dev; 20260708_kuaisearch_b1_bm25_globalidf_exact10_dev; 20260708_kuaisearch_b1_bm25_globalidf_cov10_dev
Current status: accepted under approved revised C2 gate
Acceptance notes: best variant NDCG@10 = 0.3054 and original B1-vs-B0a dominance rule failed; retained as a KuaiSearch candidate-pool dataset property. Revised sanity suite passed: shuffled-query canary, candidate-pool query conditioning, relevance-table lexical signal, B0a train-only audit, shared template/coverage checks, and confirmed top-5 review with four documented lexical limitation classes.

ID: B2z
Method: Dense bi-encoder zero-shot
Implementation type: zero-shot sentence-transformers
Input fields used: query; same candidate document template as B1
Config path: configs/baselines/b2z_dense_biencoder.yaml
Run IDs: 20260708_kuaisearch_b2z_bge_small_zh_dev
Current status: complete
Acceptance notes: BAAI/bge-small-zh-v1.5 on cuda:0; not significant over active B1.

ID: B7-bm25
Method: Static mixture, BM25 + recent-behavior
Implementation type: self-implemented analysis of upstream scores
Input fields used: B1 scores; B0b scores
Config path: configs/baselines/b7_bm25.yaml
Run IDs: 20260708_kuaisearch_b7_bm25_dev_a00..a10; 20260708_kuaisearch_b7_bm25_finalb1_dev_a00..a10
Current status: complete
Acceptance notes: original 20260708_kuaisearch_b7_bm25_dev_a00..a10 grid is retired because it used an earlier B1 score run. Formal Batch 1 line is 20260708_kuaisearch_b7_bm25_finalb1_dev_a01, best alpha = 0.1, NDCG@10 = 0.3292, significantly above B0b by +0.0153 CI [0.0109, 0.0198].

ID: B7-bge
Method: Static mixture, dense bi-encoder + recent-behavior
Implementation type: self-implemented analysis of upstream scores
Input fields used: B2z scores; B0b scores
Config path: configs/baselines/b7_bge.yaml
Run IDs: 20260708_kuaisearch_b7_bge_dev_a00..a10
Current status: complete
Acceptance notes: best alpha = 0.2; best static method in Batch 1.

ID: M3
Method: Per-request oracle
Implementation type: analysis only
Input fields used: per-request metrics for B2z, B0b, B7-bge
Run IDs: 20260708_kuaisearch_m3_oracle_dev
Current status: protocol-valid after C2 reissue
Acceptance notes: M3 gate passes with oracle NDCG@10 = 0.4232 and +28.0% relative headroom over B7-bge. The original pre-C2 copy is preserved at reports/pps_m3_headroom_summary_exploratory_pre_c2.json; the active report is reissued as protocol-valid because M3 is read-only over unchanged per-request metric inputs.
```

## Card Template

```text
ID:
Method:
Role:
Evidence channels:
Source paper/repo:
Venue/year:
Implementation type:
Input fields used:            # 必须与 doc 13 §2.4 公平性矩阵该行一致
Output score definition:
Config path:
Environment group:
Tuning budget:                # doc 13 §2.5 的额度
Dev evals used:               # 与 reports/dev_eval_log.jsonl 对账
Determinism check:            # doc 12 §5 复跑一致性结果
Run IDs:
Known limitations:
Current status:
Acceptance notes:             # doc 13 §3 对应小节验收项逐条结论
```
