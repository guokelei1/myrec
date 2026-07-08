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
| B3 | 2 | Cross-encoder reranker | query-only upper bound | query + item text | bge-reranker or similar | zero-shot | complete |
| B4 | 2 | SASRec/BERT4Rec | strong history baseline placeholder | history sequence | RecBole + original papers | adapter-only | retired placeholder; superseded by B4o |
| B5 | 2 | DIN/DCNv2 | industrial baseline placeholder | full structured features | KuaiSearch official | style adapter | retired placeholder; superseded by B5o |
| B6 | 2 | HEM/ZAM/TEM | PPS classic baseline placeholder | query + history + item text | PPS classic papers/code | style adapter | retired placeholder; superseded by B6o |
| B4o | 2b | RecBole SASRec/BERT4Rec | official strong history baseline | history sequence | RecBole + original papers | official code | in progress |
| B5o | 2b | KuaiSearch DIN/DCNv2 | official industrial baseline | full structured features | KuaiSearch official | official code | in progress |
| B6o | 2b | HEM/ZAM/TEM | PPS classic baseline | query + history + item text | PPS classic papers/code | official/faithful TBD | in progress |
| B6+ | 2 | MAI/NAM-style | recent PPS/when-personalize baseline | query + history + item text | recent PPS papers | feasibility TBD | candidate |
| B8a | 2 | Raw-history LLM rerank | quality/cost upper bound | query + history + candidates | Qwen or similar | prompt baseline | complete |
| B8b | 2 | MemRerank-style memory rerank | quality/cost upper bound | query + memory + candidates | MemRerank-style | style-adapted | complete |

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

## Batch 2 Run Cards

```text
ID: B3
Method: Cross-encoder reranker
Role: query-only semantic upper-bound check
Evidence channels: query + item text
Source paper/repo: BAAI/bge-reranker-base official weights
Implementation type: zero-shot
Input fields used: query; candidate title/brand/seller/category; candidate item_id for keys
Output score definition: cross-encoder score for (query, B1 document text)
Config path: configs/baselines/b3_cross_encoder.yaml
Environment group: gpu, cuda:0 A40
Tuning budget: zero-shot, 1 declared config
Dev evals used: 1/1
Determinism check: deterministic inference config; no sampling
Run IDs: 20260708_kuaisearch_b3_bge_reranker_base_zs_dev
Known limitations: full-dev scoring took 499.8s for 575,609 pairs; no fine-tuning run yet.
Current status: complete
Acceptance notes: NDCG@10 = 0.3068; vs B2z delta +0.0011 CI [-0.0031, 0.0053], not significant.

ID: B4
Method: SASRec-style hashed sequence adapter
Role: strong history baseline placeholder
Evidence channels: history item_id sequence + candidate item_id
Source paper/repo: SASRec/BERT4Rec target; RecBole requested by protocol
Implementation type: self-contained adapter, not RecBole
Input fields used: history.item_id; candidate.item_id; train clicked labels
Output score definition: online logistic item-id transition ranker plus train-only item-bias prior
Config path: configs/baselines/b4_sasrec_style_hashed.yaml
Environment group: core CPU
Tuning budget: trainable, 16 dev evals
Dev evals used: 7/16, including two failed weak variants retained in dev_eval_log
Determinism check: 3 seeds: 20260708/20260709/20260710; mean NDCG@10 = 0.2881, std = 0.0007
Run IDs: 20260708_kuaisearch_b4_sasrec_style_hashed_prior_dev_s20260708; 20260708_kuaisearch_b4_sasrec_style_hashed_prior_dev_s20260709; 20260708_kuaisearch_b4_sasrec_style_hashed_prior_dev_s20260710
Known limitations: RecBole 1.2.1 is not installable in the active Python 3.13 environment because ray<=2.6.3 has no cp313 wheel; this is not an official RecBole run.
Current status: retired placeholder (superseded by B4o)
Acceptance notes: B4 sanity vs Random passes for best seed: +0.0076 CI [0.0025, 0.0128]. It is significantly below B0b by -0.0252 CI [-0.0306, -0.0197]. This adapter remains appendix/implementation evidence only and must not support main-table claims about official SASRec/BERT4Rec strength.

ID: B5
Method: KuaiSearch DCN/DIN-style hashed adapter
Role: industrial CTR/ranking baseline placeholder
Evidence channels: full standardized fields
Source paper/repo: https://github.com/benchen4395/KuaiSearch commit 7ce0471b659112096f0aa7e892ed0aa4c972246a
Implementation type: self-contained adapter, not official code execution
Input fields used: user_id, query, history item/category, candidate item text/category, train clicked labels
Output score definition: online logistic CTR ranker over user, query, item text/category, and history overlap features
Config path: configs/baselines/b5_kuaisearch_dcn_din_style_hashed.yaml
Environment group: core CPU
Tuning budget: trainable, 16 dev evals
Dev evals used: 3/16
Determinism check: 3 seeds: 20260708/20260709/20260710; mean NDCG@10 = 0.2922, std = 0.0011
Run IDs: 20260708_kuaisearch_b5_dcn_din_style_hashed_dev_s20260708; 20260708_kuaisearch_b5_dcn_din_style_hashed_dev_s20260709; 20260708_kuaisearch_b5_dcn_din_style_hashed_dev_s20260710
Known limitations: official ranking code expects precomputed query/title embeddings and raw user feature files outside the standardized blind-record interface; ±10% official alignment is not complete.
Current status: retired placeholder (superseded by B5o)
Acceptance notes: best seed NDCG@10 = 0.2931; significantly below B7-best by -0.0375 CI [-0.0430, -0.0317]. This adapter remains appendix/implementation evidence only and must not support main-table claims about official KuaiSearch DIN/DCNv2 strength.

ID: B6
Method: PPS-classic style hashed query-history fusion adapter
Role: PPS classic baseline placeholder
Evidence channels: query + history + item text
Source paper/repo: HEM/ZAM/TEM target methods; no official adapter present
Implementation type: style-adapted local implementation
Input fields used: query, history item/title/category, candidate item text/category, train clicked labels
Output score definition: online logistic PPS-style ranker over query-document, history-document, and gated personalization overlap features
Config path: configs/baselines/b6_pps_classic_style_hashed.yaml
Environment group: core CPU
Tuning budget: trainable, 16 dev evals
Dev evals used: 3/16
Determinism check: 3 seeds: 20260708/20260709/20260710; mean NDCG@10 = 0.2929, std = 0.0003
Run IDs: 20260708_kuaisearch_b6_pps_classic_style_hashed_dev_s20260708; 20260708_kuaisearch_b6_pps_classic_style_hashed_dev_s20260709; 20260708_kuaisearch_b6_pps_classic_style_hashed_dev_s20260710
Known limitations: not an official HEM/ZAM/TEM reproduction; text-overlap feature construction is slow.
Current status: retired placeholder (superseded by B6o)
Acceptance notes: best seed NDCG@10 = 0.2933; significantly below B7-best by -0.0373 CI [-0.0429, -0.0316]. This adapter remains appendix/implementation evidence only and must not support main-table claims about official or externally validated HEM/ZAM/TEM strength.

ID: B8a/B8b
Method: LLM raw-history and memory-style top-20 rerank
Role: quality/cost upper bound
Evidence channels: query + history + item text + base top-20
Source paper/repo: Qwen/Qwen2.5-7B-Instruct; MemRerank-style for B8b
Implementation type: zero-shot / style-adapted
Input fields used: query, history title/category/event, candidate text/category, B7-bge scores for top-20 truncation and fallback
Output score definition: LLM reranks B7-bge top-20 on fixed 2000-request dev subset; non-subset requests preserve B7-bge scores
Config path: configs/baselines/b8a_llm_rerank.yaml; configs/baselines/b8b_llm_rerank.yaml
Environment group: gpu
Tuning budget: 3 history lengths per variant
Dev evals used: B8a 3/3; B8b 3/3
Determinism check: subset fixed at reports/b8_dev_subset_request_ids_seed20260708.txt
Run IDs: 20260708_kuaisearch_b8a_qwen25_7b_h5_dev; 20260708_kuaisearch_b8a_qwen25_7b_h20_dev; 20260708_kuaisearch_b8a_qwen25_7b_h50_dev; 20260708_kuaisearch_b8b_qwen25_7b_h5_dev; 20260708_kuaisearch_b8b_qwen25_7b_h20_dev; 20260708_kuaisearch_b8b_qwen25_7b_h50_dev
Known limitations: Qwen2.5-7B required manual resume for the last shard; all runs only rerank the fixed 2000-request subset and use B7-bge fallback outside that subset. Full-dev metrics are therefore mostly base-run scores plus subset changes; B8 comparisons use same-subset reports.
Current status: complete
Acceptance notes: B8a best full-dev NDCG@10 = 0.3302 (h=50), B8b best full-dev NDCG@10 = 0.3293 (h=50). On the fixed subset, B8a h=50 vs B7-bge delta = -0.0019 CI [-0.0089, 0.0050]; B8b h=50 vs B8a h=50 delta = -0.0053 CI [-0.0120, 0.0014]. Parse failure rates are <=0.15%.
```

## Batch 2b Official Baseline Cards

```text
ID: B4o
Method: RecBole SASRec/BERT4Rec
Role: official strong history baseline
Evidence channels: history item_id sequence + candidate item_id
Source paper/repo: RecBole 1.2.1; SASRec/BERT4Rec original papers
Venue/year: ICDM 2018 / CIKM 2019
Implementation type: official code
Input fields used: history.item_id; candidate.item_id; train interactions from records_train only
Output score definition: RecBole next-item candidate score mapped to every fixed candidate; out-of-vocab candidates use frozen cold-start margin
Config path: configs/baselines/b4o_sasrec_recbole.yaml
Environment group: recbole
Tuning budget: 16 KuaiSearch dev evaluations; first run is RecBole official/default SASRec config; BERT4Rec, if attempted, shares this pool
Dev evals used: 0/16
Determinism check: pending
Run IDs: pending
Known limitations: must run in python 3.10/isolated pps-recbole environment because active python 3.13 cannot install RecBole 1.2.1 dependency ray<=2.6.3. Environment also pins setuptools<81 because ray 2.6.3 imports pkg_resources.
Current status: in progress
Acceptance notes: Step 0 budget amendment is recorded in reports/pps_batch2b_budget_amendment.md. Unified train-interaction artifact is recorded in reports/pps_batch2b_interactions_train_manifest.json. RecBole/ml-100k sanity passed in reports/b4o_env_sanity.md: loss decreased from 336.5817 to 299.0741 over 3 epochs and example-split test NDCG@10 = 0.0358. Next required work is the RecBole atomic data adapter and fixed-candidate scoring adapter.

ID: B5o
Method: KuaiSearch official DIN/DCNv2
Role: official industrial ranking baseline
Evidence channels: query + history + item text + candidate item_id + train labels
Source paper/repo: https://github.com/benchen4395/KuaiSearch commit 7ce0471b659112096f0aa7e892ed0aa4c972246a
Venue/year: KuaiSearch dataset/repo
Implementation type: official code, alignment pending
Input fields used: query; frozen history item/category/event/time; candidate text/category/item_id; train clicked/purchased labels
Output score definition: official DIN/DCNv2 ranking score exported for fixed candidates and evaluated only by the shared evaluator
Config path: configs/baselines/b5o_kuaisearch_din_dcnv2.yaml
Environment group: kuaisearch
Tuning budget: 16 KuaiSearch dev evaluations; first run is official/default hyperparameters
Dev evals used: 0/16
Determinism check: pending
Run IDs: pending
Known limitations: official pipeline may require precomputed query/title embeddings and raw user features; missing standardized fields must be defaulted and listed in reports/b5o_protocol_diff.md. If official alignment cannot be verified, downgrade per doc 14.
Current status: in progress
Acceptance notes: Step 0 budget amendment is recorded in reports/pps_batch2b_budget_amendment.md. B5o starts only after B4o and B6o have completed or reached their documented downgrade/blocking decisions.

ID: B6o
Method: HEM/ZAM/TEM official or externally validated faithful reproduction
Role: PPS classic baseline
Evidence channels: query + history + item text + candidate item_id + train labels
Source paper/repo: HEM / ZAM / TEM official code or faithful reimplementation with Amazon PPS benchmark validation
Venue/year: SIGIR 2017 / CIKM 2019 / SIGIR 2020
Implementation type: official/faithful TBD
Input fields used: query; frozen click/purchase history; candidate title/brand/category/item_id; train clicked/purchased labels
Output score definition: PPS classic query-history-item score exported for every fixed candidate and evaluated only by the shared evaluator
Config path: configs/baselines/b6o_pps_classic.yaml
Environment group: pps_classic
Tuning budget: 16 KuaiSearch dev evaluations shared by selected HEM/ZAM/TEM variants; each variant first run uses paper/default hyperparameters
Dev evals used: 0/16
Determinism check: pending
Run IDs: pending
Known limitations: official code may require old TensorFlow and Amazon review fields. Any review-field mismatch must be documented; unvalidated reimplementations cannot enter the main table as faithful baselines.
Current status: in progress
Acceptance notes: Step 0 budget amendment is recorded in reports/pps_batch2b_budget_amendment.md. B6o proceeds after B4o, before B5o, per doc 14 priority.
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
