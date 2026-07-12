# PPS Baseline Cards

状态：占位清单。每个 baseline 完成接入前，先在这里登记边界；进入论文主表前，
补齐 tuning budget、run ID 和验收结论。

当前阶段：C01--C80 已关闭，没有 validated architecture 或 C81。ordinary
full-token joint Transformer 必须作为 trainable strong baseline 正常调优；future Hxx
只有在 `doc/31` Failure Card 通过后才能登记 architecture trial budget。`doc/24` 和
四份 design prompt 仅是历史记录。

| ID | Batch | Method | Role | Channels | Source | Impl type | Status |
|---|---|---|---|---|---|---|---|
| Random | 0 | Random scorer | sanity check | none | self | self-implemented | first |
| B0a | 1 | Popularity | lower bound | item popularity | self | self-implemented | first |
| B0b | 1 | Recent-behavior | control/lower bound | history | self | self-implemented | first |
| B1 | 1 | BM25 | query-only lexical | query + item text | classical IR | self/Pyserini | first |
| B2z | 1 | Dense bi-encoder zero-shot | query-only semantic | query + item text | bge/gte weights | zero-shot | first |
| B7 | 1 | Static mixture | key control | query score + history score | self | self-implemented | first |
| D1q/m/a | motivation | Frozen-embedding supervised diagnostics | learnability control | query/popularity + optional history | self | train-only calibrated | complete negative |
| D2t | motivation | Fine-tuned BGE query tower | strong query-only control | query + item text | BAAI BGE + self adapter | fine-tuned official weights | complete |
| D2p | motivation | Fine-tuned text + popularity | strong non-personalized control | query + item text + train popularity | self | static train-calibrated | complete |
| D2h | motivation | Fine-tuned text + causal history | interim static waterline | query + item text + history | self | static train-calibrated | complete; superseded by D2s |
| D2s | motivation | D2p + bundled causal history | complete bundled-history reference | query + item text + train popularity + history | self | static train-calibrated | complete; superseded as waterline by C5-R3 item-only |
| C5-R3 item/category | motivation | D2p + decomposed causal history | mechanism ablation / current static waterline | query + item text + train popularity + item or category history | self | fixed removal ablation | complete; `TERMINAL_FAIL` scoped to doc/23 recovery ladder; item-only baseline-to-beat |
| M3 | 1 | Per-request oracle | headroom analysis | baseline scores | self | analysis only | first |
| R1 | 3-pre | Cheap learned router | control, not proposed system | fixed M3 channel scores | self | logistic/tree router | complete |
| B9z | 3-pre | ZAM | query-conditioned nearest neighbor | query + history + item | ProdSearch | official-code minimal adapter | supplementary provisional; human review pending |
| B9t | 3-pre | TEM | transformer PPS nearest neighbor | query + history + item | ProdSearch | official-code minimal adapter | supplementary provisional; human review pending |
| B3 | 2 | Cross-encoder reranker | query-only upper bound | query + item text | bge-reranker or similar | zero-shot | complete |
| B4 | 2 | SASRec/BERT4Rec | strong history baseline placeholder | history sequence | RecBole + original papers | adapter-only | retired placeholder; superseded by B4o |
| B5 | 2 | DIN/DCNv2 | industrial baseline placeholder | full structured features | KuaiSearch official | style adapter | retired placeholder; superseded by B5o |
| B6 | 2 | HEM/ZAM/TEM | PPS classic baseline placeholder | query + history + item text | PPS classic papers/code | style adapter | retired placeholder; superseded by B6o |
| B4o | 2b | RecBole SASRec/BERT4Rec | official strong history baseline | history sequence | RecBole + original papers | official code | complete |
| B5o | 2b | KuaiSearch DIN/DCNv2 | official industrial baseline | full structured features | KuaiSearch official | official-code, proxy-aligned (last-time 10% split) | proxy-aligned; Stage B authorized |
| B6o | 2b | HEM official | PPS classic baseline | query + history + item text | PPS classic papers/code | official code; alignment failed | permanently downgraded |
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

ID: D1q/D1m/D1a
Method: supervised frozen-embedding base and mean/query-attentive history residuals
Implementation type: self-implemented motivation diagnostics
Input fields used: frozen BGE query/item embeddings, train-only popularity, causal history for residual variants
Config path: configs/analysis/supervised_motivation_diagnostics_final.yaml
Run IDs: 20260710_kuaisearch_d1q_supervised_query_dev_s20260708/09/10; 20260710_kuaisearch_d1m_mean_history_residual_dev_s20260708/09/10; 20260710_kuaisearch_d1a_query_attn_residual_dev_s20260708/09/10
Current status: complete negative diagnostic
Acceptance notes: D1q/D1m/D1a three-seed means are 0.3147/0.3145/0.3148. Both residuals have mixed seed directions and paired CIs crossing zero; D1a does not consistently beat D1m. Query-attentive event selection is not an established motivation fact.

ID: D2t/D2p
Method: fine-tuned BGE query tower; static text/popularity control
Implementation type: official pretrained encoder with self-implemented train-only adapter/protocol
Input fields used: D2t uses query and candidate title only; D2p additionally uses train-only item click counts
Config path: configs/analysis/finetuned_nonpersonalized_control_final.yaml
Run IDs: 20260710_kuaisearch_d2t_finetuned_text_dev_s20260708/09/10; 20260710_kuaisearch_d2p_text_pop_dev_s20260708/09/10
Current status: complete
Acceptance notes: D2t mean is 0.3141 and significantly exceeds B2z at the preselected seed. D2p mean is 0.3240 and significantly exceeds D2t/D1q/B0b, but remains significantly below B7. The first internal alpha result was invalidated for using full-train popularity; the corrected internal-train selection alpha is 0.6 and was frozen before dev scoring.

ID: D2h
Method: static fine-tuned text + causal recent-behavior mixture
Implementation type: self-implemented train-only calibrated control
Input fields used: seed-matched D2t score and frozen B0b score; matched wrong B0b for identity controls
Config path: configs/analysis/d2h_static_history_control_final.yaml
Run IDs: 20260710_kuaisearch_d2h_static_true_history_dev_s20260708/09/10; 20260710_kuaisearch_d2h_static_wrong_history_dev_s20260708/09/10
Current status: complete interim control; superseded by D2s
Acceptance notes: train-only alpha=0.1. True D2h mean is 0.3352+/-0.0005 and significantly exceeds B7 by +0.0046 CI [0.0012, 0.0080]. Matched wrong-history mean is 0.3090; true-minus-wrong is significant for every seed on history-present and same-query subsets. D2h and D2t have exact metric equality on all no-history requests.

ID: D2s
Method: static frozen D2p + causal recent-behavior mixture
Implementation type: self-implemented post-result fairness repair, train-only calibrated
Input fields used: seed-matched D2p score and frozen B0b score; matched wrong B0b for identity controls
Config path: configs/analysis/d2s_static_full_control_final.yaml
Run IDs: 20260710_kuaisearch_d2s_static_true_history_dev_s20260708/09/10; 20260710_kuaisearch_d2s_static_wrong_history_dev_s20260708/09/10
Current status: complete bundled-history reference; superseded as the static baseline-to-beat by C5-R3 item-only
Acceptance notes: D2h omitted the popularity term already validated in D2p. Train-only beta=0.3 was frozen under doc 21. True D2s mean is 0.3416+/-0.0004 and significantly exceeds D2h by +0.0064 CI [0.0037, 0.0090]. Matched wrong-history mean is 0.3181; true-minus-wrong remains significant for every seed on history-present and same-query subsets. D2s and D2p have exact metric equality on all no-history requests.

ID: M3
Method: Per-request oracle
Implementation type: analysis only
Input fields used: per-request metrics for B2z, B0b, B7-bge
Run IDs: 20260708_kuaisearch_m3_oracle_dev
Current status: frozen result retained; post-hoc construct validity failed
Acceptance notes: The registered oracle is NDCG@10 = 0.4232 and +28.0% relative over B7-bge. The original pre-C2 copy and protocol-valid reissue remain preserved. A tie audit records 55.97% ties, so 60.6%/35.1%/4.3% are assignments rather than strict preferences. The later Random-channel null reaches 0.4325/+30.9% and Random-oracle labels have M4 AUC 0.6952 versus 0.6688 for the actual labels (`reports/pps_m3_m4_random_canary_audit.json`). M3 permanently remains a failed diagnostic; C3-R replaces the positive claim with a different matched-history construct and does not repair this oracle.

ID: R1
Method: Cheap learned router over M3 channels
Implementation type: self-implemented control, not proposed system
Input fields used: M4 request-level features; fixed channel scores from B2z, B0b, B7-bge
Config path: artifacts/m4/m4_feature_manifest.json
Run IDs: 20260710_kuaisearch_r1b_router_lr_dev; 20260710_kuaisearch_r1a_router_cv_dev
Current status: complete control
Acceptance notes: R1b trains logistic regression on a 20,000-request train subset using records_train labels only and reads qrels_dev only through the shared evaluator. R1b NDCG@10 = 0.3072, significantly below B7-bge by -0.0234 CI [-0.0266, -0.0201], with recovery ratio -0.2521. R1a dev cross-fit NDCG@10 = 0.3106 and is only +1.12% relative over R1b. Low-recovery diagnostic found no deterministic feature/metric mismatch: train and dev oracle label distributions match closely, but LR argmax collapses mostly to query_b2z and never selects the static channel. The initial evaluation triggered doc/16 §5.2; the coupled outputs were regenerated and rechecked once, giving identical aggregate metrics. Both evaluator entries remain logged, and the missing first score snapshot prevents a byte-identity claim (`reports/pps_r1_dev_eval_reconciliation.json`). R1 remains a weak cheap control and must not be treated as the proposed system.
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
Dev evals used: 6/16 tuning + 3 frozen seeds; 8 evaluator entries because the selected seed is also a tuning run
Determinism check: passed on same-seed h128 retrain for first 1000 dev requests; 42968/42968 score rows exact, max_abs_score_diff=0.0; report reports/b4o_determinism_check.json
Run IDs: 20260709_kuaisearch_b4o_sasrec_recbole_dev_s20260708; 20260709_kuaisearch_b4o_sasrec_recbole_t01_len20_dev_s20260708; 20260709_kuaisearch_b4o_sasrec_recbole_t02_drop02_dev_s20260708; 20260709_kuaisearch_b4o_sasrec_recbole_t04_lr0005_dev_s20260708; 20260709_kuaisearch_b4o_sasrec_recbole_t03_h128_dev_s20260708; 20260709_kuaisearch_b4o_sasrec_recbole_t05_l1_dev_s20260708; 20260709_kuaisearch_b4o_sasrec_recbole_h128_dev_s20260709; 20260709_kuaisearch_b4o_sasrec_recbole_h128_dev_s20260710
Known limitations: must run in python 3.10/isolated pps-recbole environment because active python 3.13 cannot install RecBole 1.2.1 dependency ray<=2.6.3. Environment also pins setuptools<81 because ray 2.6.3 imports pkg_resources. Vocab coverage review in reports/b4o_vocab_coverage.md found only 22.2% dev candidate rows in the train-interaction item vocab; B4o keeps the documented cold-last policy rather than adding untrained candidate embeddings.
Current status: complete
Acceptance notes: Step 0 budget amendment is recorded in reports/pps_batch2b_budget_amendment.md. Unified train-interaction artifact is recorded in reports/pps_batch2b_interactions_train_manifest.json. RecBole/ml-100k sanity passed in reports/b4o_env_sanity.md: loss decreased from 336.5817 to 299.0741 over 3 epochs and example-split test NDCG@10 = 0.0358. Vocab coverage exceeded the 30% cold-start review threshold and was reviewed in reports/b4o_vocab_coverage.md. Best official RecBole SASRec run is 20260709_kuaisearch_b4o_sasrec_recbole_t03_h128_dev_s20260708 with NDCG@10=0.2976, MRR=0.2788, Recall@10=0.5169, pNDCG@10=0.3236. It is significantly above Random but significantly below B0b and B7-bge, consistent with a pure item-ID sequential baseline under a query-conditioned fixed candidate pool.

ID: B5o
Method: KuaiSearch official DIN/DCNv2
Role: official industrial ranking baseline
Evidence channels: query + history + item text + candidate item_id + train labels
Source paper/repo: https://github.com/benchen4395/KuaiSearch commit 7ce0471b659112096f0aa7e892ed0aa4c972246a
Venue/year: KuaiSearch dataset/repo
Implementation type: official-code, proxy-aligned (last-time 10% split)
Input fields used: query; frozen history item/category/event/time; candidate text/category/item_id; train clicked/purchased labels
Output score definition: official DIN/DCNv2 ranking score exported for fixed candidates and evaluated only by the shared evaluator
Config path: configs/baselines/b5o_kuaisearch_din_dcnv2.yaml
Environment group: kuaisearch
Tuning budget: 16 KuaiSearch dev evaluations; first run is official/default hyperparameters
Dev evals used: 6/16
Determinism check: passed on frozen DNN checkpoint rescore for first 1000 dev requests; 42968/42968 score rows exact, max_abs_score_diff=0.0; report reports/b5o_determinism_check.json
Run IDs: 20260709_kuaisearch_b5o_dnn_dev_s20260708; 20260709_kuaisearch_b5o_dnn_dev_s20260709; 20260709_kuaisearch_b5o_dnn_dev_s20260710; 20260709_kuaisearch_b5o_dcnv2_dev_s20260708; 20260709_kuaisearch_b5o_dcnv2_dev_s20260709; 20260709_kuaisearch_b5o_dcnv2_dev_s20260710
Known limitations: official ranking code is executable and proxy Stage A passed, but the exact paper last-day split remains unavailable. The accepted full Stage A run uses a last-time 10% proxy by `time_index`, with threshold ties assigned to test, so it can only be claimed as `official-code, proxy-aligned (last-time 10% split)`. Details are in reports/b5o_official_alignment.md, reports/b5o_protocol_diff.md, and doc/baseline_notes/20260709_b5o_stage_a_split_decision.md.
Current status: complete formal dev baseline under proxy-aligned identity
Acceptance notes: Step 0 budget amendment is recorded in reports/pps_batch2b_budget_amendment.md. Stage A evidence is recorded in reports/b5o_official_alignment.md: materializer full proxy target coverage 1.0, official BGE encoding produced 555,553 query embeddings and 6,206,709 item embeddings, and official default DNN/DCNv2 runs landed within +/-10% of Table 7 under the proxy split. DNN final LogLoss/AUC = 0.160731/0.613133; DCNv2 final LogLoss/AUC = 0.162635/0.616348. The low smoke AUC was checked in reports/b5o_smoke_auc_direction_check.md; no score/label reversal was found. Stage B uses the official DNN/DCNv2 model family on PPS standardized data with label-free dev scoring and shared evaluator only. Best formal run is DNN seed 20260708 with NDCG@10=0.3088, MRR=0.2850, Recall@10=0.5334, pNDCG@10=0.3191. DNN three-seed mean NDCG@10=0.3063+/-0.0030; DCNv2 three-seed mean NDCG@10=0.3054+/-0.0002. The best DNN run is significantly above Random (+0.0277 CI [0.0224, 0.0331]), not significantly above B0b (-0.0051 CI [-0.0105, 0.0004]), and significantly below B7-bge (-0.0217 CI [-0.0272, -0.0162]).

ID: B6o
Method: HEM official code, with TEM/ZAM family source retained for fallback
Role: PPS classic baseline
Evidence channels: query + history + item text + candidate item_id + train labels
Source paper/repo: HEM official repo `QingyaoAi/Hierarchical-Embedding-Model-for-Personalized-Product-Search` at `cd089d0ecf277e2fcdccb35f4989d05ef3e81032`; TEM/ProdSearch repo `kepingbi/ProdSearch` at `449335ba652fe7c877a008e154157d7b2a4b0e76`
Venue/year: SIGIR 2017 / CIKM 2019 / SIGIR 2020
Implementation type: official HEM code path; permanently downgraded after failed external Amazon alignment
Input fields used: query; frozen click/purchase history; candidate title/brand/category/item_id; train clicked/purchased labels
Output score definition: PPS classic query-history-item score exported for every fixed candidate and evaluated only by the shared evaluator
Config path: configs/baselines/b6o_pps_classic.yaml
Environment group: pps_classic
Tuning budget: 16 KuaiSearch dev evaluations shared by selected HEM/ZAM/TEM variants; each variant first run uses paper/default hyperparameters
Dev evals used: 0/16
Determinism check: not applicable; no formal KuaiSearch dev run
Run IDs: none
Known limitations: official code requires old TensorFlow and Amazon review fields. The official-code run is end-to-end executable but did not reproduce the Cell Phones & Accessories target within +/-10%; unvalidated reimplementations cannot enter the main table as faithful baselines. The 2026-07-09 limited reconnaissance checked five public external sources, found no public original `query_split/` or checkpoint, and found no deterministic reconstruction bug that justifies another 20-epoch run.
Current status: permanently downgraded: alignment-not-verifiable
Acceptance notes: Step 0 budget amendment is recorded in reports/pps_batch2b_budget_amendment.md. HEM Path 1 evidence is documented in reports/b6o_official_alignment.md: best observed MAP@100 = 0.0759 and best observed NDCG@10 = 0.0932, below the target 0.124/0.153 by more than the +/-10% tolerance. The upstream issue draft is archival only and will not be posted. No KuaiSearch dev evaluation has been produced; B6o exits the formal main-table alignment path.
```

## Architecture-Readiness Neighbor Cards

```text
ID: B9z
Method: ZAM
Role: query-conditioned personalized-product-search nearest neighbor
Evidence channels: request query + frozen history + candidate item identity/text-derived PV training
Source paper/repo: kepingbi/ProdSearch at 449335ba652fe7c877a008e154157d7b2a4b0e76, Apache-2.0
Venue/year: CIKM 2019 family implementation
Implementation type: official-code, adapter to KuaiSearch interface, not externally aligned
Input fields used: request-level query; frozen history truncated to upstream limit 20; candidate item_id; title + brand + category only for clicked train targets
Output score definition: official ZAM score for deterministic 1,500-wide padded candidates, with fillers removed and exact frozen candidates restored before shared evaluation
Config path: configs/baselines/b9_prodsearch.yaml
Environment group: pps-prodsearch (Python 3.13.12, isolated clone of the frozen package set)
Tuning budget: one frozen official/default point, 3 required seeds; no dev-driven grid
Dev evals used: 3 frozen final seeds
Determinism check: passed; seed 20260708 repeat is byte-identical over the complete score file and exact on 42,968 rows from the first 1,000 requests (reports/pps_zam_determinism_check.json)
Run IDs: 20260710_kuaisearch_b9z_zam_r2_dev_s20260708; 20260710_kuaisearch_b9z_zam_r2_dev_s20260709; 20260710_kuaisearch_b9z_zam_r2_dev_s20260710
Known limitations: only 11.71% of unique dev candidates and 22.00% of candidate rows occur as clicked train targets, so most official item-ID/PV candidate embeddings lack target-text training. The three runs resumed from complete epoch 13/13/12 checkpoints after modern-PyTorch dtype and singleton-batch compatibility fixes; model and optimizer state were restored, but upstream checkpoints do not store RNG state, so continuation is not bit-identical to a hypothetical uninterrupted run.
Current status: numerical suite complete; human top-5 reviewer provenance pending
Acceptance notes: Three-seed mean NDCG@10=0.2986+/-0.0006. All seeds are significantly above Random. The highest observed seed (20260710, NDCG@10=0.2994) is significantly below B0b (-0.0145 CI [-0.0198, -0.0091]), B7-bge (-0.0311 CI [-0.0365, -0.0256]), and R1b (-0.0078 CI [-0.0132, -0.0023]). Paper reporting must use the mean, not the highest seed. The existing top-5 decision lacks a reviewer/authorization field, so strict "人工确认" remains pending. Frozen wording branch: not claimably above B7-bge.

ID: B9t
Method: TEM (official item_transformer path)
Role: transformer-based personalized-product-search nearest neighbor
Evidence channels: request query + frozen history + candidate item identity/text-derived PV training
Source paper/repo: kepingbi/ProdSearch at 449335ba652fe7c877a008e154157d7b2a4b0e76, Apache-2.0
Venue/year: SIGIR 2020 family implementation
Implementation type: official-code, adapter to KuaiSearch interface, not externally aligned
Input fields used: request-level query; frozen history truncated to upstream limit 20; candidate item_id; title + brand + category only for clicked train targets
Output score definition: official item_transformer score for deterministic 1,500-wide padded candidates, with fillers removed and exact frozen candidates restored before shared evaluation
Config path: configs/baselines/b9_prodsearch.yaml
Environment group: pps-prodsearch (Python 3.13.12, isolated clone of the frozen package set)
Tuning budget: one frozen official/default point, 3 required seeds; no dev-driven grid
Dev evals used: 3 frozen final seeds
Determinism check: passed; seed 20260708 repeat is byte-identical over the complete score file and exact on 42,968 rows from the first 1,000 requests (reports/pps_tem_determinism_check.json)
Run IDs: 20260710_kuaisearch_b9t_tem_r2_dev_s20260708; 20260710_kuaisearch_b9t_tem_r2_dev_s20260709; 20260710_kuaisearch_b9t_tem_r2_dev_s20260710
Known limitations: only 11.71% of unique dev candidates and 22.00% of candidate rows occur as clicked train targets, so most official item-ID/PV candidate embeddings lack target-text training. This path is not externally aligned to the unrecoverable Amazon benchmark split/checkpoint.
Current status: numerical suite complete; human top-5 reviewer provenance pending
Acceptance notes: Three-seed mean NDCG@10=0.2940+/-0.0009. All seeds are significantly above Random. The highest observed seed (20260710, NDCG@10=0.2948) is significantly below B0b (-0.0191 CI [-0.0245, -0.0138]), B7-bge (-0.0358 CI [-0.0412, -0.0303]), and R1b (-0.0124 CI [-0.0179, -0.0072]). Paper reporting must use the mean, not the highest seed. The existing top-5 decision lacks a reviewer/authorization field, so strict "人工确认" remains pending. Frozen wording branch: not claimably above B7-bge.
```

## C3-R Claim Control Card

```text
ID: C3-R wrong-history
Method: matched wrong-user history for frozen B0b and B7-bge
Role: identity-specificity falsification; not a competitive baseline
Evidence channels: unchanged target query/candidates + earlier train history from a different user
Implementation type: self-implemented deterministic perturbation control
Input fields used: train/dev query, user_id, history, candidate item/category; no qrels during scoring
Output score definition: frozen B0b; frozen B7 alpha=0.2 with wrong-history B0b scores
Config path: configs/analysis/c3_history_identity_controls.yaml
Tuning budget: none; three seeds locked before evaluation
Dev evals used: 6 fixed claim-control evaluations, not tuning
Determinism check: deterministic hash reservoir and donor selection; assignment and score hashes recorded
Run IDs: 20260710_kuaisearch_c3r_b0b_wrong_history_dev_s20260708/09/10; 20260710_kuaisearch_c3r_b7_wrong_history_dev_s20260708/09/10
Known limitations: train-frozen donors are much staler than rolling true dev history; identity interpretation superseded by C5-R2
Current status: historical pass only; no longer authorizes C5-R
Acceptance notes: true-minus-wrong B7 mean +0.0431 on 8,119 history-present requests and +0.0321 on 2,709 same-query requests; all seed CIs lower >0. B7 equals B2z on all 4,110 no-history requests.
```

## C5-R2 Temporal-Symmetric Claim Control Card

```text
ID: C5-R2 temporal wrong-history D2s
Method: freshness-matched different-user train/earlier-dev snapshots with frozen D2s
Role: temporal-symmetric identity-specificity falsification; not a competitive baseline
Evidence channels: unchanged target query/candidates + strictly-prior rolling true or different-user history
Implementation type: self-implemented deterministic prequential perturbation control
Input fields used: standardized train/dev query, user_id, history, candidate item/category; no qrels during materialization/scoring
Output score definition: frozen D2p and beta=0.3; only the B0b history input is replaced
Config path: configs/analysis/c5r_temporal_symmetric_identity.yaml
Tuning budget: none; feasibility used labels-free covariates, then protocol/config/gate frozen before repaired outcome evaluation
Dev evals used: 3 fixed wrong-D2s claim-control evaluations, not tuning
Determinism check: timestamp-group insertion, bounded recent pools, seeded top-k selection; assignment/config/score hashes recorded
Run IDs: 20260710_kuaisearch_c5r2_d2s_temporal_wrong_dev_s20260708/09/10
Known limitations: factor-four per-request age balance is bounded rather than exact; same-query balanced log-age SMD remains about 0.23--0.26
Current status: implementation/integrity passed; scientific C5-R2 failed
Acceptance notes: 7,614 freshness-balanced requests have positive CIs in all seeds. The 1,063 same-query + freshness-balanced requests have mean +0.0095, but only 1/3 seed CIs are positive; the frozen rule required 2/3. This gate does not authorize an identity-specific interpretation; the later C5-R3 synthesis governs the current design-stage boundary.
```

## C5-R3 Candidate-History Component Control Card

```text
ID: C5-R3 item-only / category-only D2s
Method: exact executable decomposition of frozen B0b into item recurrence and deepest-exclusive category affinity
Role: finite motivation falsifier and removal ablation; item-only becomes the current static benchmark
Evidence channels: unchanged D2p query/text/train-popularity base + strictly-prior true history item or category component
Implementation type: self-implemented deterministic analysis control using the shared static-mixture writer and evaluator
Input fields used: records_dev history.item_id/cat/event and candidates.item_id/cat; no qrels during materialization/scoring
Output score definition: beta=0.3 * z(D2p) + 0.7 * z(item_component or category_component), within request
Config path: configs/analysis/c5r3_candidate_history_alignment.yaml
Tuning budget: none; beta, seeds, components, primary, sole fallback, and terminal rule locked before outcomes
Dev evals used: 6 fixed component evaluations, not tuning
Determinism check: 575,609 candidate rows reproduce public and actual upstream B0b within 1e-12; maximum error 7.1054e-15, zero violations
Run IDs: 20260710_kuaisearch_c5r3_d2s_item_only_dev_s20260708/09/10; 20260710_kuaisearch_c5r3_d2s_category_only_dev_s20260708/09/10
Known limitations: exact repeat-item signal is a narrow benchmark mechanism; it does not establish semantic transfer, user-identity causality, or a proposed architecture
Current status: implementation/integrity passed; primary and sole fallback failed; `TERMINAL_FAIL` scoped to the doc/23 item/category recovery ladder
Acceptance notes: item-only mean NDCG@10=0.3453755 and beats D2p on history-present requests in 3/3 seeds (+0.03204/+0.03214/+0.03263, all CIs positive). Category-only has 0/3 significant seeds and 0.1148% mean relative gain. Full D2s is significantly worse than item-only in all seeds. Both component controls exactly match D2p rankings and metrics on 4,110 no-history requests. These outcomes validate neither candidate primitive in doc/23; they show that exact recurrence is reliable in the tested bundle while uncalibrated cross-item/category transfer is not. C01--C80 later closed without a validated architecture; current use is as a static control in doc/31 R0 failure discovery.
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
