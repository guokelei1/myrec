# PPS Results Registry（唯一结果登记处）

规则（违反任一条，该行无效）：

1. 每个数字只准从对应 `runs/<run_id>/metrics.json` 复制，禁止手算或"记忆值"；
2. 一行 = 一个 (method, split, run_id)；复跑用新行，不覆盖旧行；
3. 显著性列由共享 compare 脚本产出（paired bootstrap 95% CI，doc 11 §1.4），
   参照方法必须注明；
4. dev 行随 Batch 进度追加；**test 行只在 Phase 5 冻结 config 后一次性填写**；
5. 列结构冻结，不许为某个方法增删列；方法特有信息写 baseline card，不写这里。
6. 可训练方法的论文主结果使用 3-seed mean +/- variability（doc/07 §11）；表中
   单个 run 行只用于逐 run 追溯，不能把最高 seed 当作论文 headline。

## Dev（KuaiSearch 主轨）

| Method | Run ID | NDCG@10 | MRR | Recall@10 | pNDCG@10 (cov.) | vs. 参照（Δ, 95% CI） | Seeds | Dev evals 用量/预算 | 状态 |
|---|---|---|---|---|---|---|---|---|---|
| Random | `20260708_kuaisearch_random_c1` | 0.2811 | 0.2583 | 0.5011 | 0.2998 (0.1401) | — | 20260708 | instrumentation | complete |
| B0a | `20260708_kuaisearch_b0a_popularity_dev` | 0.3013 | 0.2796 | 0.5216 | 0.3252 (0.1401) | vs Random: +0.0202, [0.0149, 0.0255] | 20260708 | 1/9 | complete |
| B0b | `20260708_kuaisearch_b0b_recent_behavior_dev` | 0.3139 | 0.2983 | 0.5268 | 0.3416 (0.1401) | vs Random: +0.0328, [0.0274, 0.0383] | 20260708 | 1/9 | complete |
| B1 | `20260708_kuaisearch_b1_bm25_globalidf_exact10_dev` | 0.3054 | 0.2801 | 0.5294 | 0.3280 (0.1401) | vs B0a: +0.0041, [-0.0012, 0.0098] | 20260708 | 8/9 | accepted under revised C2; original dominance failure retained |
| B2z | `20260708_kuaisearch_b2z_bge_small_zh_dev` | 0.3056 | 0.2823 | 0.5264 | 0.3198 (0.1401) | vs B1: +0.0002, [-0.0041, 0.0045] | 20260708 | 1/1 | complete |
| B7-bm25 | `20260708_kuaisearch_b7_bm25_dev_a01` | 0.3276 | 0.3088 | 0.5412 | 0.3483 (0.1401) | vs B0b: +0.0137, [0.0093, 0.0181] | 20260708 | 11/11 | retired; used earlier B1 score run |
| B7-bm25 | `20260708_kuaisearch_b7_bm25_finalb1_dev_a01` | 0.3292 | 0.3105 | 0.5438 | 0.3510 (0.1401) | vs B0b: +0.0153, [0.0109, 0.0198] | 20260708 | 11/11 replacement | complete; uses final active B1 |
| B7-bge | `20260708_kuaisearch_b7_bge_dev_a02` | 0.3305 | 0.3141 | 0.5418 | 0.3469 (0.1401) | vs B0b: +0.0166, [0.0121, 0.0211]; vs B2z: +0.0249, [0.0216, 0.0282] | 20260708 | 11/11 | complete |
| D1q | `20260710_kuaisearch_d1q_supervised_query_dev_s20260708` | 0.3146 | 0.2911 | 0.5370 | 0.3333 (0.1401) | vs B2z: +0.0089, [0.0041, 0.0136]; vs B7: -0.0159, [-0.0210, -0.0109] | 20260708/09/10; mean 0.3147+/-0.0007 | 3 fixed diagnostics | complete; frozen-embedding supervised base |
| D1m | `20260710_kuaisearch_d1m_mean_history_residual_dev_s20260708` | 0.3151 | 0.2922 | 0.5368 | 0.3346 (0.1401) | vs seed-matched D1q: +0.0005, [-0.0009, 0.0020] | 20260708/09/10; mean 0.3145+/-0.0005 | 3 fixed diagnostics | complete negative; directions +,-,+ |
| D1a | `20260710_kuaisearch_d1a_query_attn_residual_dev_s20260708` | 0.3151 | 0.2917 | 0.5374 | 0.3335 (0.1401) | vs seed-matched D1q: +0.0005, [-0.0000, 0.0011]; vs D1m: -0.0000, [-0.0014, 0.0014] | 20260708/09/10; mean 0.3148+/-0.0004 | 3 fixed diagnostics | complete negative; query attention not established |
| D1a wrong history | `20260710_kuaisearch_d1a_wrong_history_dev_s20260708` | 0.3146 | 0.2914 | 0.5364 | 0.3339 (0.1401) | true minus wrong on history-present: +0.0008, [0.0003, 0.0013] at seed08; not stable all seeds | 20260708/09/10; mean 0.3146+/-0.0007 | 3 fixed rescoring controls | complete; one-seed effect only |
| D2t | `20260710_kuaisearch_d2t_finetuned_text_dev_s20260708` | 0.3140 | 0.2914 | 0.5356 | 0.3314 (0.1401) | vs B2z: +0.0083, [0.0044, 0.0123]; vs D1q: -0.0006, [-0.0051, 0.0039] | 20260708/09/10; mean 0.3141+/-0.0002 | 3 fixed diagnostics | complete; fully fine-tuned query tower |
| D2p | `20260710_kuaisearch_d2p_text_pop_dev_s20260708` | 0.3238 | 0.3018 | 0.5459 | 0.3388 (0.1401) | vs D2t: +0.0098, [0.0068, 0.0129]; vs B7: -0.0067, [-0.0117, -0.0017] | 20260708/09/10; mean 0.3240+/-0.0002 | 3 fixed diagnostics | complete; non-personalized text + train popularity |
| D2h | `20260710_kuaisearch_d2h_static_true_history_dev_s20260708` | 0.3352 | 0.3186 | 0.5474 | 0.3524 (0.1401) | vs B7: +0.0046, [0.0012, 0.0080]; vs D2p: +0.0113, [0.0072, 0.0155] | 20260708/09/10; mean 0.3352+/-0.0005 | 3 fixed controls | complete interim; superseded by D2s because D2h omits popularity |
| D2h wrong history | `20260710_kuaisearch_d2h_static_wrong_history_dev_s20260708` | 0.3095 | 0.2853 | 0.5312 | 0.3282 (0.1401) | true minus wrong: history-present +0.0387 [0.0335, 0.0440]; same-query +0.0289 [0.0206, 0.0372] | 20260708/09/10; mean 0.3090+/-0.0004 | 3 fixed controls | complete; matched train donor |
| D2s | `20260710_kuaisearch_d2s_static_true_history_dev_s20260708` | 0.3415 | 0.3266 | 0.5530 | 0.3590 (0.1401) | vs D2h: +0.0064, [0.0037, 0.0090]; vs D2p: +0.0177, [0.0147, 0.0207] | 20260708/09/10; mean 0.3416+/-0.0004 | 3 fixed controls | complete full-history reference; superseded as waterline by C5-R3 item-only |
| D2s wrong history | `20260710_kuaisearch_d2s_static_wrong_history_dev_s20260708` | 0.3179 | 0.2952 | 0.5395 | 0.3377 (0.1401) | true minus wrong: history-present +0.0356 [0.0308, 0.0404]; same-query +0.0277 [0.0199, 0.0355] | 20260708/09/10; mean 0.3181+/-0.0005 | 3 fixed controls | historical; identity interpretation superseded by C5-R2 temporal audit |
| C5-R2 temporal wrong D2s | `20260710_kuaisearch_c5r2_d2s_temporal_wrong_dev_s20260708` | 0.3169 | 0.2931 | 0.5401 | 0.3355 (0.1401) | true minus wrong: freshness-balanced +0.0374 [0.0324, 0.0424]; same-query balanced +0.0092 [-0.0006, 0.0191] | 20260708/09/10; mean 0.3172+/-0.0003 | 3 fixed claim controls | complete; integrity passed; C5-R2 failed (same-query significant 1/3) |
| **C5-R3 item-only D2s** | `20260710_kuaisearch_c5r3_d2s_item_only_dev_s20260708` | **0.3451** | **0.3306** | **0.5575** | **0.3596 (0.1401)** | history-present vs D2p: +0.0320, [0.0283, 0.0357]; full D2s minus item-only: -0.0054, [-0.0081, -0.0026] | 20260708/09/10; mean 0.3454+/-0.0003 | 3 fixed component controls | **complete; current static baseline-to-beat; diagnostic exact-repeat signal** |
| C5-R3 category-only D2s | `20260710_kuaisearch_c5r3_d2s_category_only_dev_s20260708` | 0.3242 | 0.3017 | 0.5453 | 0.3426 (0.1401) | history-present vs D2p: +0.0006, [-0.0029, 0.0040] | 20260708/09/10; mean 0.3242+/-0.0003 | 3 fixed component controls | complete negative; 0/3 significant, C5-R3 fallback failed |
| C3-R wrong B0b | `20260710_kuaisearch_c3r_b0b_wrong_history_dev_s20260708` | 0.2811 | 0.2557 | 0.5041 | 0.3092 (0.1401) | true B0b minus wrong on history-present: +0.0494, [0.0441, 0.0546] | 20260708 | fixed claim control | complete; train-only matched donor |
| C3-R wrong B0b | `20260710_kuaisearch_c3r_b0b_wrong_history_dev_s20260709` | 0.2803 | 0.2560 | 0.5013 | 0.3097 (0.1401) | true B0b minus wrong on history-present: +0.0506, [0.0453, 0.0558] | 20260709 | fixed claim control | complete; train-only matched donor |
| C3-R wrong B0b | `20260710_kuaisearch_c3r_b0b_wrong_history_dev_s20260710` | 0.2813 | 0.2558 | 0.5043 | 0.3086 (0.1401) | true B0b minus wrong on history-present: +0.0491, [0.0437, 0.0543] | 20260710 | fixed claim control | complete; train-only matched donor |
| C3-R wrong B7 | `20260710_kuaisearch_c3r_b7_wrong_history_dev_s20260708` | 0.3022 | 0.2778 | 0.5233 | 0.3184 (0.1401) | true B7 minus wrong: history-present +0.0427 [0.0375, 0.0478]; same-query +0.0309 [0.0227, 0.0394] | 20260708 | fixed claim control | complete; frozen alpha 0.2 |
| C3-R wrong B7 | `20260710_kuaisearch_c3r_b7_wrong_history_dev_s20260709` | 0.3017 | 0.2785 | 0.5210 | 0.3168 (0.1401) | true B7 minus wrong: history-present +0.0434 [0.0382, 0.0485]; same-query +0.0335 [0.0256, 0.0418] | 20260709 | fixed claim control | complete; frozen alpha 0.2 |
| C3-R wrong B7 | `20260710_kuaisearch_c3r_b7_wrong_history_dev_s20260710` | 0.3018 | 0.2782 | 0.5221 | 0.3170 (0.1401) | true B7 minus wrong: history-present +0.0432 [0.0380, 0.0484]; same-query +0.0319 [0.0237, 0.0401] | 20260710 | fixed claim control | complete; frozen alpha 0.2 |
| M3 oracle | `20260708_kuaisearch_m3_oracle_dev` | 0.4232 | — | — | — | registered headroom +28.0%; Random-canary oracle is higher at +30.9% | 20260708 | — | frozen result retained; construct validity failed post hoc |
| R1b | `20260710_kuaisearch_r1b_router_lr_dev` | 0.3072 | 0.2845 | 0.5263 | 0.3211 (0.1401) | vs B7-bge: -0.0234, [-0.0266, -0.0201]; recovery ratio -0.2521 | 20260708 | 2 invocations: initial + §5.2 recheck | complete; cheap router, low-recovery diagnostic clean |
| R1a | `20260710_kuaisearch_r1a_router_cv_dev` | 0.3106 | 0.2889 | 0.5277 | 0.3262 (0.1401) | robustness only; +1.12% rel. vs R1b | 20260708 | 2 coupled invocations: initial + §5.2 recheck | complete; dev cross-fit reference, not registered R1 number |
| B9z | `20260710_kuaisearch_b9z_zam_r2_dev_s20260710` | 0.2994 | 0.2791 | 0.5177 | 0.3209 (0.1401) | highest seed vs B7-bge: -0.0311, [-0.0365, -0.0256] | 20260708/09/10; mean 0.2986+/-0.0006 | 3 frozen seeds | supplementary provisional trace; human review provenance pending |
| B9t | `20260710_kuaisearch_b9t_tem_r2_dev_s20260710` | 0.2948 | 0.2729 | 0.5163 | 0.3195 (0.1401) | highest seed vs B7-bge: -0.0358, [-0.0412, -0.0303] | 20260708/09/10; mean 0.2940+/-0.0009 | 3 frozen seeds | supplementary provisional trace; human review provenance pending |
| B3 | `20260708_kuaisearch_b3_bge_reranker_base_zs_dev` | 0.3068 | 0.2819 | 0.5275 | 0.3217 (0.1401) | vs B2z: +0.0011, [-0.0031, 0.0053] | zero-shot | 1/1 | complete; not significant over B2z |
| B4 | `20260708_kuaisearch_b4_sasrec_style_hashed_prior_dev_s20260709` | 0.2887 | 0.2673 | 0.5101 | 0.2973 (0.1401) | vs B0b: -0.0252, [-0.0306, -0.0197]; sanity vs Random: +0.0076, [0.0025, 0.0128] | 20260708/09/10; mean 0.2881±0.0007 | 7/16 | retired placeholder; superseded by B4o |
| B4o | `20260709_kuaisearch_b4o_sasrec_recbole_t03_h128_dev_s20260708` | 0.2976 | 0.2788 | 0.5169 | 0.3236 (0.1401) | vs Random: +0.0165, [0.0113, 0.0217]; vs B0b: -0.0163, [-0.0201, -0.0125]; vs B7-bge: -0.0329, [-0.0382, -0.0276] | 20260708/09/10; mean 0.2972±0.0004 | 6/16 tuning + 3 seeds | complete; official RecBole SASRec; high cold-start caveat |
| B5 | `20260708_kuaisearch_b5_dcn_din_style_hashed_dev_s20260709` | 0.2931 | 0.2716 | 0.5147 | 0.2975 (0.1401) | vs B7-best: -0.0375, [-0.0430, -0.0317] | 20260708/09/10; mean 0.2922±0.0011 | 3/16 | retired placeholder; superseded by B5o |
| B5o | `20260709_kuaisearch_b5o_dnn_dev_s20260708` | 0.3088 | 0.2850 | 0.5334 | 0.3191 (0.1401) | vs Random: +0.0277, [0.0224, 0.0331]; vs B0b: -0.0051, [-0.0105, 0.0004]; vs B7-bge: -0.0217, [-0.0272, -0.0162] | 20260708/09/10; DNN mean 0.3063+/-0.0030; DCNv2 mean 0.3054+/-0.0002 | 6/16 | complete; official-code, proxy-aligned (last-time 10% split) |
| B6 | `20260708_kuaisearch_b6_pps_classic_style_hashed_dev_s20260709` | 0.2933 | 0.2704 | 0.5171 | 0.3078 (0.1401) | vs B7-best: -0.0373, [-0.0429, -0.0316] | 20260708/09/10; mean 0.2929±0.0003 | 3/16 | retired placeholder; superseded by B6o |
| B8a | `20260708_kuaisearch_b8a_qwen25_7b_h5_dev` | 0.3294 | 0.3129 | 0.5407 | 0.3468 (0.1401) | vs B7-best on 2000-request subset: -0.0069, [-0.0139, 0.0002] | zero-shot | 1/3 | complete; raw-history h=5 |
| B8a | `20260708_kuaisearch_b8a_qwen25_7b_h20_dev` | 0.3301 | 0.3133 | 0.5421 | 0.3460 (0.1401) | vs B7-best on 2000-request subset: -0.0024, [-0.0093, 0.0045] | zero-shot | 2/3 | complete; raw-history h=20 |
| B8a | `20260708_kuaisearch_b8a_qwen25_7b_h50_dev` | 0.3302 | 0.3134 | 0.5422 | 0.3463 (0.1401) | vs B7-best on 2000-request subset: -0.0019, [-0.0089, 0.0050] | zero-shot | 3/3 | complete; raw-history h=50 |
| B8b | `20260708_kuaisearch_b8b_qwen25_7b_h5_dev` | 0.3293 | 0.3135 | 0.5407 | 0.3474 (0.1401) | vs B8a h=5 on same subset: -0.0005, [-0.0069, 0.0058] | zero-shot | 1/3 | complete; memory-style h=5 |
| B8b | `20260708_kuaisearch_b8b_qwen25_7b_h20_dev` | 0.3293 | 0.3135 | 0.5407 | 0.3486 (0.1401) | vs B8a h=20 on same subset: -0.0052, [-0.0120, 0.0016] | zero-shot | 2/3 | complete; memory-style h=20 |
| B8b | `20260708_kuaisearch_b8b_qwen25_7b_h50_dev` | 0.3293 | 0.3137 | 0.5406 | 0.3477 (0.1401) | vs B8a h=50 on same subset: -0.0053, [-0.0120, 0.0014] | zero-shot | 3/3 | complete; memory-style h=50 |
| M3 Batch2 oracle | `20260708_kuaisearch_m3_batch2_oracle_dev` | 0.5468 | — | — | — | registered headroom +65.4% rel.; selection-noise dominated | 20260708 | — | failed diagnostic only; not heterogeneity evidence |

## Test（Phase 5 冻结后一次性填写）

| Method | Run ID | NDCG@10 | MRR | Recall@10 | pNDCG@10 (cov.) | Seeds (mean±std) | 状态 |
|---|---|---|---|---|---|---|---|

## M3 Headroom 摘要

| 项 | 值 | 判据 |
|---|---|---|
| oracle NDCG@10 | 0.4232 | historical threshold result; construct validity failed |
| 最强单方法 NDCG@10（历史 M3 输入集） | 0.3305 (`static_b7_bge`) | excludes later D2h/D2s; B7-bm25 final-B1 = 0.3292 |
| headroom（relative） | +28.0% | ≥ +5% |
| bootstrap 95% CI 下界 | +27.2% relative | ≥ +2% |
| split-half（两半各自 headroom） | +28.2%, +27.9% | 同向且都 ≥ +2% |
| oracle tie-broken assignment | B2z 60.6%, B0b 35.1%, B7-bge 4.3% | assignment，不是严格偏好；冻结 tie order |
| tie-aware winner audit | tie 55.97%; unique B0b 28.86%, B2z 10.86%, B7-bge 4.32% | static 严格低于至少一个替代通道 40.14% |

## M3 Batch 2 Oracle 摘要

| 项 | 值 | 判据 |
|---|---|---|
| oracle NDCG@10 | 0.5468 | analysis only; 多通道 per-request max |
| 最强单方法 NDCG@10（历史 Batch 2 oracle 输入集） | 0.3305 (`static_b7_bge`) | excludes later D2h/D2s; B8a h=50 full-dev = 0.3302 |
| headroom（relative） | +65.4% | ≥ +5% |
| bootstrap 95% CI 下界 | +64.2% relative | ≥ +2% |
| split-half（两半各自 headroom） | +65.8%, +65.1% | 同向且都 ≥ +2% |
| oracle 通道选择分布 | B2z 29.8%, B0b 20.8%, B3 16.8%, B4 12.2%, B5 8.1%, B6 7.2%, B7 2.5%, B8 variants 2.8% total | 无单通道 >90% |
| caveat | 方法数增多会抬高 per-request oracle；Random null 已复现该效应 | 失败诊断，不证明异质性或 headroom |

## C3 Motivation Gate 摘要

| 项 | 值 | 判据 / 状态 |
|---|---|---|
| M4 predictability AUC | 0.6688 macro OvR, 5-fold LR | passed; 阈值 >= 0.65 |
| M4 label-shuffle canary | 0.4955 macro OvR | passed; 期望 0.50 +/- 0.02 |
| M4 intentional-leak canary | 0.8331 macro OvR, +0.1643 vs formal | passed; 泄漏列注入后应明显暴涨 |
| M4 oracle-label tie rate | 55.97% | reported; tie rule follows M3 channel order |
| M5 entropy slice E1 | high entropy query-to-oracle gap 0.1156 vs low entropy 0.1177 | failed; frozen direction expected high > low |
| M5 overlap slice E2 | low-overlap B0b mean 0.2904 vs high-overlap 0.3709; low gap 0.1211 vs high gap 0.0994 | passed |
| C3 adjudication | frozen gate failed, historical final status passed | historical record only; positive claim replaced by C3-R |
| Post-hoc entropy diagnostics | query-in-train coverage 42.5%; Spearman rho(entropy, oracle-query) -0.0110 | exploratory only; Consensus Law warning, not headline evidence |

## Post-Hoc Construct-Validity Audit

This read-only audit does not alter any frozen result. It determines whether the
M3/M4 statistics measure the construct claimed in the motivation:

| Check | Actual channel | Random-channel null | Decision |
|---|---:|---:|---|
| Three-channel oracle NDCG@10 | 0.4232 (+28.0% vs B7) | 0.4325 (+30.9%) | failed: Random is higher |
| M4 5-fold macro OvR AUC | 0.6688 | 0.6952 | failed: Random labels are more predictable |
| Oracle headroom on 4,110 no-history requests | +27.7% | +28.4% | failed: nominal history wins without history evidence |

Source: `reports/pps_m3_m4_random_canary_audit.json`. M3 headroom, M4
predictability, and selected M3 slices remain unusable as positive evidence.
They were not repaired or reinterpreted; C3-R replaces the positive claim with
a different, matched-history construct.

## C3-R / C5-R Historical Motivation, C5-R2, and C5-R3 Scoped Terminal Gate

| Check | Result | Decision |
|---|---:|---|
| Aggregate complementarity | B7 vs B0b +0.0166; B7 vs B2z +0.0249; both CIs > 0 | passed |
| Wrong B7, three-seed full-dev mean | 0.3019 +/- 0.0002 | below true B7 0.3305 |
| True minus wrong B7, history-present | mean +0.0431; conservative CI envelope [+0.0375, +0.0485] | passed all seeds |
| True minus wrong B7, same-query donors | 2,709 requests; mean +0.0321; envelope [+0.0227, +0.0418] | passed all seeds |
| No-history equivalence | B7 and B2z identical on 4,110/4,110 requests | passed |
| Historical revised insight | query-anchored personalized residual | superseded: old donor control was temporally asymmetric |
| C5-R2 freshness-balanced | 7,614 requests; deltas +0.0374/+0.0379/+0.0362; all CIs positive | aggregate correct-history value survives |
| C5-R2 same-query + freshness-balanced | 1,063 requests; mean +0.0095; only 1/3 CIs positive | failed frozen 2/3 rule; identity-specific interpretation not authorized |
| C5-R3 item-only vs D2p | 8,119 requests; +0.03204/+0.03214/+0.03263; all CIs positive | stable exact-repeat candidate signal |
| C5-R3 category-only vs D2p | +0.00059/+0.00053/-0.00003; all CIs cross zero | no independent coarse-semantic gain |
| C5-R3 full D2s vs item-only | -0.00538/-0.00521/-0.00634; all intervals negative | category component weakens item-only in all seeds |
| C5-R3 primary / sole fallback | failed / failed; integrity passed | `TERMINAL_FAIL` for the doc/23 item/category recovery ladder; neither candidate primitive is validated |
| Current stage | exact recurrence is reliable in the tested bundle; uncalibrated cross-item/category transfer is not | motivation complete; architecture/protocol formulation authorized; implementation/training requires a new design-specific pre-outcome falsifier |

Sources: historical `doc/17_intro_motivation_repair_protocol.md` and
`reports/pps_c3r_history_identity_control.json`; `doc/22` and
`reports/pps_c5r2_temporal_symmetric_identity.json`; current `doc/23` and
`reports/pps_c5r3_candidate_history_alignment.json`.

## D1/D2/D2h/D2s Motivation Strengthening

| Check | Result | Decision |
|---|---:|---|
| D1 train-fitted residuals | D1m 0.3145; D1a 0.3148; neither stably exceeds D1q 0.3147 | query-attentive event use not established |
| Fine-tuned text | D2t 0.3141 mean; seed08 vs B2z +0.0083 [0.0044, 0.0123] | supervised text signal is real but bounded |
| Non-personalized control | D2p 0.3240 mean; seed08 vs B7 -0.0067 [-0.0117, -0.0017] | text + train popularity remains below history mix |
| Interim static waterline | D2h 0.3352+/-0.0005; vs B7 +0.0046 [0.0012, 0.0080] | valid but incomplete because it omits D2p popularity |
| D2h identity control | true-minus-wrong history-present mean +0.0396; same-query mean +0.0315; all CIs positive | historical train-frozen donor result; identity interpretation superseded |
| D2h no-history boundary | D2h and seed-matched D2t have exact NDCG/MRR/Recall equality on 4,110 requests | passed all seeds |
| Complete bundled-history reference | D2s 0.3416+/-0.0004; vs D2h +0.0064 [0.0037, 0.0090] | valid full-history reference, no longer strongest static control |
| D2s identity control | true-minus-wrong history-present mean +0.0354; same-query mean +0.0276; all CIs positive | historical train-frozen donor result; identity interpretation superseded |
| D2s no-history boundary | D2s and seed-matched D2p have exact NDCG/MRR/Recall equality on 4,110 requests | passed all seeds |
| Current static waterline | C5-R3 item-only 0.3454+/-0.0003; history-present vs D2p significant in 3/3 | exact-repeat control replaces D2s as baseline-to-beat |
| Category component | category-only 0.3242+/-0.0003; 0/3 significant vs D2p; full is worse than item-only in 3/3 | semantic category alignment not established; finite C5-R3 gate terminated |
| Design interpretation | exact recurrence survives while the uncalibrated category transfer dilutes it | formulate an evidence-fidelity-aware architecture/protocol; no new mechanism is yet validated, and implementation/training remains gated |

Sources: `doc/18_supervised_motivation_diagnostics_protocol.md`, `doc/19_finetuned_nonpersonalized_control_protocol.md`,
`doc/20_d2h_static_history_waterline_protocol.md`, `doc/23_c5r3_candidate_history_alignment_protocol.md`,
`reports/pps_supervised_diagnostics_summary.json`, `reports/pps_d2_d2h_summary.json`, and
`reports/pps_c5r3_candidate_history_alignment.json`.
