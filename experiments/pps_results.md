# PPS Results Registry（唯一结果登记处）

规则（违反任一条，该行无效）：

1. 每个数字只准从对应 `runs/<run_id>/metrics.json` 复制，禁止手算或"记忆值"；
2. 一行 = 一个 (method, split, run_id)；复跑用新行，不覆盖旧行；
3. 显著性列由共享 compare 脚本产出（paired bootstrap 95% CI，doc 11 §1.4），
   参照方法必须注明；
4. dev 行随 Batch 进度追加；**test 行只在 Phase 5 冻结 config 后一次性填写**；
5. 列结构冻结，不许为某个方法增删列；方法特有信息写 baseline card，不写这里。

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
| B7-bge | `20260708_kuaisearch_b7_bge_dev_a02` | 0.3305 | 0.3141 | 0.5418 | 0.3469 (0.1401) | vs B0b: +0.0166, [0.0121, 0.0211] | 20260708 | 11/11 | complete |
| M3 oracle | `20260708_kuaisearch_m3_oracle_dev` | 0.4232 | — | — | — | headroom vs B7-bge: +28.0% rel.; CI [+27.2%, +28.9%] | 20260708 | — | protocol-valid after C2 reissue |
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
| M3 Batch2 oracle | `20260708_kuaisearch_m3_batch2_oracle_dev` | 0.5468 | — | — | — | headroom vs B7-bge: +65.4% rel.; CI [+64.2%, +66.7%] | 20260708 | — | analysis only; multi-channel oracle, selection-noise caveat |

## Test（Phase 5 冻结后一次性填写）

| Method | Run ID | NDCG@10 | MRR | Recall@10 | pNDCG@10 (cov.) | Seeds (mean±std) | 状态 |
|---|---|---|---|---|---|---|---|

## M3 Headroom 摘要

| 项 | 值 | 判据 |
|---|---|---|
| oracle NDCG@10 | 0.4232 | protocol-valid after C2 reissue |
| 最强单方法 NDCG@10 | 0.3305 (`static_b7_bge`) | B7-bm25 final-B1 = 0.3292 |
| headroom（relative） | +28.0% | ≥ +5% |
| bootstrap 95% CI 下界 | +27.2% relative | ≥ +2% |
| split-half（两半各自 headroom） | +28.2%, +27.9% | 同向且都 ≥ +2% |
| oracle 通道选择分布 | B2z 60.6%, B0b 35.1%, B7-bge 4.3% | 无单通道 >90% |

## M3 Batch 2 Oracle 摘要

| 项 | 值 | 判据 |
|---|---|---|
| oracle NDCG@10 | 0.5468 | analysis only; 多通道 per-request max |
| 最强单方法 NDCG@10 | 0.3305 (`static_b7_bge`) | B8a h=50 full-dev = 0.3302 |
| headroom（relative） | +65.4% | ≥ +5% |
| bootstrap 95% CI 下界 | +64.2% relative | ≥ +2% |
| split-half（两半各自 headroom） | +65.8%, +65.1% | 同向且都 ≥ +2% |
| oracle 通道选择分布 | B2z 29.8%, B0b 20.8%, B3 16.8%, B4 12.2%, B5 8.1%, B6 7.2%, B7 2.5%, B8 variants 2.8% total | 无单通道 >90% |
| caveat | 方法数增多会抬高 per-request oracle，不能当作可部署增益 | 仅证明异质性和 headroom 仍存在 |
