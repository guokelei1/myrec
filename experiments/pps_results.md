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
| B1 | `20260708_kuaisearch_b1_bm25_globalidf_exact10_dev` | 0.3054 | 0.2801 | 0.5294 | 0.3280 (0.1401) | vs B0a: +0.0041, [-0.0012, 0.0098] | 20260708 | 8/9 | C2 failed |
| B2z | `20260708_kuaisearch_b2z_bge_small_zh_dev` | 0.3056 | 0.2823 | 0.5264 | 0.3198 (0.1401) | vs B1: +0.0002, [-0.0041, 0.0045] | 20260708 | 1/1 | complete |
| B7-bm25 | `20260708_kuaisearch_b7_bm25_dev_a01` | 0.3276 | 0.3088 | 0.5412 | 0.3483 (0.1401) | vs B0b: +0.0137, [0.0093, 0.0181] | 20260708 | 11/11 | complete; uses earlier B1 score run |
| B7-bge | `20260708_kuaisearch_b7_bge_dev_a02` | 0.3305 | 0.3141 | 0.5418 | 0.3469 (0.1401) | vs B0b: +0.0166, [0.0121, 0.0211] | 20260708 | 11/11 | complete |
| M3 oracle | `20260708_kuaisearch_m3_oracle_dev` | 0.4232 | — | — | — | headroom vs B7-bge: +28.0% rel.; CI [+27.2%, +28.9%] | 20260708 | — | exploratory; C2 blocked |
| B3 | | | | | | vs B2z | | /16 | deferred |
| B4 | | | | | | vs B0b | | /16 | deferred |
| B5 | | | | | | vs B7-best | | /16 | deferred |
| B6 | | | | | | vs B7-best | | /16 | deferred |
| B8a | | | | | | vs B7-best（同抽样子集） | | /3 | deferred |
| B8b | | | | | | vs B8a（同抽样子集） | | /3 | deferred |

## Test（Phase 5 冻结后一次性填写）

| Method | Run ID | NDCG@10 | MRR | Recall@10 | pNDCG@10 (cov.) | Seeds (mean±std) | 状态 |
|---|---|---|---|---|---|---|---|

## M3 Headroom 摘要

| 项 | 值 | 判据 |
|---|---|---|
| oracle NDCG@10 | 0.4232 | exploratory; C2 not passed |
| 最强单方法 NDCG@10 | 0.3305 (`static_b7_bge`) | |
| headroom（relative） | +28.0% | ≥ +5% |
| bootstrap 95% CI 下界 | +27.2% relative | ≥ +2% |
| split-half（两半各自 headroom） | +28.2%, +27.9% | 同向且都 ≥ +2% |
| oracle 通道选择分布 | B2z 60.6%, B0b 35.1%, B7-bge 4.3% | 无单通道 >90% |
