# 16 - Historical C3/Router/Neighbor Execution Plan

> **Current supersession (2026-07-13).** 本文是历史计划，不能作为 next-round
> 指令。C01--C80 已关闭；当前只执行
> [`doc/31`](31_problem_discovery_and_architecture_iteration_protocol.md) 的 R0
> strong-baseline 与 Failure Card discovery。

> **Terminal supersession / 当前解释（2026-07-11）.** 下文的
> benchmark-only/no-design 表述是当时或该特定 gate 的结论。当前以
> [`doc/15_proposed_system_design_principles.md`](../doc/15_proposed_system_design_principles.md)
> 和 [`reports/pps_architecture_readiness.md`](../reports/pps_architecture_readiness.md)
>、[`terminal closure`](dev_log/20260711_architecture_exploration_terminal_closure.md)
> 为准：motivation complete；后续 C01--C16 已关闭，未得到经过验证的架构
> primitive，也未授权 proposed-system dev/full/test evaluation。C5-R3 FAIL
> 及全部数字不变。本文其余
> “当前”“终局”“开工”措辞均是 2026-07-09/早期 2026-07-10 的历史执行记录。

状态：历史执行计划。Step 1–4 已执行；其 readiness 结论后来被
`reports/pps_m3_m4_random_canary_audit.json` 暂停，不再是系统实现的开工授权。
后续 `doc/17` 的 C3-R/C5-R matched-history 修复曾记录为通过，但其时间不对称
解释已被 `doc/22` 的 C5-R2 supersede；C5-R2 未过 same-query gate，在该阶段
没有重新授权设计。这也不恢复本文的 M4/router 逻辑。
最终 `doc/23` C5-R3 又完成了 item/category component gate：item-only 成为
0.3453755 静态水线，category-only 0/3 显著，primary/fallback 均失败。当时的
gate-local 终局被记为 benchmark/analysis-only；本文本身不提供现阶段开工授权。
日期：2026-07-09。

后续强化说明（2026-07-10）：`doc/18`--`doc/21` 已完成 supervised
D1/D2/D2h/D2s controls。完整静态对照 D2s three-seed mean 0.3416 显著高于
中间对照 D2h 0.3352；D2s 后又被 C5-R3 item-only 0.3454 超过。本文 §5.2/Step 4 中所有
“B7=0.3305 为当前门槛”的句子仅是历史记录，不再约束 proposed-system 验收。

执行结果：2026-07-10 完成 Step 1–4。历史 C3 记录为通过（含冻结 M5-E1
负结果与授权重写）、R1b=0.3072、B9 ZAM/TEM 三 seed 均值为
0.2986/0.2940。后续 Random null 使 M3/M4 的构念有效性失败；见
`reports/pps_architecture_readiness.md`。本轮未产出架构提案。

定位：本轮**不做任何新系统设计**。目标是把架构设计开工前缺失的三个数字补齐，
并回填 motivation，使 `doc/15` 自检清单中依赖证据的条目全部有据可查。三个数字：

1. **M4 可预测性 AUC**（C3 gate 的最后一块，doc/11 §2.3 已规定，尚未执行）；
2. **R1 廉价 learned router 的 dev NDCG@10**（未来提议系统"必须打过的数字"，
   doc/15 §5）；
3. **B9 最近邻 query-conditioned baseline 的 dev NDCG@10**（ZAM/TEM 类，
   补 motivation 目前最大的空档："已有 query-conditioned 尝试为什么也不行"，
   同时满足 doc/15 §9 的最近邻实跑要求）。

三个数字齐了之后产出一份 readiness 备忘，架构设计才允许开工（Step 4）。

前置规范（全部沿用，不重复）：`doc/07`（论文约束）、`doc/11`（C0–C5 gate 与
M1–M6 定义）、`doc/12`（执行协议）、`doc/15`（架构设计原则）。冲突时以本文
显式条款为准，其余一律从旧规范。

---

## 0. 环境与通用纪律（每一步都适用）

### 0.1 环境

| 项 | 值 |
|---|---|
| 仓库根目录 | `/data/gkl/myrec`（所有相对路径以此为根） |
| 数据接口 | `data/standardized/kuaisearch/v0_lite/`：`records_{train,dev}.jsonl`、`qrels_dev.jsonl`、`candidate_manifest.json`、`item_catalog.jsonl` |
| 共享评测 | `scripts/evaluate_scores.py`（唯一合法指标来源）+ `scripts/compare_runs.py`（唯一合法显著性检验，paired bootstrap 10000 次） |
| run 目录约定 | `runs/<YYYYMMDD>_kuaisearch_<id>_dev/`，内含 `scores.jsonl`、`metrics.json`、`per_request_metrics.jsonl`、`config_snapshot.yaml`、`metadata.json`（含 git commit 与 seed） |
| 全局 seed | 20260708（抽样/CV/bootstrap 一律用它；多 seed 时用 20260708/20260709/20260710） |
| Python 环境 | 复用现有主管道环境；Step 3 的 ProdSearch 新建 `pps-prodsearch` 环境并冻结到 `configs/env/pps_prodsearch.txt` |

### 0.2 纪律（违反任一条 = 该步产物作废）

1. **test 全程上锁**：`records_test.jsonl` / `qrels_test.jsonl` 禁止被本轮任何
   代码读取（含"只是看一眼行数"）。
2. **qrels 隔离**：`qrels_dev.jsonl` 只允许共享 evaluator 和 M 系列分析脚本读。
   任何特征构造、router 训练、baseline 打分代码打开它 = 作废。train 标签在
   `records_train.jsonl` 内，训练代码可读（本来就是训练标签）。
3. **候选集 hash assert**：每个新 run 评测前对照 `candidate_manifest.json`。
4. **判据先于结果**：本文 §5 的所有阈值在开跑前已冻结，跑完只对答案。
5. **所有指标出自同一份 evaluator**；所有"显著"出自 `compare_runs.py`；
   minimal claimable effect 沿用 dev NDCG@10 **+2% relative**，低于按平手报。
6. 每步落盘 JSON/MD 报告到 `reports/`，负结果同样入库。
7. 结果登记到 `experiments/pps_results.md`，方法边界卡登记到
   `experiments/pps_baseline_cards.md`。

### 0.3 依赖关系与并行

```text
Step 1 (M4+C3, CPU) ──► Step 2 (R1 router, 复用 Step 1 特征)
Step 3 (B9 neighbors, GPU) 可与 Step 1/2 并行
Step 4 (回填+readiness) 需要 1/2/3 全部完成
```

---

## 1. Step 1 — M4 可预测性审计 + C3 gate 报告补完

目的：完成 doc/11 §2.3 M4 与 C3 gate 收口。回答"最优证据通道是否可以从
请求级廉价特征预测"——这是自适应主张的 falsification，先于一切架构。

### 1.1 特征构造

新脚本 `scripts/build_m4_features.py`，输入只允许：
`records_train.jsonl`（统计 train 侧先验）、`records_dev.jsonl`、
`runs/20260708_kuaisearch_b2z_bge_small_zh_dev/`（history-query 语义相似度可
复用其 embedding，若不可复用则现算，模型与 B2z 完全一致）。

特征族按 doc/11 M4 原表执行，不增不减：

| 特征族 | 特征 |
|---|---|
| query 侧 | query 长度；平均 IDF（train 上算）；train 内该 query 频次；该 query 的跨用户点击熵（train 上算，train 未出现的 query 记缺失并加 missing 指示列） |
| 候选侧 | 候选数；候选类目熵；候选品牌熵 |
| 历史侧 | 历史长度；历史-query 语义相似度（B2z embedding 均值余弦）；历史类目 vs 候选类目重合度 |

输出：`artifacts/m4/m4_features_dev.parquet` + 同构的
`m4_features_train_sub.parquet`（供 Step 2 复用，见 §2.2）。

**机械检查**：特征文件列名中不得出现任何来自 qrels / per_request_metrics 的
派生量；脚本内不 import 任何 evaluator 模块。

### 1.2 标签构造与模型

- 标签 = M3 oracle 的逐请求最优通道（三分类），从 M3 三个输入 run 的
  `per_request_metrics.jsonl` 现算 argmax（run id 见
  `reports/pps_m3_headroom_summary.json` 的 `input_run_ids`；tie 时按
  M3 同规则处理并记录 tie 率）。
- 模型只准两种：logistic regression、深度 ≤ 3 的决策树（doc/11 原文：审计
  特征信息量，不是拟合能力）。
- 评估：dev 内 **5-fold 交叉验证**（seed 20260708），报告 macro one-vs-rest
  AUC 的 fold 均值 ± 标准差，外加 per-channel AUC 与特征重要性。

### 1.3 canary（必跑，任一失败 = 管道有 bug，先修再读数）

1. 标签随机打乱后重跑 → AUC 应塌到 0.5 ± 0.02；
2. 把"该请求 history_b0b 的 per-request NDCG"直接放进特征（故意泄漏）→
   AUC 应显著暴涨。跑完删除该泄漏列，确认正式特征文件无此列。

### 1.4 M5 切片方向收口

E1/E2 方向证据大部分已有（`reports/pps_m3_bidirectional_slice.json`）。补齐
doc/11 M5 的熵分桶视角：

- 按 train 跨用户点击熵分桶（高/中/低三桶）：高熵桶内 query-only 相对
  oracle 的差距应大于低熵桶（E1 方向）；
- 按历史-候选类目重合度分桶：低重合桶内 history-only（B0b）应塌陷（E2 方向）；
- 每桶抽 5 个真实案例人工过目，附在报告里。

输出：`reports/pps_m5_slices.json` + `reports/pps_m5_case_review.md`。

### 1.5 C3 gate 收口

汇总 M3（已通过）+ M4 + M5，按 doc/11 C3 判据表出结论，落盘
`reports/pps_c3_motivation.json`。判据引用 §5.1，不得临场改。

### 1.6 验收清单

- [ ] `m4_features_dev.parquet` 存在，含 missing 指示列，无标签派生列；
- [ ] 两个 canary 结果写入报告；
- [ ] `pps_m4_predictability.json`：5-fold AUC、per-channel AUC、特征重要性、
      tie 率；
- [ ] `pps_m5_slices.json` + case review；
- [ ] `pps_c3_motivation.json`：三判据逐条 pass/fail + 结论一句话。

预算：CPU-only，一个工作日内。

---

## 2. Step 2 — R1 廉价 learned router（"必须打过的数字"）

目的：在 M3 三通道（query_b2z / history_b0b / static_b7_bge）之上训一个
廉价 router，产出未来提议系统的中间门槛数字（doc/15 §2/§5）。**R1 是对照，
不是系统**；本步骤明确不做任何表示学习。

### 2.1 官方版 R1b（train 拟合，dev 只评一次）

协议干净版，作为登记数字：

1. 从 `records_train.jsonl` 按 seed 20260708 抽 **20000 个请求**（保留至少
   1 个正例、候选数 ≥ 5，过滤统计落盘）；抽样清单落盘
   `reports/pps_r1_train_subset_manifest.json`；
2. 三个通道用与 dev **完全相同的实现与配置**给这 2 万请求打分（B2z zero-shot、
   B0b 规则、B7 用已冻结的 α，不得在 train 上重调 α）；
3. 用共享 evaluator 的同一指标实现对 train 子集算 per-request NDCG@10
   （标签来自 records_train 本身；新增薄封装
   `scripts/eval_train_subset.py`，内部必须调用 evaluator 的同一指标函数，
   不许复制粘贴指标代码）；
4. oracle-argmax 生成通道标签，用 §1 同款特征（`m4_features_train_sub.parquet`）
   训 logistic regression（主）与深度 ≤3 树（次）；
5. 应用到 dev：逐请求选通道、拼装 `scores.jsonl`（直接取所选通道的分数），
   走共享 evaluator 出数 →
   `runs/<date>_kuaisearch_r1b_router_lr_dev/`。

### 2.2 稳健性版 R1a（dev 内 5-fold cross-fitting）

同特征同模型，dev 内 5-fold（seed 20260708）：每折用其余四折的 oracle 标签
训练、对该折预测通道，拼出全 dev 的 scores → `runs/<date>_kuaisearch_r1a_router_cv_dev/`。
R1a 数字只作稳健性参考（它在分布上占便宜），**登记数字以 R1b 为准**；若
R1a 与 R1b 差距 > 2% relative，须在报告中解释。

### 2.3 可选扩展（各限一次，不得网格搜）

- soft 版：按预测概率对三通道 z-score 加权混合（一次，不调温度）；
- 特征消融：只留 query 侧特征的 router（一次，用于回答"gate 信息主要在哪"）。

### 2.4 对比与登记

- `compare_runs.py`：R1b vs B7-bge、R1b vs B0b、R1b vs oracle 输入各通道；
- 计算 **recovery ratio** = (R1b − 0.3305) / (0.4232 − 0.3305)，写入报告；
- 汇总 `reports/pps_r1_router_summary.md` + 登记 `experiments/pps_results.md`，
  边界卡注明"对照方法，非提议系统，永不进主系统"。

### 2.5 防作弊检查

- [ ] R1b 全程未读 `qrels_dev.jsonl`（fit 阶段）；dev 只在最终 evaluator 一步
      被使用；
- [ ] 特征文件与 Step 1 同一生成代码，无标签派生列；
- [ ] 候选 hash assert 通过；
- [ ] 通道分数复用/重跑的 config 与 M3 输入 run 逐项 diff 为空（除 split 外）。

预算：GPU 数小时（B2z 给 2 万 train 请求编码），其余 CPU。

---

## 3. Step 3 — B9 最近邻 query-conditioned baseline（ZAM + TEM）

目的：补 motivation 空档——证明（或诚实报告）**已有的 query-conditioned
个性化方法族**在本任务的固定候选设定下同样打不过静态混合。同时完成
doc/15 §9"最强最近邻实跑"要求。

### 3.1 选型与来源

| 编号 | 方法 | 理由 |
|---|---|---|
| B9z | ZAM（Ai et al. 2019，query-attentive user embedding + zero attention） | query-conditioned 个性化的领域经典，直接对应"attention 决定用不用历史" |
| B9t | TEM（Bi et al. 2020，transformer-based embedding model） | 最近邻中的 transformer 代表，与未来主干同架构族 |

来源：`https://github.com/kepingbi/ProdSearch`（PyTorch，含 ZAM/TEM/HEM/AEM），
clone 到 `baselines/pps_prodsearch/`，**pin commit 并记录 license**。
注意：这与已退役的 B6o（TF1.4 HEM 官方仓库）是不同代码库，不违反
`b6o_official_alignment.md` 的"不得复活 B6o"决定；本条差异须写进边界卡。

### 3.2 身份标签与对齐策略（预先冻结，吸取 B6o 教训）

- 身份标签：**`official-code, adapter to KuaiSearch interface, not externally
  aligned`**。
- **不做外部 Amazon 对齐**。理由引用 `reports/b6o_official_alignment.md`：该
  benchmark 家族的原始 split/checkpoint 公开不可重建，外部对齐已被证明不可
  验证；再烧一次预算没有信息量。此决定在此显式声明，不算默默突破 doc/14。
- 作为补偿的内部有效性检查（全部必过，替代外部对齐）：
  1. 显著 > Random（`compare_runs.py`）；
  2. 3 seeds（20260708/09/10）报告均值与方差；最高观测 seed 只作 run
     追溯和保守对比，不作为论文 headline（服从 doc/07 §11）；
  3. 决定性检查（同 seed 复跑前 1000 dev 请求分数逐位相等）；
  4. 抽 20 个请求人工看 top-5（防"指标正常但排序荒谬"）；
  5. 训练曲线收敛证据（loss 曲线落盘）。

### 3.3 Adapter 契约

- 只准从 `records_train/dev.jsonl` 读数据；history 用接口冻结的 ≤50 事件，
  可截短须写边界卡；item 文本用 title+brand+cat 与其他 baseline 一致；
- 候选打分：对每请求固定候选池打分输出 `scores.jsonl`，候选 hash assert；
- 词表/分词：与官方代码默认一致（中文按字符或官方 tokenizer，写进边界卡）；
- 超参：官方默认为主，只允许一个**预先声明的小网格**（embedding size ×
  learning rate，各 ≤ 2 值，共 ≤ 4 组合/模型），在 dev 上选点的规则沿用
  现有 baseline 惯例并登记完整网格结果。

### 3.4 止损（预先冻结）

- 总预算：**≤ 4 GPU-天**（两模型合计，含调试）；
- 每模型 ≤ 2 次完整训练尝试 + 声明网格；
- 若 B9z/B9t 任一无法在预算内跑通内部有效性检查 → 该方法降级为
  `attempted, not runnable`，写报告，motivation 保持现有"representative"
  措辞并加一句诚实脚注；**不许**为救它修改协议或延长预算。

### 3.5 输出

- `runs/<date>_kuaisearch_b9z_zam_dev_s<seed>/`、`.../b9t_tem_dev_s<seed>/`；
- `reports/pps_b9_neighbor_summary.md`（数字、seed 方差、内部检查、与
  B7-bge/B0b/R1b 的 compare 结果）；
- 边界卡两张进 `experiments/pps_baseline_cards.md`；
- doc/15 §9 最近邻对照表中 ZAM/TEM 两行标记"已实跑"。

---

## 4. Step 4 — motivation 回填 + 架构 readiness 备忘

前置：Step 1/2/3 全部完成并落盘。

### 4.1 论文回填（`paper/introduction_and_motivation.md`）

按 §5.3 的预写措辞分支改 §2.2 与 traceability 表：

- 加入 B9z/B9t 数字，与 SASRec/B5o 并列为"representative personalized
  baselines"，明确其为 query-conditioned 家族；
- 加入 R1b 数字与 recovery ratio，作为"廉价条件化已能吃掉多少 headroom"
  的测量（这同时是对 oracle 上界 caveat 的正面回应）；
- M4 AUC 写入 §2.2 或 §2.3（"条件化是可学的"证据）；
- traceability 表加四行（M4/M5/R1/B9），boundaries 段加 R1"对照非系统"、
  B9"not externally aligned"两条口径。

### 4.2 readiness 备忘（`reports/pps_architecture_readiness.md`）

一页纸，内容固定为：

1. 三个数字 + C3 gate 结论；
2. 未来系统的**双门槛**明确写死：显著 > B7-bge（0.3305）**且**显著 > R1b
   （数字待填），显著定义沿用 §0.2；
3. doc/15 §10 自检清单逐条预填当前状态（哪些证据已备、哪些待架构提案补）；
4. 历史模板当时要求结尾写“架构设计自此可以开工”；该分支现已被本文顶部的
   C5-R2/C5-R3 supersession 明确撤销。

**本轮到此为止，不写任何架构提案。**

---

## 5. 预先声明的判据（开跑前冻结，跑完只对答案）

### 5.1 C3 gate（沿用 doc/11，此处只抄录不新设）

| 判据 | 通过 | 失败动作 |
|---|---|---|
| M3 headroom | 已通过（+28.0%，CI 下界 +27.2%，split-half 同向） | — |
| M4 AUC（5-fold macro OvR 均值） | ≥ 0.65 → gate 可学 | 0.60–0.65 → 主张收缩为"静态可分桶改进"；< 0.60 → 放弃自适应主张，按 doc/11 C3 失败链走（切 Amazon-C4 复核） |
| M5 切片方向 | E1/E2 方向正确 | 方向反 → insight 表述重写后重审 |

### 5.2 R1 解读带（无 pass/fail，只解读并记录后果）

recovery ratio r = (R1b − 0.3305) / (0.4232 − 0.3305)：

| 带 | 解读 | 后果 |
|---|---|---|
| r ≥ 0.6 | 廉价特征已恢复大部分 headroom | 未来系统的叙事重心必须移到"超出通道选择的表示学习增量"，且双门槛中 R1b 成为主要门槛 |
| 0.3 ≤ r < 0.6 | 正常区间 | 双门槛照常 |
| r < 0.3 且 M4 AUC ≥ 0.65 | 特征可预测但 router 不涨 → 先查实现/特征-metric错位 | 修复并复核一次；仍 < 0.3 则如实登记 |

任何情况下 R1b 数字都进论文（它让 oracle 上界的 caveat 变成测量而非声明）。

### 5.3 B9 结果的措辞分支（预写，防事后编故事）

| 结果 | motivation 措辞 |
|---|---|
| B9 最优 ≤ B7-bge（不显著高于） | "including query-conditioned attention baselines (ZAM/TEM), representative personalized methods do not exceed the static-mixture waterline"——motivation 闭环，最强预期分支 |
| B9 最优显著 > B7-bge 但 ≤ R1b | B9 成为新 baseline-to-beat；§2.2 改写为"existing query-conditioned methods improve over static mixture but remain far below oracle / below a cheap router"；doc/15 §5 的门槛同步换成 B9 |
| B9 最优显著 > R1b | 同上，且 readiness 备忘必须警示：最近邻已较强，架构提案的创新性论证（doc/15 §9）负担加重 |
| B9 跑不通（止损触发） | 保持现有"representative"措辞 + 诚实脚注；doc/15 §9 的最近邻实跑要求由 R1（条件化对照）部分承担，并在架构阶段重估 |

### 5.4 显著性与效应

全部沿用 §0.2：paired bootstrap 10000 次、95% CI 下界 > 0 为显著、
+2% relative 为最小可主张效应。

---

## 6. 交付物总清单

| # | 交付物 | 步骤 |
|---|---|---|
| 1 | `scripts/build_m4_features.py` + `artifacts/m4/*.parquet` | 1 |
| 2 | `reports/pps_m4_predictability.json`（含 canary） | 1 |
| 3 | `reports/pps_m5_slices.json` + `pps_m5_case_review.md` | 1 |
| 4 | `reports/pps_c3_motivation.json`（C3 收口） | 1 |
| 5 | `scripts/eval_train_subset.py` + `reports/pps_r1_train_subset_manifest.json` | 2 |
| 6 | `runs/*_r1b_router_lr_dev/`、`*_r1a_router_cv_dev/` + `reports/pps_r1_router_summary.md` | 2 |
| 7 | `baselines/pps_prodsearch/`（pin commit）+ `configs/env/pps_prodsearch.txt` | 3 |
| 8 | B9z/B9t runs（3 seeds）+ `reports/pps_b9_neighbor_summary.md` + 边界卡 | 3 |
| 9 | `paper/introduction_and_motivation.md` 回填 diff | 4 |
| 10 | `reports/pps_architecture_readiness.md` | 4 |
| 11 | `experiments/pps_results.md` 登记全部新数字 | 1–4 |

---

## 7. 开发者开工 Prompt（直接粘贴使用）

```text
你在 /data/gkl/myrec 仓库工作。本轮任务的唯一执行依据是
doc/16_next_round_c3_router_neighbor_plan.md，请先完整阅读它以及它引用的
doc/11（M4/C3 定义）、doc/12（执行协议）、doc/15（设计原则，本轮只读不实现）。

任务：按 doc/16 的 Step 1 → Step 2 → Step 3 → Step 4 顺序执行（Step 3 可与
1/2 并行）。本轮不做任何新系统/新架构设计，只补三个数字并回填 motivation：
(1) M4 可预测性 AUC 并收口 C3 gate；(2) R1 廉价 learned router 的 dev
NDCG@10；(3) B9 = ZAM/TEM 最近邻 query-conditioned baseline 的 dev NDCG@10。

硬性纪律（违反任一条 = 产物作废，doc/16 §0.2）：
- test split（records_test / qrels_test）全程禁读；
- qrels_dev 只允许共享 evaluator 与 M 系列分析脚本读，任何特征/训练/打分
  代码不得打开它；
- 所有指标只出自 scripts/evaluate_scores.py，所有显著性只出自
  scripts/compare_runs.py；
- 每个 run 评测前 assert 候选集 hash 与 candidate_manifest.json 一致；
- doc/16 §5 的判据已冻结，跑完只对答案，不许看着结果调阈值或改措辞分支；
- 负结果照常写报告入库；每步交付物见 doc/16 §6，缺一不算完成。

执行要求：
- 每完成一个 Step，按该 Step 在本文中写好的检查规则逐条核对（Step 1：§1.3
  canary + §1.6 验收清单 + §5.1 C3 判据；Step 2：§2.5 防作弊检查 + §5.2 R1
  解读带；Step 3：§3.2 内部有效性检查 + §3.4 止损 + §5.3 B9 措辞分支；Step
  4：§4.1/§4.2 内容核对 + §6 交付物清单），再把报告落盘 reports/ 并在
  experiments/pps_results.md 登记。核对全部通过则直接进入下一 Step，无需等待
  确认；任一条不通过则暂停，向我汇报该步结论与未通过项（一段话 + 关键数字），
  得到处理意见后再继续；
- Step 3 的止损（≤4 GPU-天、每模型 ≤2 次完整训练、预声明小网格）触发时
  立即止损写报告，不得自行延长；
- 遇到 doc/16 未覆盖的协议歧义，停下来问，不要自行发挥；
- 全部完成后产出 reports/pps_architecture_readiness.md，本轮结束。
  不要开始写任何架构提案。
```

---

## 8. 与现有规范的关系

| 文档 | 关系 |
|---|---|
| `doc/11` | M4/M5/C3 的定义与阈值来源，本文 §1/§5.1 只抄录执行细节 |
| `doc/12` | run/评测/登记协议，本文全部沿用 |
| `doc/14` | 官方 baseline 框架；本文 §3.2 显式声明 B9 不做外部对齐（引用 B6o 证据），属记录在案的偏离 |
| `doc/15` | §5 双门槛、§9 最近邻实跑由本轮产出证据；架构设计在 Step 4 之后才允许启动 |
| `reports/b6o_official_alignment.md` | B9 对齐策略的依据；ProdSearch 仓库 ≠ B6o 官方 HEM 仓库，不构成复活 B6o |
