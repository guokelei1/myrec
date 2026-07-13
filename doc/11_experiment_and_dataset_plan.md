# 11 - 实验与数据集设置完整方案

状态：C0--C5 motivation/data evidence retained；C01--C80 architecture search
terminally closed；当前进入 `doc/31` 定义的 R0 problem-discovery reset。没有 C81，
新的 architecture implementation/training 尚未授权。

前置：`10_direction_decision.md`（场景与数据轨道）、
`07_paper_design_constraints.md`（Tier 1/Tier 2 约束）、
`31_problem_discovery_and_architecture_iteration_protocol.md`（当前迭代状态机）。

当前权威边界：

- C5-R3 只建立 exact recurrence 强、coarse category transfer 未建立、混合会稀释
  item-only；它没有验证 architecture primitive。
- C01--C80 没有产生 CCF-A 级 proposed architecture。C80 在 fresh labels 前因
  mechanical contract 失败，utility 未知。
- Amazon ordinary full-token Transformer 已建立 token-level source observability，
  但还没有建立 ordinary Transformer 的明确架构缺陷。
- 当前先完成 scope audit、full-token parity、strong-baseline tuning 和 Failure Card；
  Failure Card 通过前不创建 tracked architecture source tree，只允许 doc/31 定义的
  `tmp/` disposable prototype。

历史 amendment、逐候选 gate 和数值保留在 `reports/`、`systems/README.md`、
`doc/16`--`doc/30` 及
`doc/dev_log/20260712_c01_c80_terminal_retrospective.md`，不再作为本文当前授权。

场景（冻结，后文不再改动）：

```text
用户发出真实商品搜索 query 后，结合用户历史行为（点击/购买序列，含商品明文文本）、
候选商品文本/属性，对该次搜索请求的固定候选池做个性化排序。
```

本文回答四件事：

1. 动机实验怎么做，怎样构成"现有方案都 fail"的证据链，核心对比工作是哪些；
2. 数据集怎么获取、审计、处理成统一格式；
3. 搭建实验架构的完整流程，每个关键节点的检查意见（checkpoint），防止错误积累；
4. insight 如何在负结果后收缩，并如何转成可证伪的 design hypothesis。

全流程分 6 个 Phase，每个 Phase 结束有一个 checkpoint（C0–C5）。正向 claim
只有 checkpoint 通过才能进入依赖该 claim 的实现；checkpoint 失败必须收缩
claim，不得按原路径继续。C5-R3 的失败因此终止其 multi-granular/coarse
路径，但其合格负结果可以定义一个新的、更窄 design problem；该路径须另立
pre-outcome gate，且在通过前只允许 formulation/minimal probe。检查结果一律
落盘为 `reports/pps_<phase>_audit.json`（沿用本仓库 VERA 阶段的 gate 报告习惯）。

---

## Phase 0 — 数据获取与审计

### 0.1 下载

| 数据 | 来源 | 用途 |
|---|---|---|
| KuaiSearch（优先 Lite） | HF `benchen4395/KuaiSearch` + GitHub 官方处理脚本 | 主轨 |
| Amazon-C4 | HF `McAuley-Lab/Amazon-C4` | 副轨（英文验证，Phase 5 才用） |
| C4 用户历史配套 | HF `zhiyuanpeng/amazon-c4-user-purchase-history` | 副轨 |
| Amazon-Reviews-2023 元数据（按需类目） | HF `McAuley-Lab/Amazon-Reviews-2023` | 副轨 item 文本 |
| JDsearch | GitHub `rucliujn/JDsearch` | 鲁棒性锚点（Phase 5 才用） |

Phase 0–4 只碰 KuaiSearch。副轨和锚点在主轨结论成立后才启动，避免三线并行拖垮迭代速度。

### 0.2 KuaiSearch 五表审计清单

对照官方 schema 逐项核实（不要信 README，信下载后的实际数据）：

1. **ranking 表**（8140 万行）：确认能否按 `(user_id, session_id, query)` 聚合出"一次搜索请求 → 候选列表"结构；每行是否为一个 (请求, target item) 对；`is_clicked` / `is_purchased` 标签确实存在且非全零。
2. **recall 表**（257 万行）：`impressed_item_ids` 是否等于 ranking 表聚合后的候选集（两表交叉验证，抽 1000 个请求核对）；不一致则明确记录哪张表作为候选池的 ground truth。
3. **item 表**：`item_id → title/brand/seller/cat L1-L3` 的 join 覆盖率（历史 item 和候选 item 分别统计）。
4. **用户历史**：ranking 表的 recent clicked/purchased items 字段是否带时间或至少有序；**能否证明历史事件都发生在请求之前**（防未来信息泄漏，这是本任务最致命的坑）。若 history item 没有 per-event timestamp，允许用 recall 日志自身的 user-item 事件流做交叉验证，不能仅凭字段名或直觉放行。
5. **relevance 表**：query-item 0-3 分与 ranking 表 query 的重合度（决定它能否作为语义校准辅助）。
6. 时间字段：session/请求级 timestamp 是否可用（决定能否做时间切分）。

### 0.3 规模控制

- 全量 ranking 表 8140 万行不可用于迭代。取**连续时间窗**（非随机行采样，保住 session 完整性和时间切分能力）：目标规模 train ≈ 20 万请求、dev ≈ 2.5 万、test ≈ 2.5 万。
- item 表全量保留（只做文本查找，不占训练成本）。
- 采样脚本必须固定 seed 并落盘采样清单（user_id/session_id 列表），保证全程可复现。

### ✅ Checkpoint C0（数据可用性 gate）

通过标准（写入 `reports/pps_c0_data_audit.json`）：

- [ ] 候选可按请求聚合，候选数分布合理（中位数 ≥ 10；若中位数 < 5，fixed-candidate ranking 不成立 → 触发回退）；
- [ ] 点击率在合理区间（每请求至少 1 个点击的请求占比、整体 CTR；若 CTR > 50% 或 < 0.1%，怀疑标签语义理解错误，回头读 schema）；
- [ ] 历史/候选 item → 文本 join 覆盖率 ≥ 95%（低于则统计缺失模式，决定 mask 策略）；
- [ ] 历史无未来泄漏：优先抽 1000 请求验证 history item 的交互时间 < 请求时间；若 ranking history 不带 per-event timestamp，则使用 recall 日志内交叉验证替代：
  1. 扫 `recall_lite` 构造 `events[user_id][item_id] = sorted(time_index)`，事件来自 `clicked_item_ids ∪ purchased_item_ids`；
  2. 从 recall 请求中、限定 `time_index ≥ 全局中位数` 后用 seed 20260708 抽 1000 个 `(user_id, session_id, query, time_index)`；
  3. 扫 `rank_lite` 命中抽样请求首行，取 `recently_clicked_item_ids + recently_purchased_item_ids`；
  4. 对每个 history item 分类：`past_supported`（存在 event time < t）、`same_time_only`（无 < t，仅有 == t）、`future_only`（仅有 > t）、`unobserved`（该 user-item 在 recall 窗口无事件）；
  5. 通过条件：`checked_requests == 1000`，且 `future_only / (past_supported + same_time_only + future_only) ≤ 0.001`，且 `past_supported / total_history_items ≥ 0.20`。`same_time_only` 率 > 5% 时必须人工复看；报告 caveat：这是日志内交叉验证（覆盖率 X%），不是官方字段级保证，`unobserved` 部分不可证伪。
- [ ] 时间字段可支撑全局时间切分。

**失败回退**：

- 候选结构不成立 → 改用 recall 表 `impressed_item_ids` 构造；仍不成立 → 主轨换 JDsearch（结构成熟），KuaiSearch 降级为语义辅助。
- `recently_clicked_item_ids` / `recently_purchased_item_ids` 泄漏检查失败 → 标准化阶段弃用这两个 ranking history 字段，改用选定时间窗内 recall 请求的先前 click/purchase 事件自建 history（只保留 `event_time < request_time`，最近 ≤50 条，按时间升序）。报告 caveat：序列更短，窗口初期用户 history 为空，但因果性由构造保证。

**不要在结构不清或因果性不成立的数据上硬做**。

---

## Phase 1 — 任务协议冻结（做完就不许改）

### 1.1 任务定义

```text
输入  x = (query 文本, 用户历史序列[item 文本+行为类型+时间], 固定候选集[item 文本+属性])
输出  候选集上的全排序
标签  is_clicked（主）、is_purchased（次）
```

### 1.2 统一记录格式（含标签物理隔离）

所有方法（包括 BM25 和 LLM）只准从这一个 JSONL 接口读数据，禁止任何方法私自回原始表取特征（07 号文档 §3 的落地）：

```json
{
  "request_id": "…",
  "user_id": "…", "session_id": "…", "ts": 0,
  "query": "明文query",
  "history": [
    {"item_id": "…", "title": "…", "brand": "…", "cat": ["L1","L2","L3"],
     "event": "click|purchase", "ts": 0}
  ],
  "candidates": [
    {"item_id": "…", "title": "…", "brand": "…", "seller": "…",
     "cat": ["L1","L2","L3"], "clicked": 0, "purchased": 0}
  ],
  "masks": {"history_present": true, "text_coverage": 0.98}
}
```

**标签物理隔离（防止评测标签流入打分代码）**：

- `records_train.jsonl`：candidates 保留 `clicked`/`purchased`（训练需要）；
- `records_dev.jsonl` / `records_test.jsonl`：candidates **不含任何标签字段**（blind records）；
- 标签单独落盘为 `qrels_dev.jsonl` / `qrels_test.jsonl`：

```json
{"request_id": "…", "clicked": ["item_a", "item_b"], "purchased": ["item_a"]}
```

- qrels 文件只允许 **共享 evaluator 和 M3–M6 分析脚本** 读取。任何 baseline
  的打分/训练代码 import 或打开 qrels 文件 = 该方法全部 run 作废；
- C1 时机械检查：dev/test records 中 grep 不出 `clicked`/`purchased` 字段。

**历史构造规则（冻结，所有方法共用）**：

- history = 请求 ts 之前该用户最近的 ≤ 50 条 click/purchase 事件，按时间升序；
- 该截断在**标准化阶段一次性完成**，写进 records；方法可以进一步截短
  （必须写进 baseline card），但禁止自行延长或回原始表重取更长历史；
- history 为空的请求保留，`masks.history_present = false`。

### 1.3 切分协议

- **全局时间切分**：按请求 ts 排序，前 80% train / 中 10% dev / 后 10% test；
- 同一 session 不跨 split（session 归属其最后一次请求的时段）；
- 保留标注子集：cold user（train 中历史长度 < 5）、长尾 item（train 内出现 < 5 次）、短/长 query——只用于切片报告，不用于筛数据。

### 1.4 指标声明（提前宣布，主指标裁决一切）

| 角色 | 指标 |
|---|---|
| 主指标 | **NDCG@10（click 标签）** |
| 次指标（只描述不裁决） | MRR、Recall@10、purchase-NDCG@10、AUC/logloss |
| 最小可主张效应（提前声明） | dev NDCG@10 相对最强 baseline **+2% relative**；低于按平手报告 |

**指标精确定义（写死，与单元测试一一对应）**：

- 相关性为二值：`rel(i) = 1` 当且仅当该候选被点击（purchase 指标同理换标签）；
- `DCG@10 = Σ_{i=1..10} rel(i) / log2(i+1)`（i 为排序位次，从 1 开始）；
- `IDCG@10` 用 `min(该请求正例数, 10)` 个 rel=1 计算；`NDCG@10 = DCG@10 / IDCG@10`；
- `MRR = 1 / (第一个正例的位次)`，在**全候选列表**上计算（不截断）；
- `Recall@10 = |top10 ∩ 正例| / |正例|`；
- 所有指标先按请求计算，再对请求取**未加权平均**；
- `purchase-NDCG@10` 只在"至少一个 purchase 正例"的请求上计算，必须同时报告
  该子集占比（coverage），防止用小子集讲大故事；
- relative 提升定义：`(a − b) / b`。

**排序与并列（tie）规则（防操纵，evaluator 内部实现，所有方法一致）**：

- evaluator 按 `score` 降序排序；score 相等时按
  `sha256(request_id + candidate_item_id + "20260708")` 升序打破并列；
- 禁止用 item_id 字典序或输入行序做 tie-break（二者可能与流行度/上架时间相关，
  会给某些方法免费送分）；
- score 必须是有限浮点数；出现 NaN/Inf/缺失 → evaluator 直接报错，不静默置 0。

**"显著" 的统一定义（本计划所有 gate 中的"显著高于"都指此检验）**：

- 请求级 paired bootstrap：对 dev 请求集合重采样 10000 次，计算两方法
  NDCG@10 差值的 95% CI；
- "A 显著高于 B" ⇔ 95% CI 下界 > 0；
- 该检验由共享脚本（`compare_runs.py` 一类）实现一次，禁止各方法自算 p 值；
- gate 阈值与实测效应落在噪声区间内时，按 gate 失败处理（07 §11）。

### 1.5 过滤规则（全部落盘统计）

- 测试/验证集剔除无任何正例的请求（无法算 ranking 指标），train 保留（可做 CTR 目标）；
- 剔除候选数 < 5 的请求；
- 每条过滤规则报告删了多少数据。

### ✅ Checkpoint C1（协议 gate）

- [ ] split 清单落盘并 hash（`reports/pps_c1_protocol.json` 记录三个 split 的请求数/用户数/候选数/正例率 + 文件 hash）；
- [ ] **候选集 manifest**：每个请求的候选 item 列表单独落盘并 hash——后续每个方法评测时 assert 候选集 hash 一致，这是"identical candidate sets"的机械保证；
- [ ] **标签隔离检查**：`records_dev/test.jsonl` 中机械检查不含 `clicked`/`purchased` 字段；`qrels_dev/test.jsonl` 单独落盘并 hash 进协议报告；
- [ ] 指标代码单元测试：NDCG/MRR/Recall 对 3 个手算小例断言通过（指标代码错误是最贵的错误，会污染之后所有数字），且至少 1 个测例覆盖 **tie-break 规则**（同分候选在不同输入行序下指标不变）；
- [ ] 两个 canary 测试跑通：(a) 随机打乱标签 → 所有指标塌到随机水平；(b) 把正例 item 的 title 拼进 query → 指标暴涨。两个 canary 不符合预期说明管道有 bug。

**从 C1 之后，协议文件进入 git 并视为冻结。任何改动 = 全部实验作废重跑。**

---

## Phase 2 — 动机实验（核心：证明现有方案各自 fail 且缺口互补）

### 2.1 动机的逻辑结构

动机不是"我们的方法比 baseline 好"，而是一个**失败矩阵**：

```text
E1  纯 query 方法：在"同 query 不同用户点不同商品"的请求上系统性失败（不个性化）。
E2  纯历史/序列方法：在 query 与历史背离的请求上系统性失败（被历史误导）。
E3  静态融合（全局固定权重）：显著低于逐请求 oracle 切换 → 缺口在"自适应"，
    不在"再加一个通道"。
E4  LLM 重排：质量有上界但延迟/成本不可用，且随历史长度增加而退化
    （复现 "历史使用低效"）。
```

E1+E2 证明两类单通道方法互补失败，E3 证明简单融合不解决，E4 证明"堆大模型"不是答案。四条齐了，"轻量、query 条件化的证据选择"才有立身之地。

### 2.2 方法矩阵（核心对比工作全集）

| 编号 | 方法 | 证据通道 | 对应文献/实现 | 证据性质 |
|---|---|---|---|---|
| B0a | Popularity（train 内点击量） | 无 | 自实现 | 下界 |
| B0b | Recent-behavior（候选与近期历史 item/类目重合打分） | 历史 | 自实现 | 下界 |
| B1 | BM25(query → title+brand+cat) | query 词法 | Pyserini/自实现 | 单通道 |
| B2 | Dense bi-encoder（中文：bge-m3 或 gte 系列，zero-shot + 可微调版） | query 语义 | 官方权重 | 单通道 |
| B3 | Cross-encoder（bge-reranker，zero-shot + 微调） | query 语义（强） | 官方权重 | 单通道上界 |
| B4 | SASRec / BERT4Rec（query-blind，候选打分） | 历史 ID 序列 | RecBole | 单通道 |
| B5 | DIN / DCNv2（KuaiSearch 官方 ranking baseline 复现） | 全特征工业 CTR | 官方脚本优先 | 场景原生 SOTA |
| B6 | HEM / ZAM / TEM（PPS 经典，明文文本适配） | query+历史浅融合 | 官方代码 + adapter | 领域经典 |
| B7 | **Static mixture**：`α·z(query_score) + (1−α)·z(history_score)`，α 全局网格搜 dev | 双通道静态 | 自实现 | **最重要的对照** |
| B8 | LLM zero-shot rerank（Qwen2.5-7B/72B，top-20 候选，历史截断 5/20/50 三档） | 全部（prompt） | 推理脚本 | 语义上界+成本 |

实现优先级：**B0/B1/B7 最先**（零训练成本，当天出数），B2/B4 其次，B3/B5/B6 再次，B8 最后（只跑 dev 抽样 2000 请求）。

每个方法登记边界卡（官方代码 / 仅 adapter / 改结构 / zero-shot），存 `experiments/pps_baseline_cards.md`。

### 2.3 动机实验设计

**M1 单通道全跑**：B0–B6 在同一候选 manifest 上出 dev 指标。

**M2 静态混合**：B7，α ∈ {0, 0.1, …, 1.0}。产出两个数：最优全局 α 的 NDCG@10；α 曲线形状（平 → 融合本身没张力，警报）。

**M3 逐请求 oracle headroom（本方向的生死实验）**：

```text
oracle = 对每个 dev 请求，在 {最优 query 单通道, 最优历史单通道, B7} 里逐请求取
         NDCG@10 最高者。
headroom = oracle NDCG@10 − max(全局单方法 NDCG@10)
```

- headroom ≥ +5% relative → 自适应机制有空间，方向活；
- headroom < +2% → **insight 证伪，触发回退**（见 C3）。

**选择噪声护栏**（对多个含噪 per-request 指标取 max 会系统性抬高 headroom，
必须同时报告以下三项，缺一不得宣布 M3 通过）：

1. headroom 的请求级 bootstrap 95% CI（复用 1.4 的检验实现）；
2. split-half 一致性：把 dev 请求随机分成两半（固定 seed），两半各自计算
   headroom，方向与量级须一致（同号且都 ≥ +2%）；
3. oracle 通道选择分布：若 oracle 在 >90% 请求上选同一个通道，headroom 主要
   来自剩余请求的噪声，自适应主张要降级——此时必须结合 M4 的可预测性结果
   一起判断，不许单独引用 headroom 数字。

**M4 可预测性审计（决定 gate 是否可学）**：

用请求级特征预测 M3 中 oracle 选了哪个通道（三分类/二分类），模型只准用 logistic regression + 浅决策树（此处刻意不用大模型——审计的是特征信息量，不是拟合能力）：

| 特征族 | 特征 |
|---|---|
| query 侧 | query 长度、平均 IDF（具体度代理）、train 内该 query 频次、**该 query 的跨用户点击熵**（同 query 下不同用户点击 item 的分布熵，train 上算） |
| 候选侧 | 候选数、候选类目熵、候选品牌熵 |
| 历史侧 | 历史长度、历史-query 语义相似度（B2 embedding）、历史类目 vs 候选类目重合度 |

产出：预测 AUC + 特征重要性。AUC ≥ 0.65 → gate 可学；< 0.6 → 自适应主张不成立，即使 headroom 存在也只能做静态改进。

**M5 失败切片（写进论文 intro 的证据）**：

- 按"跨用户点击熵"分桶：高熵桶里 B1/B2/B3（query-only）相对 oracle 的差距 → E1 证据；
- 按"历史-候选类目重合度"分桶：低重合桶里 B4/B0b（history-heavy）的塌陷 → E2 证据；
- B7 vs M3 oracle 的差 → E3 证据；
- 每桶附 5 个人工检查的具体案例（真实 query + 历史 + 排序对比），防止指标假象。

**M6 LLM 成本-质量曲线**：B8 三档历史长度的 NDCG@10 + 每请求延迟/token 成本。预期复现"历史变长不涨反跌"→ E4 证据。

### ✅ Checkpoint C2（baseline 可信度 gate，在看 M3 结果之前完成）

- [ ] 所有方法评测前 assert 候选集 hash 与 manifest 一致（有一个方法私自换候选池，整个对比作废）；
- [ ] B5（DIN/DCN）与 KuaiSearch 官方论文数字对齐到 ±10%（对不齐 → 先查协议差异并写入报告，不许带着不明差异前进）；
- [ ] B1 BM25 显著 > B0a popularity（不满足 → query 字段或分词有 bug）；
- [ ] B4 history-only 显著 > random（不满足 → 历史 join 有 bug）；
- [ ] 每个方法的 dev 指标由**同一份评测脚本**产出（禁止各方法自带评测代码）；
- [ ] 抽 20 个请求人工看 B1/B2 的 top-5 排序是否符合直觉（挡住"指标高但排序荒谬"的静默错误）。

**顺序纪律：C2 通过之前不许看 M3–M6 的结果**（先保证仪器准，再读读数，防止拿有 bug 的 headroom 自我说服）。

### ✅ Checkpoint C3（动机 gate = 方向生死判定）

提前声明判据，跑完只对答案：

| 判据 | 通过 | 失败动作 |
|---|---|---|
| M3 headroom | 点估计 ≥ +5% relative **且** bootstrap 95% CI 下界 ≥ +2% **且** split-half 同向 | < +2%（或 CI 下界 < 0，或 split-half 反向）→ 主轨方向证伪，切 Amazon-C4 副轨重跑 M1–M4（C4 的长描述 query 分布不同，结论可能反转）；C4 也失败 → 回到 10 号文档重新选题 |
| M4 AUC | ≥ 0.65 | 0.6–0.65 → 主张收缩为"静态可分桶改进"；< 0.6 → 放弃自适应主张 |
| M5 切片方向 | E1/E2 两个失败模式方向正确 | 方向反了 → insight 表述重写后重审 |

C3 通过后产出**动机报告** `reports/pps_c3_motivation.json` + 一页纸失败矩阵图表——这就是论文 intro 的骨架，也是后续所有开发的锚。

---

## Phase 3 — 数据集正式定稿（动机成立后才值得花这个钱）

1. 把 Phase 1 的抽样协议扩到论文规模（train 请求量按算力上调，dev/test 不动——**dev/test 一旦见过结果绝不扩充或更换**）；
2. 补齐切片标注：cold/warm user、head/tail item、短/长 query、高/低点击熵 query、历史-query 对齐/冲突，全部作为 record 级 tag 写进 JSONL；
3. 发布件准备：处理脚本 + seed + split hash + 过滤统计 + 字段文档（构造过程可审计本身是论文贡献之一）；
4. 启动副轨与锚点的数据处理（此时才做）：
   - **Amazon-C4**：query 直接用官方；历史用 `amazon-c4-user-purchase-history`（已保证只含目标交互之前的购买）；候选池 = BM25 top-100 ∪ {ground truth}，落盘 manifest；leave-one-out 时间切分；诚实标注"query 为 review 改写（半合成）"；
   - **JDsearch**：官方 test 候选（≤200）+ 0/1/2/3 标签原样用，train 内切 dev；只跑不依赖明文的方法（gate/历史选择机制），用途是证明机制增益不依赖明文文本。

### ✅ Checkpoint C4（数据定稿 gate）

- [ ] 三个数据轨用**同一个 JSONL 接口**（1.2 节），差异只体现在 mask（07 §2/§3 的 unified interface 约束机械化）;
- [ ] 三轨各自通过 C0 式审计（泄漏 canary 必须重跑）；
- [ ] `if dataset == X` 在代码库里 grep 不出来（允许 `if masks.history_present` 这类证据条件）。

---

## Phase 4 — 候选 insight（后续验证，先立此存照）

**历史状态（2026-07-10）：本节原路线已退役但不删除。** Insight-2 已被
`rho=-0.0110` 证伪；Insight-1 因其 M3/M4 前置构念失效而不再消耗实验预算。
下面原文用于审计研究路径，当前有效替代见本节末的 C5-R。

以下两个 insight 都以 M3/M4 为前置证据，各自带独立廉价证伪实验。建议先验证 Insight-2（更便宜），其结论直接决定 Insight-1 的建模粒度。

### Insight-1：槽位互补原理（Slot-Complementarity）

```text
Observation: query 已经指定的属性槽位（品牌/类目/规格），历史不应再插手；
             个性化的有效作用面只在 query 留空的槽位上（价格带/品牌/风格）。
             即：个性化 = 对 query 未指定槽位的补全，而不是对用户的整体偏置。

Architecture consequence: 历史记忆按属性面（facet）分解——品牌偏好通道、
             类目偏好通道、价格带偏好通道；query 解析出"已指定槽位 mask"，
             融合时只放行未指定槽位的历史通道。

Falsification（廉价，不用建完整系统）:
  1. 用规则/词典从 query 抽"是否指定品牌/类目"（中文电商 query 上足够准）；
  2. 做两个只加一路特征的线性 rerank：+品牌偏好分、+类目偏好分；
  3. 分桶对比：在"query 已指定品牌"的桶里，品牌偏好通道应无增益甚至负增益；
     在"query 未指定品牌"的桶里应有显著增益。类目同理。
  若增益与槽位指定状态无关（两桶差不多），insight 为假。
```

新颖性所在：现有 PPS 工作（ZAM/TEM/MAI）回答"个性化多少"（标量强度），本 insight 回答"**个性化什么**"（结构化方向），且给出一个可被单独证伪的行为学预测（指定槽位上个性化有害）。

### Insight-2：分歧定律（Consensus Law）

```text
Observation: 一条 query 值得个性化的程度，等于用户群体在该 query 下行为分歧的
             程度——跨用户点击熵高的 query 个性化收益大，共识型 query（大家都点
             同一个商品）个性化收益≈0。该量只用日志就能算，不需要任何模型。

Architecture consequence: 一个 pre-computed、query 级的个性化先验
             （personalization prior），作为融合门控的输入乃至初始化；
             未见过的 query 用具体度/候选熵特征回归外推。

Falsification（几乎零成本，M4 的直接延伸）:
  1. train 上算每条高频 query 的跨用户点击熵；
  2. dev 上算每条 query 的实测个性化收益（B7 最优 α 请求级重估，或
     oracle 通道选择）；
  3. 二者做相关（Spearman ρ）。ρ ≥ 0.4 → 定律成立且可直接写成一张
     "个性化收益 vs 行为熵"的散点图（论文核心图）；ρ < 0.2 → 为假。
```

新颖性所在：把"何时个性化"从模型内部的隐式 attention 变成一个**数据集无关、可跨数据集复测的行为定律**（KuaiSearch 上发现 → C4/JDsearch 上复测，正好用满三条数据轨），insight 层级高于"一个新模块"。

### 两者的合并出路（如果都成立）

单一原语即可表述："**query 条件化的证据路由**：query 决定历史证据在'是否用（Insight-2）'和'用哪个面（Insight-1）'两个维度上的通行权"。满足 07 §4"一个原语 + 至多三个组件"的预算。

### ✅ Checkpoint C5（insight 验证 gate）

- [ ] 两个 falsification 各自出具通过/失败结论（失败也写报告——负结果决定主张收缩方式，不许解释掉）；
- [ ] 每个成立的 insight 至少有一张"无模型也看得懂"的证据图（分桶柱状 / 散点+相关系数）；
- [ ] 只有 C5 通过的 insight 才允许进入建模阶段；两个都失败 → 论文降级为 benchmark+分析文（KuaiSearch 空白期内这仍是可发的退路）。

### ❌ Checkpoint C5-R2（历史 identity gate）

```text
Observation: 在 query-conditioned 候选池内，query score 与 target-user history
             aggregate 互补；history 收益具有 user-identity specificity，且
             history 缺失时该收益不可用。

Architecture consequence: query-anchored personalized residual。先形成
             query-candidate base，再由 target-user history 与当前 query/candidate
             的联合交互产生 masked residual。

Falsification: matched wrong-user history 必须显著弱于 true history；same-query
             donor 子集方向必须保持；no-history 请求必须严格等价于 query-only。
```

原 C5-R 执行结果曾记为 **passed**，但其 train-frozen wrong-history 控制现已
因时间不对称而 superseded。历史数字继续保留：

- B7 同时显著高于 B0b（+0.0166）与 B2z（+0.0249）；
- history-present 上 true B7 比 wrong-history B7 平均高 +0.0431，三个 seed 的
  CI 下界均 > 0；
- 2,709 个 same-query donor 请求上平均高 +0.0321，三个 seed 的 CI 均 > 0；
- 4,110 个 no-history 请求上 B7 与 B2z 逐请求指标完全一致。

C5-R2 使用 `doc/22` 的 earlier-dev/train strictly-prior donor snapshot，并对
target/donor latest-event age 施加逐请求 factor-four 上界：

- 7,614 个三 seed 共同 freshness-balanced 请求上，true-minus-wrong D2s 为
  +0.0374 / +0.0379 / +0.0362，三个 CI 下界均 > 0；
- 1,063 个 same-query + freshness-balanced 请求上，平均差值 +0.0095，但只有
  seed 20260710 的 CI 下界 > 0；seed 20260708/09 均跨零；
- 不同用户、严格先验、freshness 阈值、candidate coverage 与 4,110 个
  no-history fallback 审计全部通过。

冻结规则要求 same-query 至少 2/3 seed 显著，因此 **C5-R2 failed**。当前可保留
“rolling correct-history bundle 在广义匹配群体上有稳定预测价值”，但不能把
same-query identity specificity 写成已建立前提，也不能据此正式启动
personalized-residual system 训练。

### ❌ Checkpoint C5-R3（item/category recovery ladder 的终局 gate）

```text
Observation candidate: correct rolling history 对当前 candidate 的 fine item
                       与 coarse category alignment 都提供独立、非冗余信号。

Primary consequence: 一个 multi-granular candidate-history evidence-matching
                     primitive。

Sole fallback: 若 primary 失败，仅允许 coarse semantic category alignment；
               必须 3/3 seed 显著且三 seed 平均相对增益 >=2%。

Terminal rule: 两条均失败则 benchmark/analysis-only；不得在看到结果后把
               exact-item recurrence 改成已授权 paper primitive。
```

执行结果：

- item-only vs D2p：+0.03204/+0.03214/+0.03263，3/3 CI 下界 > 0；
- category-only vs D2p：+0.00059/+0.00053/-0.00003，0/3 显著，三 seed
  平均相对增益仅 0.1148%；
- full D2s vs item-only：-0.00538/-0.00521/-0.00634，3/3 CI 上界 < 0；
- 4,110 个 no-history 请求 rank/metric mismatch 均为 0；candidate、qrels、
  config、component reconstruction 与 dev-log integrity 全部通过。

因此 primary/fallback 均 **failed**，冻结的 terminal rule 生效。论文可承重的
诊断 insight 是“测试 bundle 的 history gain 集中于 exact repeat-item memory；
coarse category alignment 未建立且会稀释 item-only”。C5-R3 没有验证其中
任何候选 primitive。后续 C01--C80 又证明该宽 observation 不能直接选择架构；
full-token positive control 首先指向 representation interface。它现在只授权 R0
failure discovery，不直接授权 architecture formulation。

---

## Phase 5 — Problem discovery、architecture iteration 与 Tier-2 收尾

当前状态：**只授权 R0；architecture hypothesis 尚未获得准入。** 完整状态机、ID、
feedback 分类和 artifacts 见 `doc/31`。

### 5A. R0 problem discovery（当前授权）

1. 审计 KuaiSearch、Amazon、JDsearch 的信息对象与 confirmation data/power；
2. 在 KuaiSearch 与 Amazon 完成 ordinary full-token true/null/wrong observability
   parity，shuffle 默认只报告；
3. 先冻结一页 R0-M Motivation Brief，量化 problem prevalence、severity、shared
   baseline blind spot、Transformer asset、recoverable payoff 和 cheapest kill test；
4. 先做 model-family adequacy，再给存活的 ordinary full-token joint Transformer 与
   trainable baseline 对称的局部调参预算；
5. 只有 no-history/base preservation、ranking competitiveness 和 observability 都通过，
   才在 strong baseline 上建立 failure atlas；
6. 用独立 split/dataset 复现一个 ranking-relevant failure，排除简单修复和 nearest
   prior method，形成 Failure Card。

R0 每个 scientific round 最多 3 个 active failure idea、只 probe 前 2 个；日常只写
五字段 iteration record。工程 repair 不增加 scientific round。Failure Card 前允许
`tmp/` 下 CPU/tiny-data disposable prototype，不允许 architecture GPU training、
evaluator call 或 utility claim。

### 5B. Architecture development（Failure Card 通过后才授权）

1. 从 `Fxx` 推出一个 `Hxx` primitive 和 cheap falsifier；
2. 用 `Ixx` 处理 mechanics，用 `Txxx` 记录 score-affecting dev feedback；
3. 依次验证 mechanics、learnability、utility、specificity 和 attribution；
4. normal tuning 在冻结预算内允许，paper-level joint gate 不得提前阻塞 discovery；
5. simple/matched control 追平时降级为 interface/baseline result，不追加模块 rescue。

### 5C. Frozen confirmation and finals

1. 对 `DEV_SURVIVOR` 冻结 config、checkpoint rule、MDE、statistics、seeds、controls、
   candidate hash 和 numerical contract；
2. 使用一个经过 power analysis 的独立 holdout，不为每个 primitive 切碎 fresh cohort；
3. confirmation 失败关闭 claim，不在同一 cohort rescue；
4. 通过后完成 matched baselines、ablation、efficiency 和跨域 claim-specific 验证；
5. test 集只运行一次。

---

## 附录 A — 全流程一页图

```text
Phase 0  下载+审计 ──C0──► Phase 1  协议冻结 ──C1──►
Phase 2  动机实验 [C2 仪器校准 → M1–M6 → C3 生死判定] ──►
Phase 3  数据定稿(三轨) ──C4──► Phase 4  insight 证伪 ──C5──►
Phase 5A  R0 source+Motivation Brief+adequate strong baseline+Failure Card ──►
Phase 5B  Hxx/Ixx/Txxx bounded development ──►
Phase 5C  frozen confirmation+Tier-2+one-shot test

回退链：C0 fail → JDsearch 主轨；C3 fail → Amazon-C4 重跑动机；
        C5-R3/C01--C80 terminal → R0 problem-discovery reset；
        无合格 Failure Card → measurement/negative-design paper；
        confirmation fail → close claim，不 rescue holdout。
```

## 附录 B — 防错误积累的通用纪律

1. **仪器先于读数**：指标单测、canary、候选 hash、官方数字对齐，全部在看任何"有意义的结果"之前完成（C1/C2 的存在意义）。
2. **判据先于结果**：C3/C5 的阈值都已写死在本文，跑完只对答案，不许看着结果调阈值。
3. **一份评测脚本**：所有方法共用；任何方法的分数不是这份脚本产出的，一律无效。
4. **所有决策在 dev**：test 全程上锁，Phase 5 末尾开一次。
5. **每个 checkpoint 落盘 JSON 报告**：出错时能定位是哪一层引入的，不用全链路重查。
6. **负结果入库**：M3 headroom 小、insight 证伪、切片方向反——全部写报告存档，它们决定主张怎么收缩，而不是被遗忘。
