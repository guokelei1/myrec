# 15 - 提议系统设计原则（Proposed System Design Principles）

状态：**motivation 已完成，当前允许进入 proposed-system design formulation；
正式实现/训练仍须先通过本文定义的 pre-implementation design gate**。本文规定
当前设计阶段的 insight、边界与验收条件，不预设具体层数、维度或超参数。

证据边界修订：2026-07-10。原 M3/M4 positive claim 因 Random null 失败而永久
退出设计依据；替代证据来自先锁定、后执行的 matched wrong-user history
控制，以及 train-only 校准的 supervised/non-personalized/static controls
（`doc/17`--`doc/21`）。

C5-R2 修订（2026-07-10）：原 train-frozen wrong-history 对照被构念审计判定为
与 rolling true history 时间不对称。`doc/22` 的 freshness-matched prequential
替代控制已实现且完整性通过；7,614 个 balanced 请求保持稳定正增益，但 1,063
个 same-query balanced 请求只有 1/3 seed 显著，未过冻结 gate。当前权威裁决为
`reports/pps_c5r2_temporal_symmetric_identity.json`；下文任何“established”或
“authorized”措辞均以本修订为准。

C5-R3 结果边界（2026-07-10）：`doc/23` 在结果前冻结 item/category 分解、
multi-granular primary、唯一 coarse-category fallback 与其失败终局。item-only
3/3 显著超过 D2p；category-only 0/3 显著；full D2s 3/3 显著弱于 item-only。
因此 C5-R3 正确否证的是“item 与 category 都独立有益”以及“coarse category
alignment 足以承重”这两条具体 claim。它**不否证 design formulation 本身**。
结果反而建立了更窄的设计张力：历史证据的可信度不均，模型若把可靠的 exact
recurrence 与未经验证的跨商品迁移同等处理，会稀释强信号。静态水线仍为
item-only mean 0.3453755。权威数值见
`reports/pps_c5r3_candidate_history_alignment.json`。

阶段授权必须严格区分：

1. **Design formulation（当前已授权）**：允许定义单一 primitive、信息流、
   最近邻差异、廉价 falsifier、matched-capacity controls 与执行协议；
2. **Pre-implementation design gate（待冻结并通过）**：验证候选 primitive
   确实区分可靠复现与可迁移证据，而非换名 attention/router；
3. **Full implementation/training（尚未授权）**：只有第 2 步通过后，才进入
   正式模型训练、调参与跨数据集验证。

适用范围：本文档只规定设计时必须遵守的原则与禁区，**不包含任何具体
架构设计**，也**不重复 motivation 的数据**。它是 `doc/07_paper_design_constraints.md`
（整篇论文规范）在架构设计阶段的专门化：doc/07 管全论文，本文管"提议系统
该长成什么样"，与 doc/07 一致并补充，不重复其已有条款，只引用。

设计目标定位：一个面向 CCF-A 顶会质量的、query-conditioned 个性化商品排序
系统。本文的目的是在动手设计前钉死约束，防止为了单纯追求数值而拟合出
路由拼接、旧范式套壳、或无 insight 的模块堆叠。

---

## 0. 设计的来源：从 motivation 反推，而非从数值正推

设计的正当性必须从 `paper/introduction_and_motivation.md` 的 §1/§2 反推出来。
motivation 当前保留三条可直接进入设计推导的事实锚点（**不能脱离**这些事实
去引入 motivation 里不存在的张力）：

1. 候选池已被 query-conditioned。微调 text tower 的 D2t 显著高于 zero-shot
   B2z；加入合法 train popularity 的 D2p 达到 0.3240，但仍显著低于包含真实
   history 的静态组合。这不等于所有 query-only 信号已饱和。
2. C5-R3 把 B0b 精确分为 item recurrence 与 category affinity。item-only mean
   0.3453755，history-present 上 3/3 显著超过 D2p；category-only 0/3 显著，
   且 full D2s 3/3 显著弱于 item-only。因而测试 bundle 的稳定 history gain
   集中于 exact repeat-item，而非已建立的 coarse semantic preference。
3. 证据可用性有明确边界：33.6% 请求 history 为空，此时 D2s 与 seed-matched
   D2p 的 NDCG/MRR/Recall 逐请求完全等价；非空 history 中位长度仅 6，且
   38.4% 与候选在 deepest category 上零重合。D1m/D1a 直接训练 residual 后
   仍不能稳定超过其 base，query-attention 也不优于 mean-history；因此 event
   selection 是待证假设。M3/M4 仍是失败诊断，不参与推导。

三条事实共同支持的不是“category semantic memory 有效”，也不是“query
attention 已被证明”，而是一个更窄的设计问题：**同一段 history 内的候选证据
具有不同可信度；系统必须保存已验证的高置信复现信号，并防止未经验证的迁移
信号污染排序。** 任何设计决策都必须映射回这一问题；映射不上的组件即装饰，
按 §7 删除。

这也回答“现有系统是否不够好”：**在当前数据、候选协议与已测试方法范围内，
答案是肯定的。** 官方 RecBole SASRec（mean 0.2972）、proxy-aligned KuaiSearch
DNN/DCNv2（0.3063/0.3054）、带 provenance caveat 的 ZAM/TEM adapter
（0.2986/0.2940）以及 D1 query-attentive residual 都没有超过 item-only
0.3454；在冻结的 history-present 对比上，full D2s 自身也被 item-only 三个
seed 均显著超过。这些结果支持“代表性系统尚未
可靠利用超出 exact recurrence 的历史证据”，而不支持“所有既有系统都失败”
或“新系统必然成功”。这一区分正是当前 design formulation 的正当性边界。

---

## 1. 原则一：Insight 驱动，且可追溯到 motivation

- 设计必须由 **1–2 个 load-bearing insight** 驱动。insight 不是组件清单、不是
  结果表、不是训练 trick，而是一句能解释"这个架构为什么应该存在"的紧凑观察
  （`doc/07 §1`）。
- 每个 insight 必须：
  - 能从 §1/§2 motivation **直接推出**，不能凭空引入新张力；
  - 填 `doc/07 §1` 模板：Observation → Architecture consequence → Falsification；
  - 预测至少一个失败模式；
  - 可被一个廉价对照证伪（先证伪再建系统）；
  - 比"我们结合了词法+语义+协同信号"更强。
- **禁止**："我们组合了 X+Y+Z"式无 falsifiable primitive 的设计；insight 只在
  报告里出现、架构里找不到对应物的设计；insight 与 motivation 锚点无映射的
  设计。

当前 bounded insight 框架：

```text
Observation: history evidence has unequal empirical fidelity. Exact candidate
             recurrence is stable, while the tested coarse category transfer
             is non-informative and dilutes the stronger item-memory ranking.

Architecture consequence: formulate one candidate-conditioned evidence-fidelity
             calibration primitive. It must preserve reliable recurrence and
             allow a transferable personalized residual only when the joint
             query/history/candidate evidence supports it. This is one learned
             evidence contract, not a router over fixed scorers.

Falsification: before full implementation, a frozen cheap design gate must show:
             (i) repeat-present behavior does not degrade the item-only control;
             (ii) on the 4,677 history-present requests with no exact-repeat
             candidate, a transfer probe has stable positive value over D2p;
             (iii) coarse-only, wrong-user, shuffled-event, and query-masked
             evidence cannot reproduce that value; and
             (iv) no-history behavior remains rank-equivalent to D2p.
```

The observation is established by C5-R3; the architecture consequence is a
**design hypothesis**, not an empirical outcome. Design formulation may now
make the primitive precise. It may not claim semantic transfer, begin full
training, or select modules from dev outcomes until the falsifier above is
frozen and passed.

## 2. 原则二：一个成体系架构，不是路由 / 分类 / 集成

- 设计必须是**单一系统架构 + 单一 primitive**（`doc/07 §4`：primitive before
  components）。所有模块是这个 primitive 内部的实例、通道或诊断，而非独立
  打分器的拼接。
- **作为主系统，明令禁止**：
  - 固定通道上的 **router / mixture-of-experts**——它只恢复 oracle，不学习
    新的证据利用方式，且本质是 §8 警告的"oracle 当设计模板"。
  - **query-type 分类头 + 分支打分器**——这是 `if query-type == X` 的旧分支
    范式（违反 `doc/07 §2`），且属于老分类架构。
  - 命名模块的**无统一 primitive 拼接集成**——读起来像几篇论文缝在一起
    （`doc/07 §0` 要防的失败模式）。
  - **"固定通道分数 + 学习组合器"两阶段流水线**——本质仍是 router，不是
    end-to-end 学习证据表示。
- **learned router over 现有通道只能作为对照 / 消融**（见 `doc/baseline_notes/batch2b_motivation_logic.md`
  §5），不是主系统。它是一道**必过的中间门槛**：若主系统打不过一个廉价
  learned router，说明架构没有学到超出"通道选择"的东西，架构无价值。
- 架构必须**端到端可训练**，自己学习证据的表示与加权，而非组装固定通道的
  输出。
- 本节禁止的是对 **fixed scorer outputs** 的事后路由；不禁止单一端到端模型
  内部的可微 gating/attention。后者只有在表示、权重和排序目标联合训练，且能
  通过 matched-capacity 消融归因时，才属于统一 primitive，而非换名 router。

## 3. 原则三：Transformer / LM 架构族

- 主干必须是 **Transformer 或 LM-based 架构**。这是本项目预先选择的研究
  范围与现代强对照要求，不是 M3/M4 数据能够推出的科学结论；论文不得写成
  "oracle 证明必须用 Transformer"。
- **禁止作为核心**：纯 MLP / 分类 CTR 模型（DCN、DIN 类）、GBDT、传统协同
  过滤、手工特征 logistic 模型——这些是 baseline（B5o 等），不是提议系统。
- per-request 证据条件化机制须由 **attention / 序列建模**承载，但具体形式
  仍须由信息流、最近邻差异和消融决定；"attention 很自然"本身不构成创新性
  或因果归因。
- **"LM-based"指架构族，不等于在线调用大 LLM**。效率是 claim 的一部分
  （`doc/07 §12`，"lightweight"）：在线 LLM 调用目标为零（须证明，不能断言）。
  可接受形态包括紧凑 Transformer encoder 从头训练、冻结 / 蒸馏 LM encoder
  等；参数规模不是判据，架构族才是。

## 4. 原则四：统一接口，禁止 per-dataset / per-evidence 分支

- 单一架构处理 query + history 记录，靠 **mask / 条件化**处理证据缺失，不写
  `if dataset == X`（`doc/07 §2–3`）。
- 主轨 KuaiSearch、副轨 Amazon-C4、锚点 JDsearch 共用同一模型契约。query 有无、
  history 有无、item 文本有无均以 missing-evidence mask 表达，不以分支架构
  表达。
- 允许的条件化：证据是否在场、history 长度、query 长度、候选数、文本覆盖率、
  证据置信度。不允许的条件化：按数据集切代码路径、按 query-type 切打分器。

## 5. 原则五：比较门槛——必须打过 item-only 静态水线 AND 廉价 router

- 提议系统的价值 = **系统 − 最强 baseline**（当前为 C5-R3 item-only
  three-seed mean 0.3453755；seed 20260708 为 0.3450874）。full D2s 0.3416290
  仍是 bundled-history reference，但已被 removal ablation 超过。打过
  query-only/D2p/full D2s 都不算完整贡献。
- 必须内置一个**廉价 learned router over 现有通道**作为中间对照：这是架构
  价值的诊断对照（见 §2）。正式 R1b 为 0.3072，显著低于 B7-bge，因此当前
  又更低于 item-only；仍须单独 compare R1b，以证明增益不是协议或实现偶然，
  但不得把它叙述成更高的独立数值门槛。
- 若未来新协议继续采用 2% minimal claimable effect，系统须显著高于
  item-only、full D2s、D2p 和 R1b；按当前 item-only mean 对应约 0.3522831。
  binding 的是 item-only，R1b 仅保留为 fixed-channel router 诊断。
- **禁止**"加了个 transformer 就打过 BM25 / query-only"式的 trivial win。
- 必须存在**参数 / 算力匹配的 backbone 对照**：增益须归因于 query-conditioned
  证据机制，而非额外容量（`doc/07 §6.1, §15`）。机制消融（保留容量、移除
  条件化）应导致增益消失。

## 6. 原则六：可证伪，过 gate 再建系统

- 原 insight 的时间对称廉价 falsification 已按 `doc/22` 先锁定后执行。
  freshness-balanced 总体优势跨三个 seed 显著，no-history fallback 完全等价；
  但 same-query balanced 仅 1/3 seed 显著，因此完整 C5-R2 **未通过**。
- C5-R3 的有限 component falsification 已按 `doc/23` 完成：multi-granular
  primary 与唯一 coarse-semantic fallback 均失败，integrity 全通过。冻结终局
  继续约束这两条 claim；不能把 exact recurrence 或 coarse category 事后改写成
  已验证 paper primitive。但“可靠复现与不可靠迁移必须区分”是该负结果直接
  建立的设计张力，允许据此制定新的 pre-implementation falsifier。
- D1 的 train-fitted residual 检验为负：D1m/D1a 对 base 都是两正一负且 CI
  跨零，D1a 不稳定优于 D1m。因此设计可以测试 query-conditioned event use，
  但不得把它写成 motivation 已建立的事实；其价值必须由 D2s、matched-capacity
  与 query/history perturbation 共同证实。
- 当前 M4 机械上以 0.6688 通过冻结阈值，但 Random-oracle 标签 AUC 为 0.6952，
  history-present 子集上的原 M4 仅 0.6281；R1b 也未恢复 headroom。因此 M4
  不能再支持"廉价特征含有效条件信息"，只能作为被 Random null 否定的原始结果。
- 当前授权仅到 **design formulation**。原 query-anchored residual、
  multi-granular alignment、coarse semantic alignment、channel router、
  Consensus Law 与 entropy-conditioned personalization 均不能作为已验证前提。
  候选 evidence-fidelity calibration primitive 必须在设计协议中先精确定义，再
  通过本节的 repeat/non-repeat、wrong/shuffled/query-mask 与 no-history gate；
  不得通过放宽 C5-R2/C5-R3 判据或另取事后 subset 为旧 claim 翻案。
- 若后续 C4/C5 显示机制打不过 router 或静态混合，按 `doc/07 §13` **收缩
  claim**，而不是堆模块救场。失败的 gate 只能用于收缩主张，**不得被事后
  重新解释为对系统有利的证据**。
- **条件化坍缩诊断（必做）**：机制训练完成后，必须验证其行为确实随请求
  变化——例如证据权重须随 query/history mask、matched wrong-history 和
  event permutation 产生方向一致的变化，且 per-request 权重方差显著大于
  随机初始化对照。不得再用已失效的 M3 oracle 标签证明条件化。若权重近似
  常数，说明机制已退化为静态策略，条件化 claim 不成立，即使聚合指标
  更高也不能归因于条件化（此时增益更可能来自容量，回到 §5 的容量对照）。
- **event-level 假设诊断（必做）**：query swap/mask、history event
  permutation/mask 与 unconditioned matched-capacity 对照必须改变内部证据权重
  并消除相应增益。否则只能主张一个更强 encoder，不能主张 query 决定哪些历史
  事件是证据。

## 7. 原则七：复杂度预算与反"拟合架构"

- **一个 primitive + 至多 3 个命名组件**（`doc/07 §14`）。每个组件靠消融交租：
  移除后主指标变化 < minimal claimable effect 的组件必须删除（可降级为附录
  负结果）。
- **禁止**（这些都是"为追数值拟合架构"的典型信号）：
  - post-hoc 加模块追回某个 gap；
  - per-query-type 特殊分支 / 特殊case 阈值；
  - fallback 启发式规则（计入复杂度预算）；
  - 架构选型依据 test 数字回溯调整。
- 架构决策在 **dev** 上做；test 在冻结 config 后**只跑一次**（`doc/07 §10`）。
- minimal claimable effect 在 finals 前**声明**（`doc/07 §11`），低于它按 tie 报。

## 8. 原则八：oracle 证明什么、不证明什么

- 当前 oracle **不支持**可利用的 diagnostic headroom：Random 通道 oracle
  达到 0.4325/+30.9%，高于原 M3 的 0.4232/+28.0%；无 history 请求上原 oracle
  仍给出 +27.7%。它只记录先选后评的上偏，不再证明 router 有空间。
- oracle 仍可描述同标签上的 pairwise 排序差异，但不能证明新架构有必要。
  learned router R1b 的失败与 Random canary 一致，不能被解释为"更复杂 router
  才能吃掉 headroom"。
- split-half 的 +28.2%/+27.9% 只说明选择偏差对随机划分稳定；+28.0% 作为
  construct-validity 失败结果保留，不再称为 diagnostic upper bound。
- oracle 既未在 channel level 提供可利用证据，也不证明 query 能在一段 history
  内定位 event-level evidence/noise；前者先修 gate，后者再按 §6 单独证伪。
- 因此架构的正当性必须来自通过修订 gate 的 **insight**。当前 oracle 只作
  失败记录，既不是 headroom 支撑证据，也**不能当设计模板**。
- **禁止**把"oracle 说 routing 有效"当成新架构的理由——那只能论证 router，
  不能论证一个学习证据表示的端到端架构。新架构相对 router 的增量价值必须
  单独证明（§5）。

## 9. 原则九：创新性——新原语，不是旧架构的换皮或拼装

- 提议系统的 primitive 必须是**机制级创新**：创新性体现在 primitive 的
  数学形式与信息流上（query、history、candidate 三者如何交互），而**不是**
  训练 trick、损失函数微调、特征工程或超参搜索。
- **最近邻机制对照表（设计提案必含）**：提案必须显式列出与本 primitive
  最接近的已有机制，并对每一个说明本机制**不可归约**到它的具体差异点，
  且该差异点可被一个消融验证（把差异点退化回已有机制，增益应消失）。
  至少覆盖以下机制家族：
  - CTR 目标注意力（DIN 类 target attention：candidate 对 history 加权）；
  - 长序列检索式建模（SIM/UBR 类：先检索相关历史再建模）；
  - 个性化商品搜索的 query-attentive 用户建模（HEM/AEM/ZAM 类）；
  - Transformer 序列推荐 + query 拼接（SASRec/BERT4Rec 加 query 特征）；
  - 个性化搜索的 Transformer 编码器（TEM/CoPPS 类）。

架构开工前的最近邻实跑状态（doc/16 Step 3）：

| 邻居 | 状态 | KuaiSearch dev 3-seed mean | 最高观测 seed 与 B7-bge | 身份边界 |
|---|---|---:|---|---|
| ZAM | 已实跑；人工 review provenance 待作者确认 | 0.2986 +/- 0.0006 | -0.0311，CI [-0.0365, -0.0256] | official-code adapter, not externally aligned |
| TEM | 已实跑；人工 review provenance 待作者确认 | 0.2940 +/- 0.0009 | -0.0358，CI [-0.0412, -0.0303] | official-code adapter, not externally aligned |

两者只证明这些具体邻居在当前 Option-A/cold-product 边界下未超过静态混合；
不能外推为所有 query-conditioned PPS 方法都弱。正式来源：
`reports/pps_b9_neighbor_summary.json`。
两者在 human-review provenance 完成前只作补充设计上下文，不承重
motivation，也不阻塞 design formulation；主张边界由 C5-R3 component result、
C5-R2 identity boundary、D1 negative 和已闭环的代表性 baseline 共同约束。
- **禁止的伪创新**（均属"旧架构换皮"）：
  - 拿现有个性化架构加一个 query 特征拼接 / query embedding 相加，
    宣称"query-conditioned"；
  - 把已有注意力机制重新命名（如把 target attention 改叫"证据选择器"）
    而信息流不变；
  - 两个及以上已有架构的串联 / 并联，各自机制不变，只在输出处融合——
    这是 §2 禁止的拼接集成的变体；
  - 创新点只存在于论文叙述里，消融上与最近邻机制统计打平。
- 判据分成两层，不得混淆：若参数匹配的最近邻与本 primitive 统计打平，
  **性能优势 claim 不成立**；机制创新性则由信息流不可归约性、退化消融和
  最近邻对照共同判断，不能只凭一个 tie 自动否定，也不能只凭命名自动成立。
  若机制可归约到最近邻且退化消融无差异，创新性 claim 才不成立。最近邻中
  最强的 1–2 个仍必须实跑，不能只做文字比较。
- 与 §2 的分工：§2 禁止的是"多打分器拼接"这一**结构**失败模式；本条
  禁止的是"单一结构但机制陈旧"这一**机制**失败模式。二者都过才算新架构。

---

## 10. 架构阶段提交前自检清单

### 10.1 Design formulation 入场（当前状态）

- [x] motivation 已完成，且不依赖失效的 M3/M4 oracle。
- [x] bounded observation 可追溯到 C5-R3：exact recurrence 稳定，coarse
      category transfer 无独立增益且会稀释 item-only。
- [x] 已填写 Observation → Architecture consequence → Falsification 模板；
      architecture consequence 明确标为待设计/待证假设。
- [x] 当前最强水线、失败 claim、no-history contract 与禁止路径均已登记。
- [x] 当前只授权写设计提案、最近邻机制表、控制矩阵与执行协议；不授权训练。

因此 **design formulation 可以立即开始**。

### 10.2 Full implementation / training 入场（尚未满足）

设计提案进入实现前，以下必须全部为真；任一不满足，不进入正式训练：

- [ ] 将 evidence-fidelity calibration 写成一个数学上明确、不可归约为
      fixed-score router 的 primitive。
- [ ] 单一 primitive，**不是** router / 分类头 / 集成拼接。
- [ ] Transformer / LM 主干，attention 做 per-request 条件化。
- [ ] 统一接口，无 `if dataset == X` 分支，证据缺失用 mask。
- [ ] 能清楚解释如何**同时**打过 item-only、full D2s、D2p 和廉价 learned router。
- [ ] 有参数 / 算力匹配对照，增益可归因到条件化机制而非容量。
- [ ] 复杂度 ≤ 1 primitive + 3 组件，每个组件有 rent-paying 消融。
- [ ] pre-implementation design gate 已先冻结后通过：repeat-present 不弱于
      item-only；4,677 个 non-repeat history-present 请求上显著优于 D2p；
      wrong/shuffled/coarse/query-mask 不能复现；no-history 等价 D2p。
- [x] D1 negative 已登记；query-attention 未被当成既成 motivation 事实。
- [x] oracle 只作失败诊断，未被当成 headroom 证据或设计模板。
- [ ] 效率边界已想清（在线 LLM 调用为零、latency 含在线特征构造）。
- [ ] 已填最近邻机制对照表（§9），差异点逐条可消融验证，最强最近邻
      已列入实跑 baseline，不是旧架构换皮或多架构拼装。
- [ ] 已计划条件化坍缩诊断（§6）：训练后验证证据权重确实随请求变化。
- [ ] 已计划 event-level 假设诊断（§6）：query/history masking、event
      permutation 与 matched-capacity unconditioned control 能证伪机制归因。

---

## 11. 与现有规范的关系

| 文档 | 关系 |
|---|---|
| `doc/07` | 整篇论文规范。本文是其架构设计阶段的专门化，不重复，只引用。 |
| `doc/10 §2.2` | insight 模板来源。本文 §1 锐化其为本项目具体框架。 |
| `doc/11` | 原 C3/C5 历史判据保留；C5-R3 否证其两条候选路径，但不阻塞新的 design formulation。 |
| `doc/16` | M4/R1/B9 历史执行记录；M4 readiness 被 Random canary 否定，R1/B9 仍作 baseline。 |
| `doc/17` | 修复协议：aggregate complementarity + matched wrong-user identity control。 |
| `doc/18` | D1 supervised residual 诊断：mean/attention residual 均无稳定增益。 |
| `doc/19` | D2 fine-tuned text 与 non-personalized popularity control。 |
| `doc/20` | D2h 强化静态水线与 true/wrong identity reissue。 |
| `doc/21` | D2s 完整静态水线修复：将已验证的 D2p popularity 纳入 history mix。 |
| `doc/23` | C5-R3 item/category 分解；item-only 成为水线，primary/fallback 均失败。 |
| `doc/baseline_notes/batch2b_motivation_logic.md §5` | 提议系统定位备忘（非 router、transformer、含 router 对照）。本文 §2/§5 正式化为约束。 |
| `paper/introduction_and_motivation.md` | insight 锚点来源。本文 §0 三条锚点从其反推。 |

本文档现为当前 design formulation 的约束。偏离任一原则需在 `doc/dev_log/`
记录理由并显式修订本文，不得默默突破。**当前允许编写 architecture proposal
与 design execution protocol；不允许启动正式训练。** 训练授权只来自 §10.2
的 pre-implementation gate，而不是重新解释 C5-R3。
