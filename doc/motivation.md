# Motivation：LLM 个性化重排中的 recurrence–transfer 失衡

状态：2026-07-20，V1.2 第一轮与后续机制探索已完成盘点，当前进入用户授权的新架构开发阶段。
本文继续记录冻结观察和机制结论边界；结果仍是 preliminary，不能替代后续多 seed 或前向时序
确认。当前执行规则见
[`../experiments/motivation/candidate_contrast_architecture_plan.md`](../experiments/motivation/candidate_contrast_architecture_plan.md)，
相关工作见 [`42_recurrence_transfer_related_work_zh.md`](42_recurrence_transfer_related_work_zh.md)。

## 1. 背景与研究问题

Ranking 是推荐系统中的核心任务：系统先从大规模商品库中召回候选，再根据用户当前需求
和长期偏好决定候选的展示顺序。本文关注的 reranking，是对已召回的小规模候选集合做
最后一阶段的精细排序。近年来，LLM 的文本理解、上下文建模和指令遵循能力使其逐渐被
直接用于这一阶段。[InstructRec](https://arxiv.org/abs/2305.07001) 将推荐与搜索任务统一
为指令学习，[TALLRec](https://arxiv.org/abs/2305.00447) 通过轻量参数高效训练使 LLM
适配推荐，[RecRanker](https://doi.org/10.1145/3705728) 则面向 top-k recommendation
系统研究 pointwise、pairwise 和 listwise 的 LLM 排序。它们代表了同一个基本方向：让
语言模型读取当前需求、候选商品以及用户信息，并直接产生候选排序分数或次序。

用户历史是个性化排序最重要的信息来源之一。即使两个商品同样符合当前 query，不同用户
仍可能偏好不同品牌、价格带、材质、功能或风格。因此，一个
history-conditioned LLM-based reranker 理应能够从历史行为中识别与当前 query 相关的
偏好，并利用这些偏好提高候选排序质量。已有 LLM 推荐与个性化商品搜索工作也普遍将
“加入历史后总体指标提高”视为个性化有效的证据。

然而，总体 NDCG 或 MRR 的提高没有说明模型如何利用历史。如果历史中的旧商品恰好再次
成为当前目标，模型只需识别重复商品就可能得到较大收益；这不等价于模型已经学会从旧
行为中抽取偏好，并把偏好迁移到一个历史中从未出现的新商品。因此，本文追问：

> LLM-based reranker 的历史收益，究竟有多少来自对历史商品的直接 recurrence，
> 又有多少来自跨商品的 preference transfer？

## 2. Recurrence 与 transfer 的可检验分解

**Recurrence** 指当前正例商品已经出现在用户历史中。模型可以通过 exact-item identity、
名称或高度相似的文本重新找到该商品。**Transfer** 指当前正例从未出现在历史中，模型
必须先从其他历史商品中抽取偏好，再将偏好对齐到当前候选。后者要求模型同时完成
query-aware history selection、偏好抽象和跨商品匹配，不能仅依赖目标商品复现。

对同一 checkpoint、同一请求和同一候选集合，分别输入真实历史、空历史和 wrong-user
历史，并计算逐请求 graded NDCG@10 差值。核心 surface 为：

- `target-repeat`：当前正例商品出现在历史中，测量 recurrence；
- `target-nonrepeat/no-candidate-overlap`：正例不在历史中，且整个候选集合与历史商品没有
  ID 交集，测量 strict transfer；
- `target-nonrepeat/other-candidate-overlap`：正例不重复，但其他候选与历史重叠，作为
  overlap surface 单独报告；
- `target-nonrepeat/no-history` 和 `no-observed-positive`：用于完整重构总体人口，不把 0
  增量混入前三个有历史、可解释的 surface。

strict-transfer surface 切断历史和当前候选之间的 exact-item recurrence。若真实历史仍能
可靠提高目标排序，收益才可归因于某种可迁移信息。除逐 surface 平均增量外，实验同时
报告其对 all-request 总体增益的人口加权贡献，避免小 surface 的大均值主导解释。

## 3. 冻结证据

### 3.1 先导证据

先导 KuaiSearch 实验测试过两个 history-conditioned LLM baseline。下表为同
checkpoint 的 `full-history - null-history` NDCG@10；“可靠”表示 normalized-query
cluster-bootstrap 区间不跨 0。

| 数据集 | LLM baseline | overall | recurrence | strict transfer |
|---|---|---:|---:|---:|
| KuaiSearch | Qwen3-Reranker-0.6B | `+0.0131`，可靠 | `+0.2315`，可靠 | `+0.0032`，不可靠 |
| KuaiSearch | InstructRec / Flan-T5-XL | `+0.0006`，不可靠 | `+0.0338`，可靠 | `-0.0002`，不可靠 |

两者都会响应历史并在 recurrence 请求上获得可靠收益，但没有在 strict transfer 上建立
可靠增益。Qwen 的人口加权 recurrence 贡献为 `+0.01401`，strict-transfer 贡献仅
`+0.00094`；其他 overlap 请求产生负向抵消后，overall 为 `+0.01305`。

非 LLM 与跨数据集实验只提供边界。KuaiSearch TEM 同样表现为 recurrence 可靠、strict
transfer 不可靠，但后续多 seed 没有稳定复现；JDsearch full-token ranker 的 strict
transfer 增量为可靠的 `+0.0286`，说明 transfer 并非不可能，不过 recurrence 增量达到
`+0.4249`，约为前者的 14.9 倍。这些结果足以提出问题，但不足以支持多种 LLM ranking
机制的共同结论。

### 3.2 V1.2 controlled-Qwen 第一轮

V1.2 在看见新 holdout 结果前固定四个主方法：专用 reranker 锚点 Q0，以及共享同一
`Qwen3-0.6B` 初始化的 Q1 InstructRec-style、Q2 RecRanker-style 和 Q3 TALLRec-style
独立最小重写。四者使用同一 32k 训练人口、候选契约、字段白名单、pilot seed
`20260714` 和共享 evaluator；Q0 的专用 reranker 预训练边界与 Q1--Q3 不匹配，因此不能
称为完全 matched-backbone 比较。W0 CoPPS-style 非 LLM structural witness 独立列出。

方法、配置、checkpoint、history assignment 和 evaluator 冻结后，才建立 4,000-request
新确认人口。其 surface 计数为 recurrence `288`、strict transfer `1,079`、other overlap
`290`、no history `266`、no observed positive `2,077`。下表为新确认集
`full - null` 的准确点估计；星号表示相应 5,000 次 normalized-query cluster-bootstrap
95% 区间不跨 0。全精度区间见结果登记与机器可读摘要。

| KuaiSearch method | Full NDCG@10 | overall | recurrence | strict transfer | other overlap | recurrence contribution | strict contribution |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q0 Qwen3-Reranker-0.6B | `0.20573` | `+0.01363*` | `+0.21542*` | `+0.01000` | `-0.06319*` | `+0.01551` | `+0.00270` |
| Q1 InstructRec-GeneralQwen | `0.19735` | `+0.01412*` | `+0.20732*` | `+0.00695` | `-0.03695*` | `+0.01493` | `+0.00188` |
| Q2 RecRanker-GeneralQwen | `0.20433` | `+0.01547*` | `+0.20971*` | `+0.00952` | `-0.03035*` | `+0.01510` | `+0.00257` |
| Q3 TALLRec-GeneralQwen | `0.19738` | `+0.01340*` | `+0.25671*` | `-0.00164` | `-0.06404*` | `+0.01848` | `-0.00044` |

四个主方法的 overall 和 recurrence 区间都为正；四个 strict-transfer 区间都跨 0，Q3
点估计为负；四个 other-overlap 区间都为负。`full - wrong-user` 得到同一方向：四行的
overall 与 recurrence 区间为正，strict-transfer 区间仍全部跨 0，other-overlap 区间仍
为负。wrong-user 中 3,571 个有历史请求有 2,745 个使用全局 other-user fallback，故它是
诊断控制，不能单独证明因果的用户特异性。

从 all-request 人口加权点估计看，Q0--Q3 的 recurrence 贡献分别为 `0.01551`、
`0.01493`、`0.01510`、`0.01848`，strict-transfer 贡献仅为 `0.00270`、`0.00188`、
`0.00257`、`-0.00044`。这是贡献分解，不是直接的 `recurrence - strict` 配对显著性检验；
因此可说收益在数值上由 recurrence 主导，不能说两个 surface 的差异已被直接显著性检验
证明。

internal-dev 8k、既有 2k cohort 和新 4k 中，Q0--Q3 的 recurrence 均为正且区间不跨 0。
Q1/Q2 的 dev-only strict-transfer 正区间没有在既有 2k 或新 4k 复现；Q0/Q3 的 strict
transfer 在三个人口均不可靠。由此，dev 上的局部正信号不能作为确认结果。

### 3.3 W0 transfer witness

W0 不进入四方法主表。新 4k 上它的 `full - null` overall 为
`+0.0012046 [-0.0006629, +0.0033676]`，recurrence 为
`+0.0201449 [+0.0060808, +0.0387518]`，strict transfer 为
`+0.0003419 [-0.0047349, +0.0052779]`。`full - wrong-user` 的 overall 和 strict 区间也
跨 0，只有 recurrence 区间为正。W0 因而只建立了小幅 recurrence response，没有建立
预期的 strict-transfer recovery 或 recoverable headroom。

一个旧 W0 scorer 因把 whitespace-normalized encoder query 当作 request identity，在
15,880 行后触发 hash mismatch；它没有发布 metadata、没有读取 qrels，已经由 raw-query
identity fix v2 checkpoint 和完整评分取代。这是机械非结果。Q2/Q3 resume canary 与 smoke
run 同样只证明工程契约，不参与 transfer 结论。五个正式方法均已完成，不存在
under-converged/pending 方法。

## 4. 当前假设、结论与边界

已有工作解释了为什么 recurrence 与 transfer 应分开研究。
[Repetition and Exploration in Sequential Recommendation](https://staff.fnwi.uva.nl/m.derijke/wp-content/papercite-data/pdf/li-2023-repetition.pdf)
发现序列推荐模型通常在 repeat-next 上远强于 explore-next，容易的重复样本可能成为
accuracy shortcut。另一方面，[ZAM](https://assets.amazon.science/37/52/3425ce394654af4687c7feba6b0f/a-zero-attention-model-for-personalized-product-search.pdf)、
[RTM](https://arxiv.org/abs/2004.09424)、
[CoPPS](https://doi.org/10.1145/3580305.3599287) 和
[BATA](https://doi.org/10.1145/3726864) 说明类别、属性、评论语义和 query--history 关系
可以支持跨商品偏好建模，但它们没有报告本文的 recurrence--transfer 分解。

V1.2 对核心假设给出的是**部分支持**：

- 支持的部分：在一个预先固定 seed、四个预先固定 Qwen ranking 变体和冻结的新确认人口
  上，overall history gain 与 recurrence 响应均可靠；人口加权收益在数值上主要来自
  recurrence，strict transfer 没有在新 4k 确认。该方向也与 internal-dev 和旧 cohort 的
  recurrence 结果相容。
- 未支持的部分：W0 没有恢复可靠 strict-transfer 信号，因此本轮不能用 witness 建立
  “当前 LLM 方法存在已可恢复的 KuaiSearch transfer headroom”。
- 当前最窄、可支持的观察是：**这些 frozen history-conditioned LLM ranker variants 在
  单 seed 的 KuaiSearch 回溯式确认人口上呈现 recurrence-dominant history use；严格迁移
  仍未建立。**

以下边界必须与结果同时保留：

- 只有一个训练 seed；bootstrap 只覆盖请求/normalized-query 抽样不确定性，不覆盖训练
  seed、不提供 family-wise 多重比较保证，也没有方法间配对优劣检验。
- 新 4k 是训练人口之前的 retrospective source-train population，不是 forward temporal
  holdout，也不声称 user/item/query 隔离。聚合审计显示 460 个 holdout 请求的事件后来出现
  在训练 history 中，1,018 个训练请求含 holdout event，416 个用户重叠；因此它证明的是
  recipe 冻结、request/session/time 边界及 qrels/score 时序门禁，不是行为完全独立泛化。
- Q0 与 Q1--Q3 的预训练角色不同；Q1--Q3/W0 均为 `-style` 独立最小重写，不能外推为官方
  InstructRec、RecRanker、TALLRec 或 CoPPS 的失败。
- strict-transfer 区间跨 0 不等于可迁移信号不存在；本轮也不能区分任务天然难度、监督
  不足、模型机制、统计功效或数据边界的相对作用。
- source test 从未打开。第一轮实验已经在注册停止点结束；当前获准开展机制诊断，但不增加
  论文方法或数据集、不打开 source test，也不直接提出或执行新 transfer 架构。

## 5. 当前机制问题

下一阶段不再重复证明“strict transfer 没有建立”，而是解释这一观察来自哪里。当前竞争解释
包括：

1. 当前历史与商品字段中的可迁移偏好信号不足，或统计功效不足；
2. 模型没有从历史中选择与当前 query 相关的行为；
3. 模型保留了 item/text recurrence，却没有形成跨 item-ID 的偏好抽象；
4. 偏好信息进入了内部表示，但没有被候选比较和最终 readout 使用；
5. recurrence 样本或非个性化相关性在训练目标和梯度中形成 shortcut；
6. 单 seed、回溯人口或 cohort shift 放大了当前观察。

W0 这个具体 CoPPS-style recipe 的继续优化不是当前高优先级。同一输入边界内是否存在
可学习的 strict-transfer signal 仍需用简单 positive control 检查，但它不作为深入模型机制
分析的默认阻断门。数据构造结果也不能停留在“换数据是否有效”，必须解释它暴露了什么架构
限制，以及相应训练信号为何需要与特定架构机制结合。

机制结论至少需要一个可逆行为干预和一个独立证据源（表示中介、训练动力学或 recoverability
control），并通过相应 negative control。主要预算用于 history routing、跨 ID 偏好抽象、
candidate readout 和训练 shortcut，并进一步形成面向高水平方法创新的架构机会矩阵。机制证据
收束后再进入具体方法实现。

## 6. 证据与当前计划索引

- V1.2 机器可读当前结论：
  [`../reports/motivation_current_summary.json`](../reports/motivation_current_summary.json)
- V1.2 结果登记、全精度区间与冻结 run identity：
  [`../experiments/pps_results.md`](../experiments/pps_results.md)
- 方法来源与实现边界：
  [`../experiments/pps_baseline_cards.md`](../experiments/pps_baseline_cards.md)
- 第一轮冻结协议：
  [`../experiments/motivation/protocol.yaml`](../experiments/motivation/protocol.yaml)
- 已完成的机制分析计划：
  [`../experiments/motivation/mechanism_analysis_plan.md`](../experiments/motivation/mechanism_analysis_plan.md)
- 部分完成并归档的四卡 Transformer 深挖计划与冻结清单：
  [`../experiments/motivation/transformer_deep_dive_plan.md`](../experiments/motivation/transformer_deep_dive_plan.md)、
  [`../experiments/motivation/transformer_deep_dive_manifest.yaml`](../experiments/motivation/transformer_deep_dive_manifest.yaml)
- 层定位后的反向组件状态移除扩展 V2（等长 neutral donor，独立于冻结 D2 family）：
  [`../experiments/motivation/transformer_component_necessity_extension_plan_v2.md`](../experiments/motivation/transformer_component_necessity_extension_plan_v2.md)、
  [`../experiments/motivation/transformer_component_necessity_extension_manifest_v2.yaml`](../experiments/motivation/transformer_component_necessity_extension_manifest_v2.yaml)；
  V1 在任何 score 启动前因位置混杂审计被 V2 取代，仅保留为先验记录。
- 最终全面报告合同与21项补充证据注册表：
  [`../experiments/motivation/transformer_comprehensive_report_plan.md`](../experiments/motivation/transformer_comprehensive_report_plan.md)、
  [`../experiments/motivation/transformer_supplemental_evidence_registry.yaml`](../experiments/motivation/transformer_supplemental_evidence_registry.yaml)、
  [`../experiments/motivation/transformer_supplemental_evidence_registry_manifest.yaml`](../experiments/motivation/transformer_supplemental_evidence_registry_manifest.yaml)。
- 设计资格不由绝对层号或单向 patch 决定：机器综合固定要求同一 parent bytes 上的
  same-request sufficiency、same-minus-wrong-history specificity 与 position-preserving neutral
  removal necessity 三门同时通过才称 history-specific state mediator；进一步通过 cross-request
  stress 与全部方向/尺度/随机方向结构负控，并在 Q2/Q3 的相同功能节点复现，才可改变全局设计排序。
- 当前“衰减而非字面反转”的解释边界：
  [`dev_log/2026-07-18_transformer_signal_attenuation_interpretation.md`](dev_log/2026-07-18_transformer_signal_attenuation_interpretation.md)
- 新 holdout history assignments 审计：
  [`../reports/motivation_kuaisearch_assignments.json`](../reports/motivation_kuaisearch_assignments.json)
- 先导证据已压缩进本节；当前可复核的机器报告从 V1.2 summary 开始。
- 完整相关工作：
  [`42_recurrence_transfer_related_work_zh.md`](42_recurrence_transfer_related_work_zh.md)
- 当前 Candidate-Contrast 架构开发计划：
  [`../experiments/motivation/candidate_contrast_architecture_plan.md`](../experiments/motivation/candidate_contrast_architecture_plan.md)

## 7. 首轮机制诊断与架构机会（2026-07-17）

M0--M3 已在预注册停止点完整收束。机器可读证据、完整 H0--H5 矩阵、逐层系统覆盖、
36 个注册 patch 单元、矛盾结果、机械性 non-result 和架构机会矩阵见：

- [`../reports/motivation_mechanism_first_diagnosis.json`](../reports/motivation_mechanism_first_diagnosis.json)
- [`../reports/motivation_mechanism_first_diagnosis.md`](../reports/motivation_mechanism_first_diagnosis.md)

当前最窄而可靠的机制链不是“LLM 没有使用历史”，而是：可见字段中存在有限的跨商品
信号，Q2 的中后层能够局部解码 brand/category 偏好代理；但这一表示尚未形成跨模型稳定、
request-specific 的偏好中介。到最终 readout 前，Q2/Q3 的正确 block-27 历史状态都会重现
full-history 的负 target margin，说明历史状态被因果使用，却以有害的符号或校准进入候选
比较。block 13 的正确状态反而把 margin 推到 null 之上；Q2 的 cross-request donor 效果更大，
所以不能把该移动解释为用户偏好恢复。简单地把训练曝光做 surface balance 也没有产生可信的
NDCG 改善，并显著恶化 margin。

| 假设 | 当前状态 | 首轮诊断 |
|---|---|---|
| H0 可见偏好信号不足/功效不足 | unresolved | 不存在正的 recoverability ceiling，但 Q2 的局部可解码性反对“完全无信号” |
| H1 query-aware routing 失败 | weakened | relevant filtering 不是充分解释；内部 attention routing 与 abstention 仍未被因果隔离 |
| H2 未形成跨 ID 偏好抽象 | weakened | Q2 有局部属性表示，Q3 不复现；可解码不等于可用或 request-specific |
| H3 偏好进入但 readout 未使用 | weakened | 最终状态确实被使用，但方向有害；问题更像符号、校准和偏好特异性，而非简单“不使用” |
| H4 easy-surface/recurrence 梯度 shortcut | weakened | 仅支持窄的 other-overlap 梯度冲突；recurrence dominance 不普遍，完整 surface balance 被反证为直接修复 |
| H5 单 seed/cohort/测量不稳定 | unresolved | 跨模型层深几何有共性，但表示与 fold 异质性仍不足以排除不稳定性 |

对完整 LLM4Rec 链路的覆盖不是只停留在 prompt 或微调：输入序列化、token budget、
embedding/位置、attention Q/V 梯度、层间表示、混合 residual state、history routing、
candidate interaction、native readout、损失与分 surface 梯度、LoRA 已有部分证据。attention
output、MLP 分支和 optimizer/scheduler 的有效更新归因仍是明确的未测区域；K、Q/K norm、
RoPE 后节点、单 head/edge、attention/MLP residual increment、final RMSNorm 也尚未从混合
post-block state 中拆开。因此当前报告既给出结果，也明确标出不能从已有 patch 推断的模块。

架构机会按证据排序如下：

1. **首选：ID-free factorized preference path + signed candidate residual。** 用可审计的
   brand/category/attribute/intent factor slots 表示历史偏好，把候选分数显式拆成 query-only
   relevance 与带符号的 candidate-specific residual，并加入空偏好/abstention 路径。factor
   slot、residual 和 gate 都必须可独立置零、patch 和反事实交换。
2. **次选：分别验证 factor bottleneck 与 signed residual。** 前者针对 H2，后者针对 H3；
   如果表示可解码性提高而 residual mediation 仍缺失，应把瓶颈定位到 readout，而不是继续
   扩大表示容量。
3. **边界方向：稀疏 query-conditioned router。** 只有在固定 token/position、selected-event
   可逆消融和显式 null route 都通过后，才值得提升为主方法；当前 relevant filtering 证据
   不足以单独支持它。
4. **暂缓：通用 surface-balanced sampling/gradient budget。** 已完成的固定曝光控制没有支持
   它作为立即修复；未来若重开，应先做 optimizer-state-aware 的有效更新归因，并以当前负
   控制为必须击败的基线。

这些机会在机制报告中仍不是已验证方法或论文贡献。Transformer 深挖后来只完成部分组件证据并
暂停；完整盘点见
[`../reports/motivation_post_stage_experiment_inventory_zh.md`](../reports/motivation_post_stage_experiment_inventory_zh.md)。
用户已于 2026-07-20 明确授权把主要机会收束为 Candidate-Contrast Personalization 并进入实现、
训练与同协议 benchmark。source test 继续关闭，不切换数据集，也不恢复旧诊断队列。

这里的全层 post-block sweep 只回答“full-history state 对 native ranking 行为的因果充分性轨迹在何处
发生相邻变化”，用于 split-sample 选择组件分解窗口和判断局部/分布式形态；它不直接观测
history-token flow，也不把某个 block index 提升为架构证据。最终设计判断只使用功能节点及双向干预
模式，不使用 outcome-selected 或跨模型硬对齐的绝对层号。
