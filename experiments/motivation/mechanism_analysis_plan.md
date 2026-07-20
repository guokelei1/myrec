# Motivation mechanism analysis plan

状态：2026-07-17，机制分析阶段已获用户授权。V1.2 第一轮实验已经完成并冻结；本计划只负责解释
为什么当前 history-conditioned Qwen ranker 呈现 recurrence-dominant history use，而不预设
LLM 原理上无法 transfer，也不直接提出或实现解决架构。

## 1. 当前事实与目标

冻结的新 KuaiSearch 4k 结果表明：Q0--Q3 的 overall 与 recurrence history gain 均为正且
区间不跨 0，strict-transfer 区间全部跨 0，W0 也没有建立 strict-transfer recovery。这个
观察不能区分数据可识别性、监督、历史读取、偏好抽象、候选对齐、训练目标、统计功效或人口
漂移。

本阶段目标是把这些解释拆成可证伪的竞争假设，用行为干预、表示诊断和训练动力学证据定位
主要瓶颈，并把结论转化为面向 CCF-A 级方法工作的架构设计约束。数据、监督与统计功效不作为
脱离模型的独立主线；分析数据构造时必须同时追问它暴露了什么架构限制、需要什么表示或路由
能力，以及训练信号应如何与架构共同设计。重点机制包括 history routing、跨 item-ID 偏好
抽象、candidate-conditioned readout 和训练目标 shortcut。完成点是一份机制结论与架构机会
矩阵；具体方法实现仍在机制证据收束后单独开展。

## 2. 不可越过的证据边界

- 第一轮 `protocol.yaml`、Q0--Q3/W0 checkpoint、config、score bundle、release lock 和结果
  报告均为冻结证据；不得覆盖或把机制阶段结果写回原 run。
- 机制开发、probe 选择和普通调参只使用 KuaiSearch train 与 train-only internal-dev。
- 已观察的 legacy 2k 与 new 4k 只能做描述性复核，不能再充当未见确认集，也不能依据其结果
  选择 probe、阈值、层、slice 或 checkpoint。
- source test 保持关闭。机制阶段不引入不同数据集来替代 KuaiSearch。
- 所有模型输入继续执行显式字段白名单；训练/打分代码不得读取 dev/confirmation/test qrels。
- 需要标签的评估仍由共享 evaluator 在 score integrity audit 通过后执行。
- 诊断性重训练必须保留原 checkpoint 与 matched control；不得把结果更好的诊断变体冒充
  第一轮 baseline 或论文方法。
- 可以为机制诊断或后续架构设计 train-only 的采样、配对、增强与辅助监督，但不得改变统一
  标准记录接口、确认人口、候选集合、qrels 边界或共享 evaluator。数据设计必须服务于明确
  架构假设，不能成为绕开模型问题的 method-only evaluation branch。

## 3. 竞争假设

| ID | 假设 | 支持证据 | 反证条件 |
|---|---|---|---|
| H0 | 当前可见历史与商品文本缺少足够、可学习的跨商品偏好信号，或 strict-transfer 测量功效不足 | 合法 positive control 也不能超过 null；oracle ceiling 很低；最小可检测效应大于合理效应 | 同一字段边界内的监督 probe 在未见请求上稳定恢复 strict-transfer signal |
| H1 | query-aware history selection 失败，相关历史被噪声淹没 | 删除高相关历史显著性弱，删除无关历史反而改善；相关历史注入不改变正确方向 | 保留/注入相关历史能稳定引起预期 score-margin 变化 |
| H2 | 模型依赖 item/text recurrence，没有形成不同-ID偏好抽象 | 不同-ID同属性替换破坏响应；层内表示不能解码历史偏好属性 | 表示 probe 可跨 item ID 解码偏好，且对语义保持干预稳定 |
| H3 | 偏好已进入表示，但未被候选比较或最终 readout 使用 | 表示可解码，full/null activation patch 有中介信号，但最终 target margin 不随之变化 | 表示变化能稳定介导候选相对分数和排序变化 |
| H4 | 训练目标被容易的 recurrence 样本或非个性化相关性捷径主导 | recurrence/transfer 梯度贡献失衡或冲突；train-only 平衡控制恢复 transfer response | 平衡采样、分层 loss 或去 recurrence 控制不改变 transfer 行为 |
| H5 | 观察主要来自 seed、人口漂移或测量不稳定，而非共同机制 | 诊断效应只在单人口/单 seed 出现，跨 cohort 方向不一致 | 预注册 probe 在独立请求簇和必要的第二 seed 上复现 |

这些假设不互斥。最终可以得到“多个瓶颈共同作用”或“证据仍不足”，不能为了得到单一故事而
删除矛盾结果。

## 4. 执行顺序

### M0：数据、信号与功效的架构关联审计

先排除明显的数据与测量问题，并把每项观察映射到可能的架构约束，避免把致命混淆误称为
模型机制，也避免把数据现象当作与模型无关的终点：

1. 审计 strict-transfer 请求的历史长度、query--history 语义相关性、brand/category/attribute
   可对齐率、正例等级、候选难度和 query-cluster 有效样本量；
2. 粗略计算当前方差下的最小可检测效应，确认现有 strict-transfer 人口不是显然无功效；
3. 建立一个不使用 raw item ID、只使用当前允许字段的轻量 train-only transfer probe；
4. 同时运行 label-shuffle、history-shuffle 和 query-shuffle negative controls；
5. 报告 oracle/probe ceiling，而不是只报告是否显著。

W0 的具体 CoPPS-style recipe 低优先级维护。positive control 可以是更简单、可解释的
transfer-only probe，不要求继续救 W0，也不要求把 recoverability 做成独立研究。其结果必须
继续追问：当前架构缺少哪种归纳偏置，以及何种数据构造只有与该架构机制结合才会有效。

M0 默认不阻断 M1--M3。只有发现标签泄漏、候选/时间边界错误、strict-transfer 有效样本近乎
不可用或字段内完全不存在可定义的偏好关系时才暂停。普通的弱 ceiling、宽区间或 probe 失败
记为竞争解释和论文边界，但继续开展架构导向的模型内部分析。

### M1：输入级可逆干预

在同一 checkpoint、query、候选集合、token budget 和打分参数下，对 Q0--Q3 运行统一的
请求级干预：

- 按 train-only 定义的 query relevance 保留 top-k 历史、删除 top-k 历史和删除 bottom-k
  历史；
- 注入等长的 query-relevant different-ID 历史、无关历史与 wrong-user 历史；
- 做 history order shuffle、history-query shuffle、title/brand/category 分字段 mask；
- 在 recurrence 请求上移除 exact target，在 strict-transfer 请求上做同属性不同-ID替换；
- 所有替换保持历史长度、可见字段和近似 token 数匹配，并审计 candidate leakage。

主行为端点为 strict-transfer target-versus-best-competitor score-margin change 和
`full - null` graded NDCG@10；recurrence、other-overlap 与 overall 用于解释副作用。每个预注册
干预全部报告，不按结果挑选 slice。

### M2：表示与中介诊断

便宜的输入 probe 先覆盖 Q0--Q3。深层 activation 分析预先从 Q2 与 Q3 开始：两者共享同一
General Qwen 初始化，但分别代表 full-parameter joint ranking 与 LoRA alignment，选择依据是
机制边界而非结果好坏。

1. 在各层提取 query、history summary 和 candidate scoring position 的表示；
2. 用 train-only 数据拟合跨 item-ID 的 brand/category/attribute preference linear probe；
3. 比较 full/null/relevant-only/irrelevant-only 的 layerwise representation shift；
4. 进行 full-to-null activation patch 或受控 mediation，定位偏好信息是否到达 readout；
5. probe 训练与模型评估人口隔离，报告随机标签和随机层基线。

若需要扩展到 Q0/Q1，必须使用同一 probe 定义并原样报告前两个 anchor 的结果，不能因方向
选择模型。

### M3：目标与训练动力学

在冻结训练人口上进行 matched diagnostic controls，不把它们称为新方法：

- 分别统计 recurrence、strict transfer 和 other-overlap 样本的 loss、梯度范数、方向余弦、
  更新占比与 learning-curve；
- 比较原始采样与预注册的 surface-balanced sampling/stratified loss；
- 做去 recurrence、transfer-only 和 query-matched hard-negative 小预算对照；
- 保持 backbone、初始化、总 optimizer step、可见字段和 evaluator 不变；
- 原 pilot seed 用于 matched comparison。若结论依赖重训练稳定性，在看结果前登记第二 seed，
  两个 seed 同时报告。

平衡控制若改善 strict transfer，只支持“目标/梯度分配是瓶颈之一”，不自动等于最终解决方案。

### M4：三角验证、机制判定与架构机会矩阵

一个主要机制结论至少需要：

1. 一个可逆的行为干预；
2. 一个独立证据源，例如表示中介、梯度动力学或 recoverability control；
3. 相应 negative control 通过；
4. 至少在两个独立 normalized-query 请求簇划分上方向一致。

如果证据只来自 post-hoc slice、单层 probe 或单个 checkpoint，结论必须标为 exploratory。
第一轮 4k 可以用于检查已经冻结的机制预测是否与旧观察相容，但不能恢复“未见确认”的身份。

机制证据收束后，将 H1--H4 映射为架构要求，而不是立刻拼装模块：

- routing failure 对应显式 query-conditioned history routing 与可审计稀疏选择；
- abstraction failure 对应去 item-ID 的 preference bottleneck、属性/意图因子化或跨商品原型；
- readout failure 对应 candidate-conditioned preference matching 与可传递的相对分数路径；
- objective shortcut 对应 transfer-aware pair construction、counterfactual supervision、分层梯度
  约束或 curriculum；
- 若多个环节同时成立，优先形成一个端到端可证伪的统一机制，而不是若干松散 trick。

数据可以与新架构共同设计，例如 different-ID preference pairs、query-matched hard negatives、
attribute-preserving counterfactual history 和 recurrence/transfer 分层 batch；但这些只能改变
train-only 学习信号，不能改变评测定义。最终机会矩阵需说明创新点、必要模块、训练信号、
可证伪预测、关键消融以及与 CoPPS/BATA/HMPPS/MemRerank 等工作的实质差异。

## 5. 统计与报告

- 继续使用 request-level 输出和 normalized-query cluster bootstrap；同时报告样本数、点估计、
  区间和 effect size，不把 `p > 0.05` 写成无效应。
- 多干预比较采用预注册层级：signal ceiling → input selection → abstraction/readout → objective。
  同一层内报告全部比较并控制 false discovery rate；探索性结果单列。
- 主要模型比较是 same-checkpoint paired intervention，不把不同 surface 的绝对 NDCG 难度
  当作机制差异。
- 机制 run 使用 `YYYYMMDD_kuaisearch_mech_<probe_id>_<purpose>`，每个输出目录独立可写，
  记录 config、code revision、数据/manifest hash、checkpoint identity、seed 和 qrels 边界。
- 原始 activation、gradient、score dump 与日志放在 `runs/` 或 `artifacts/`；跟踪目录只保留
  protocol、probe manifest、聚合数字和简洁结论。

## 6. 计算与阶段门

- 不设置数据分析、内部诊断和架构综合之间的固定时间或算力比例，按信息增益动态推进；
- 数据构造实验必须登记对应的架构假设、预期行为变化和反证条件，不能只比较数据 recipe；
- 可以并行推进 M0、无训练 M1 与小样本 instrumentation smoke，再按证据启动 activation dump
  或诊断训练；不等待 M0 形成完整独立结论；
- 单个连续 GPU job 仍不超过 4 小时，必须可 resume；存在独立 ready job 时使用四卡并行；
- 只有数据完整性或有效人口出现致命失败时才阻断内部归因；普通 recoverability 不确定不阻断；
- 任一机械失败、数值异常或 coverage 缺失只算工程状态，不算机制证据；
- 不按结果增加模型、数据集、seed、层或 slice。

## 7. 交付与停止点

本阶段交付：

- 一个冻结的 probe manifest 与输入/标签/候选完整性审计；
- signal ceiling、输入干预、表示/中介和梯度诊断的机器可读聚合结果；
- 一张 H0--H5 证据矩阵，逐项标记 supported、weakened、rejected 或 unresolved；
- 一张架构机会矩阵，明确最值得形成统一创新机制的瓶颈组合、训练数据需求和关键消融；
- `doc/motivation.md` 中的机制结论与下一步方法设计约束。

完成首轮机制判定后停止。未经新的用户指示，不实现 transfer 架构、不进入新数据集、不打开
source test，也不把诊断性平衡训练包装成论文方法。
