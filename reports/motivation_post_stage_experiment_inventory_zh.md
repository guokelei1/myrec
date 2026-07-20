# Motivation 之后的 LLM4Rec transfer 机制实验总盘点

盘点时间：2026-07-20 16:13（Asia/Shanghai）  
比较起点：Git 提交 `0f5c5be`，2026-07-17 11:44，提交说明 `motivation done`  
阶段状态：机制探索盘点完成；全部旧 GPU scorer、queue、watcher 和 resume loop 保持停止；
当前转入新架构开发准备  
本文性质：面向项目负责人的中文解释与状态快照，不替代冻结协议、机器指标或最终论文结果

## 1. 一页结论

motivation 阶段得到的冻结观察是：Q0--Q3 四种 history-conditioned Qwen 排序器都能从历史中
获得总体收益，但收益主要来自“历史商品再次出现”的 recurrence；在候选与历史商品完全不重合的
strict transfer 上，没有一个模型在新 4k 确认人口中建立可靠增益。这不等于“LLM 不可能迁移偏好”，
只说明当前四个冻结 recipe 在单 seed、KuaiSearch 回溯人口上的迁移尚未建立。

motivation 结束后，仓库做了两轮工作：

1. **M0--M4 首轮机制诊断（已完成）**：检查数据中有没有可迁移信号、模型是否选错历史、是否形成
   跨 item 表示、表示是否进入 readout、训练梯度是否被 recurrence shortcut 主导。结论是：没有一个
   单独解释成立。Q2 中存在局部 brand/category 可解码信号，Q2/Q3 的最终历史状态也确实被 readout
   使用，但会把关键 target margin 推向有害方向；简单筛选 relevant history 或做 surface balance
   都不是充分修复。
2. **D0--D7 Transformer deep-dive（部分完成，现已暂停）**：从 embedding、28 层 residual、attention、
   MLP、RMSNorm、RoPE、native readout、loss、optimizer 和 LoRA 路径继续拆组件。截至盘点，正式
   deliverable 完成 `7/19`，固定 D2 因果 bundle 完成 `60/60`，但 Q3 selected-branch 仍只有
   `2709/3918` 个 fold-1 请求，attention edge、MLP、RoPE/context、Q0/Q1/Q3 readout 和 optimizer
   replay 等关键闭环尚未完成。因此现在还**不能回答“到底是哪一个 Transformer 组件导致 transfer
   不好”**。

当前最稳妥的机制图景不是“历史没读到”或“偏好向量字面反转”，而是：

- history-conditioned 总位移在深层继续变大；
- 其中与 candidate-relative、brand/category transfer 有关的成分在中层较强、晚层相对衰减；
- Q2 尤其表现为大量 request-common 历史位移压过 candidate-relative selectivity；
- 最终状态既可能提高一部分 slate 的 NDCG，又同时把注册 target 相对强竞争者的 margin 推错，说明
  问题更像**候选特异的符号/校准/选择性失配**，而不是信息完全消失；
- Q3 并不完整复现 Q2 的 common-dominance 几何，故目前没有跨模型统一的单组件故事。

7 月 20 日之后又注册了 N8--N34 大量扩展问题。它们绝大多数是“计划或实现已存在、正式科学实验
未完成”，并已在 24 小时收口和 full-pause 决策中延期。不能把这些计划数当成已完成实验数。

## 2. 先把 Q0、Q1、Q2、Q3 讲清楚

Q0--Q3 是 motivation 阶段冻结的四个模型方法，不是四个机制实验编号。

| 编号 | 实际含义 | 训练/打分方式 | 在后续机制分析中的角色 |
|---|---|---|---|
| Q0 | `Qwen3-Reranker-0.6B` 专用 reranker | pointwise 二分类式排序；专用 reranker 预训练 | 专用 reranker 锚点；与 Q1--Q3 不构成完全 matched backbone |
| Q1 | InstructRec-style General Qwen | 全参数训练；一个 prompt 放完整候选 slate，以候选回复 likelihood 排序 | listwise prompt、KV-cache、多 token readout 的广度锚点 |
| Q2 | RecRanker-style General Qwen | 全参数训练；yes/no logit，RankNet + ListNet 联合目标 | deep-dive 主锚点；最适合检查全参数表示、readout 和 optimizer |
| Q3 | TALLRec-style General Qwen | 只在 Q/V projection 上做 rank-8 LoRA；完整 Yes/No 序列 likelihood | deep-dive 主锚点；用于与 Q2 比较低秩适配和多位置 native readout |
| W0 | CoPPS-style 非 LLM witness | 语义历史视图 + 对比训练 | motivation 的 transfer positive-control 尝试；未恢复可靠 strict transfer，后续低优先级 |

冻结的新 4k 确认结果中，Q0--Q3 的 recurrence `full-null` NDCG@10 增益约为
`+0.207` 到 `+0.257`，strict-transfer 增益只有 `-0.0016` 到 `+0.0100`，且四个 strict-transfer
区间全部跨 0。这就是后续所有机制实验要解释的起点。

常见术语：

- **recurrence**：正例商品已经在用户历史中出现，模型可以做 exact-item/text matching。
- **strict transfer**：正例不在历史中，且候选集合与历史 item ID 无交集；必须跨商品迁移偏好。
- **full/null/wrong-user**：真实历史、空历史、错误用户历史三种同请求对照。
- **target margin**：目标商品分数减去最佳低 gain 竞争者分数；比 NDCG 更直接地看关键相对排序方向。
- **smoke / identity gate**：只检查代码、hook、数值恒等和 resume，不是科学结果。
- **representation / descriptive**：说明某种几何或相关性存在，不等于该组件因果导致最终排名。
- **sufficiency patch**：把 full-history 状态写入 null-history recipient，看该状态是否足以重现行为。
- **necessity/removal**：从 full 路径移除或替换组件状态；比单向 sufficiency 更接近“这个组件是否必要”。

## 3. 时间线与总体完成度

| 时间 | 事件 | 状态 |
|---|---|---|
| 7 月 17 日 11:44 | motivation 第一轮冻结完成 | 基线起点 |
| 7 月 17 日 | M0--M3 数据、输入、表示、patch、梯度和 matched-control 全部执行并综合 | 已完成 |
| 7 月 17 日晚 / 18 日 | 用户授权四卡 Transformer 深挖；D0--D7 manifest 冻结 | 已授权并启动 |
| 7 月 18--19 日 | D1 全层表示、D2 全层 post-block、部分 D3/D6/D7 与 17 项补充几何完成 | 部分完成 |
| 7 月 19--20 日 | component necessity、comprehensive report、N8--N34 计划与大量实现继续扩展 | 多数仅注册/实现 |
| 7 月 20 日 | 先执行 24h triage，只保留高进度 Q3 shard；随后用户要求 full pause | 全部停止，保留可恢复状态 |

当前自动审计快照：

| 口径 | 当前值 | 正确解释 |
|---|---:|---|
| M0--M4 首轮机制诊断 | 完成 | 已形成冻结 H0--H5 矩阵和架构机会排序 |
| deep-dive 正式 deliverable | `7/19 = 36.8%` | 按产物闭环计，不是按 GPU 时间计 |
| D2 固定因果 bundle | `60/60` | 固定层/条件全部完成 |
| D2 含条件 selected branch | `61/62 = 98.4%` 单元 resolved | Q2 完成，Q3 部分完成；按请求加权总执行约 `99.5%` |
| 正式 run registry | 107 个声明；100 完整且过 integrity，6 个机械失败，1 个仍登记 running | full pause 后没有进程；部分 metadata 故意保留 `running/resumable` |
| supplemental registry | `17/21 = 81.0%` | 多数是描述性几何，不等于 17 个因果结论 |
| component artifact | `24/40 = 60%` | 18 个组件中 16 个至少有一种产物，但只有 5 个有已完成 causal-role 产物 |
| comprehensive report 合同 | `3/12` 前置项完成 | 最终综合报告尚不能生成 |

自动 closeout 当前显示 `failed`，含义不是“科学实验失败”，而是：还有 13 个 pending condition，且
`d6_q1_branch_b13_v1` 的 terminal metadata 缺少绑定的 failure record。这个审计状态应理解为
**收口不完整**。

## 4. M0--M4：首轮机制诊断逐项盘点

### 4.1 M0：数据、信号和统计功效

**想回答什么**：当前字段里到底有没有跨商品偏好信号？strict-transfer 人口是否小到根本测不出？

**预期判别**：如果合法的 ID-free 轻量 probe 都不能优于 null，或合理效应小于最小可检测效应，H0
“信号/功效不足”会加强；若 probe 能稳定恢复，则 H0 会被削弱。

**实际设计**：审计 2,195 个 internal-dev strict-transfer 请求；统计历史长度、query-history
相关性、brand/category 对齐、候选难度和 normalized-query cluster；训练不使用 raw item ID 的
线性 recoverability probe，并运行 label/history/query shuffle 负控。

**状态与耗时**：正式综合已完成。M0 多个分析 metadata 没有完整 `elapsed_seconds`，可核实的登记
仅约 `0.01` 进程小时，因此无法诚实给出总计算耗时。

**得到的结果**：可见字段的语义对齐不完整，但不是完全没有 brand/category 关系；当前人口对
约 `0.02` 的效应有合理功效。轻量 probe 没有建立正的 `full-null` recoverability ceiling，且不同
endpoint/负控方向不完全一致。因此 H0 仍是 `unresolved`，但“字段里完全无信号”被 Q2 后续局部
可解码结果削弱。

### 4.2 M1：输入级历史干预

**想回答什么**：是不是 query-aware history selection 失败，相关历史被无关历史淹没？模型是不是
只认 exact recurrence，而不能接受同属性不同 ID 的历史？

**预期判别**：relevant-only 应优于 full/irrelevant；语义保持的 different-ID 替换应保留效果；
语义破坏或 overlap removal 应按预期改变 margin。

**实际设计**：Q0--Q3 统一跑 relevant-6、irrelevant-6、order shuffle、semantic-preserving、
semantic-breaking、candidate-overlap swap 等输入干预，共形成预注册 48-hypothesis family，并做
token 长度/截断审计。

**状态与耗时**：正式 score/evaluation 已完成；25 个带 metadata 的 M1 run，累计记录约
`29.45` 进程小时。四卡并行，不能换算成 29.45 小时日历时间。

**得到的结果**：relevant-only 从未可靠优于 frozen full；Q0/Q2 的 relevant-vs-irrelevant NDCG
没有通过 family correction，Q0 margin 甚至显著反向。different-ID semantic preservation 也没有
形成跨模型一致的正收益。由此 H1“筛掉无关历史即可解决”被削弱；H2 的“可用跨 ID 抽象不足”仍
有行为层支持，但不能证明模型只依赖 exact recurrence。history order 有孤立效应，但不是统一解释。

### 4.3 M2：表示、层定位和 activation patch

**想回答什么**：偏好信息是否进入 Transformer 表示？如果进入了，是在中间丢失，还是到了最后
却没有被正确 readout？

**预期判别**：真实 brand/category probe 应优于 random label；same-request full-to-null patch
应改变目标 margin，且优于 identity/cross-request donor。

**实际设计**：Q2/Q3 在 state 0/7/14/21/28、query/history-summary/candidate-readout 三类位置提取
表示；训练 brand/category 线性 probe；在 block 13/27 做 same、cross、identity activation patch。

**状态与耗时**：正式表示与 patch 综合已完成；26 个 M2 metadata run 中 22 completed、4 个机械
失败，累计记录约 `13.43` 进程小时。早期 tokenizer offset、raw-query hash bug 等失败均被修复后
重跑，不能作为科学结果。

**得到的结果**：

- Q2 中后层能局部解码 brand/category proxy，Q3 没有以同样方式稳定复现；这削弱“完全没有抽象”。
- Q2/Q3 的正确 block-27 full state 都几乎完整重现 full-history 的负 target margin，说明最终历史
  状态被 readout 使用了，但使用方向有害；所以“信息进入但完全没被用”也被削弱。
- block 13 的 correct state 会把 margin 推到 null 之上，但 Q2 的 cross-request donor 推得更远，
  因而不能称为用户偏好恢复。
- 最窄解释是：中层存在可用方向的历史响应，晚层形成有害 endpoint；但 patch 的是混合 post-block
  state，尚不能归因到 attention、MLP、residual 或 RMSNorm。

### 4.4 M3：loss、梯度与 matched training control

**想回答什么**：是不是 recurrence/easy surface 吃掉了训练预算，或者不同 surface 梯度互相冲突？

**预期判别**：recurrence 梯度 mass 应显著占优或与 strict transfer 冲突；固定步数的 surface-balanced
训练若改善 strict transfer，则说明目标分配是瓶颈之一。

**实际设计**：Q2/Q3 在 base/final checkpoint 上按 recurrence、strict-transfer、other-overlap
各取固定请求，计算梯度 norm/cosine/mass 与 label-shuffle；Q2 从同一初始化做 256-step original
mixture 与 surface-balanced matched control，并用 DID 比较。

**状态与耗时**：正式诊断和 matched control 已完成；14 个 M3 metadata run 累计记录约
`3.80` 进程小时。训练 supervisor 曾遇到 checkpoint 目录契约错误，但复用已完成 checkpoint 后
修复，没有重训或改变结果。

**得到的结果**：final checkpoint 上 Q2/Q3 都出现 narrow other-overlap 梯度冲突，但 recurrence
mass dominance 只在 Q3 明显；recurrence 与 strict-transfer 梯度本身在两个 anchor 上大体同向。
surface-balanced Q2 没有可信 NDCG DID，target-margin DID 反而稳定恶化。因此 broad H4 被削弱，
“全量 surface balance”被反证为直接修复，optimizer-aware 的实际 update 归因仍未完成。

### 4.5 M4：首轮综合

**状态**：已完成，报告为 `first_mechanism_diagnosis_complete`。

| 假设 | 首轮状态 | 通俗解释 |
|---|---|---|
| H0 信号/功效不足 | unresolved | 数据信号有限且 probe ceiling 不高，但不能说完全无信号 |
| H1 history routing 失败 | weakened | relevant filtering 不是充分修复；内部 attention routing 尚未测清 |
| H2 没有跨 ID 抽象 | weakened | Q2 有局部可解码性，但未建立跨模型、request-specific 的可用抽象 |
| H3 信息进入但 readout 不用 | weakened | 最终状态被用了，只是符号/校准有害 |
| H4 recurrence 梯度 shortcut | weakened | 只有窄冲突；简单 balance 反而伤害 margin |
| H5 seed/cohort 不稳定 | unresolved | 只有一个训练 seed，Q2/Q3 异质性仍明显 |

首轮提出的首选架构机会是：ID-free factorized preference state + 显式 signed candidate residual +
abstention gate。它是由 H2/H3 约束推导出的候选设计，不是已验证方法。用户已于 2026-07-20
授权把它收束为 Candidate-Contrast Personalization 开发计划；授权不追溯改变本节实验当时的状态。

## 5. D0--D7 Transformer deep-dive 逐项盘点

### 5.1 总表

| 阶段 | 计划回答的问题 | 希望看到的判别结果 | 当前状态 | 已得结论 |
|---|---|---|---|---|
| D0 instrumentation | hook 是否精确覆盖 28 层、attention/MLP/residual/RMSNorm/readout，且不改原分数 | no-op 与原 scorer 误差 `<=1e-5`，代数重组过低精度界 | 工程主体完成；有 superseded mechanical failures | 只证明测量链可用，不产生 transfer 结论 |
| D1 29-state representation | 偏好 proxy 在哪一段增强/衰减，full/null 差异如何演化 | real-label 超过 random，full-null excess 在两 fold 同向 | **正式完成** | 中层 task-aligned 成分较强，晚层相对衰减；总 history delta 却继续增大 |
| D2 causal sweep/branch | endpoint 符号在哪个相邻层改变，attention/MLP/norm 哪个节点承担变化 | 相邻 transition 独立 fold 复现；same 同时优于 cross/wrong/random/scale controls | 固定 sweep 完成；Q2 branch 完成；Q3 branch `2709/3918`，暂停 | Q2 选 block 15、Q3 选 block 17；Q2 尚未通过 history-specific 单组件闭环 |
| D3 attention | history formation/transport 是否由特定 head/edge/QKV 路径造成 | joint causal edge 干预改变 margin/NDCG，并通过 identity 与负控 | head observation 完成；edge/group formal 未完成 | 目前只有描述性 head/GQA 几何，不能归因 attention |
| D4 MLP | SwiGLU group/MLP branch 是否形成或覆盖有用偏好成分 | full branch/formation intervention 在 Q2/Q3 复现，并通过结构负控 | formal deliverable `0/1`；若干 group/formation partial 或 smoke | 尚无可用 MLP 因果结论 |
| D5 context/RoPE | full/null 长度和相对位置是否造成假信号，RoPE 压缩/扩张是否有方向效应 | compression 区别于等幅 expansion，NDCG 不落入 `±0.005` 等价带 | formal `0/2`；描述性 position/RoPE geometry 完成 | 已确认 full/null 存在位置差，但未证明位置因果造成 transfer gap |
| D6 native readout/Q0/Q1 | final RMSNorm、yes/no/log-prob readout 与 Q0/Q1 路径是否造成错配 | same/cross 分离；input/output 或 native term 定位符号变化 | 仅 Q2 formal 完成；Q0 b20 `1996/8000`，Q0/Q1/Q3 其余延期 | Q2 final norm input/output patch 相同；最终状态可增 NDCG 却伤 target margin |
| D7 loss/optimizer/LoRA | RankNet/ListNet、AdamW 实际 update、LoRA path 是否导致瓶颈 | objective cosine 冲突或 effective update/LoRA 方向与行为对应 | Q2 objective、Q3 LoRA path 完成；optimizer replay 未完成 | Q2 RankNet/ListNet 高度同向；LoRA 几何仅描述，尚无 optimizer 因果归因 |

### 5.2 D1 已完成的关键观察

candidate-readout 的 category `full-minus-null excess` 在中层达到峰值、晚层下降：Q2 四区间约
`0.025/0.165/0.133/0.025`，Q3 约 `0.033/0.225/0.145/0.093`。但 strict-transfer candidate
readout 的 full-null L2 从 state 13 到 state 28，Q2 放大约 `26.8x`、Q3 约 `12.0x`。

这意味着“所有历史信息被擦掉”不成立。更准确的是：总 history-conditioned state 增长，但其中
task-aligned、candidate-relative 的特殊成分被稀释、旋转或被更大分量覆盖。Q2 晚层更新约
`97.35%` 的能量属于 candidate-common 分量，common 与 relative change 相差约 `51.9x`；Q3 不
完整复现这一模式。静态 candidate embedding 的 full-null 差精确为 0，个性化差异是在 Transformer
上下文化后产生，不支持把静态 item embedding 当作主要根因。

### 5.3 D2 已完成和未完成的边界

Q3 all-native-position gate 已通过：block 13 的 strict-transfer margin patch-null 为
`+0.00285 [0.00074, 0.00506]`，block 27 为 `-0.01478 [-0.01791, -0.01174]`；
`all-native - first-position-only` 约 `1e-8`，所以首轮只 patch 第一个 native position 不是该符号
差异的解释。

全层因果 sweep 在独立 fold 定位到：

- Q2：block `14 -> 15`，相邻效应均值 `-0.01134`，combined BH q 约 `0.0028`；
- Q3：block `16 -> 17`，相邻效应均值 `-0.00920`，combined BH q 约 `0.0028`。

这定位了“endpoint effect 发生负转折”的功能深度，但不等于 activation/preference vector 自身
反向。Q2 block 15 的七节点 branch 已评估；例如 block-output same patch 的 target margin 为负，
但 same-minus-wrong-history 为正，不满足预注册 harmful history-specific mediator 的同向三门。
因此不能把 Q2 block 15 的某一个 attention/MLP/norm 节点定为根因。Q3 branch 尚未完成，不能做
跨模型确认。

### 5.4 D6/D7 已完成的窄结论

Q2 final RMSNorm input 与 output 的 same patch 结果完全相同：strict-transfer NDCG
`+0.01173 [0.00373, 0.01925]`，target margin 却为
`-0.03137 [-0.04079, -0.02183]`。这排除了“该干预下最后一次 RMSNorm 新制造符号变化”的窄解释，
但不排除 RMSNorm 在其他方向/尺度干预中的作用。它也说明 NDCG 与注册 target margin 可以同时给出
一正一负，不能用单一 overall utility 掩盖候选级错配。

Q2 RankNet/ListNet 每请求梯度 cosine 在 base/final 和三个 surface 上约为 `0.89--0.98`，没有达到
冲突 SESOI；因此 Q2 两个 loss term 直接互相打架不是当前证据支持的解释。AdamW moments、clip、
weight decay 和 scheduler 后的实际 delta 尚未完成 replay。Q3 LoRA A/B、`B@A`、SVD/head concentration
已经描述，但没有 inference-time branch necessity，因此不能说某个 LoRA rank/head 导致 transfer 差。

## 6. 21 项 supplemental evidence 的状态

这些补充项主要利用已经生成的 activation/gradient/parameter 产物做离线几何分析，计算便宜、覆盖
广，但多数只能提高或降低某个解释的优先级，不能替代正式 causal intervention。

**已完成 17 项**：embedding-readout geometry、activation anisotropy、candidate block flow、
candidate residual geometry、query causal floor、preference subspace、RMSNorm flow、attention
pattern、full/null position shift、QK stage geometry、RoPE position geometry、frozen logit lens、
objective common nullspace、Q2 objective family shares、Q2 parameter-update geometry、Q2 update
anisotropy、Q3 LoRA head geometry。

**仍 pending 4 项**：

1. D4 MLP feature-formation extension；
2. D6 native-readout diagnostics；
3. component-state reverse necessity V2；
4. component functional design-gate synthesis。

补充项的共同结论是把“字面信号反转”收紧为“task-aligned/candidate-relative 成分相对衰减或被
common/off-task 分量覆盖”；但 attention、MLP、RMSNorm、RoPE 或 optimizer 中没有任何一个仅凭
这些描述性结果获得单独根因资格。

## 7. Component necessity 与最终综合报告

component-necessity V1 在任何正式 score 启动前发现位置混杂，被 V2 取代。V2 计划在相同 full
recipient 中用等长 content-neutral donor 做反向状态移除，要求 sufficiency、wrong-history
specificity 和 reverse necessity 同时通过，才能称为 history-specific mediator。

当前 V2 尚未完成，四个 supplemental pending 项之一就是它。最终 comprehensive report 要求：19 个
formal deliverable、21 个 supplemental、D2 全部 terminal、18 组件覆盖、无审计失败、人工 decision
worksheet、机器 JSON 和 13-section Markdown 全部闭环。当前只完成 `3/12` 个前置要求，最终报告
尚不存在，也不应强行生成一个“已经找到根因”的版本。

## 8. N8--N34：扩展计划到底是什么、现在做到哪

N 系列是 7 月 20 日继续扩展的 operator-level 诊断。当前 full-pause 的统一解释是：除少量 smoke/
实现检查外，均未形成正式科学结果；不得自动恢复。

| 编号 | 计划问题与期望结果 | 当前真实状态 |
|---|---|---|
| N8 | attention × MLP 联合移除，区分相加、抵消或非线性 residual interaction | 已计划/有 queue；24h triage 延期，未形成正式结果 |
| N9 | 把 history formation、history→readout transport、candidate readout 串联干预 | 已有 manifest/实现/queue；延期，未形成正式结果 |
| N10 | Q3 全 28 层 LoRA rank-path ablation；四模型 native candidate-gap perturbation | 两套 manifest/实现已存在；延期，未形成正式结果 |
| N11 | 直接干预 scaled QK attention logits，区分 Q/K 几何与 softmax 前路由 | 仅出现 Q3 b13 CPU smoke/partial；不是科学结果，已暂停 |
| N12 | 分离 MLP gate/up/SiLU/SwiGLU/product 各阶段 | 仅出现 Q2 b13 CPU smoke；不是科学结果，已暂停 |
| N13 | 分离 Q/K/V projection stage | 有 Q3 b13 CPU smoke 等工程产物；未完成正式 family |
| N14 | history token embedding stage 干预 | 有 Q2 b13 CPU smoke；未完成正式 family |
| N15 | attention/MLP increment 与 residual composition | manifest/queue 已写；未正式完成 |
| N16 | RMSNorm variance rescale 与 learned gain 分离 | manifest/queue 已写；未正式完成 |
| N17 | Q/K head RMSNorm operator | 预注册 inactive；triage 延期 |
| N18 | GQA repeat-KV grouping boundary | 预注册 inactive；triage 延期 |
| N19 | Q3 完整 scaled Q/V LoRA adapter branch | 预注册 inactive；triage 延期 |
| N20 | Q1 prefix/continuation KV-cache phase boundary | 预注册 inactive；triage 延期 |
| N21 | FP32 LoRA adapter / BF16 base cast boundary | 明确属于 later batch，未授权正式运行 |
| N22 | LoRA input dropout boundary | later batch，inactive |
| N23 | gradient-checkpoint bridge / recomputation / RNG 等价 | later batch，inactive |
| N24 | objective 与 optimizer effective-update training boundary | later batch，inactive |
| N25 | 完整 SwiGLU formation/nonlinearity，含 gate/up/product reverse removal | 曾纳入 closeout batch，但 full pause 后延期；未完成 |
| N26 | final RMSNorm/native readout 的整合复核，必要时做最小 variance-vs-gain replication | 只应整合 D6，不应重复大 sweep；当前未闭环 |
| N27 | causal mask 与 softmax topology | 明确延期，formal GPU 未授权 |
| N28 | 完整 scaled pre-mask QK tensor formation | 明确延期，formal GPU 未授权 |
| N29 | attention--MLP `2x2` factorial non-additive interaction | 明确延期，formal GPU 未授权 |
| N30 | query/history/candidate token embedding interface 因果干预 | preparation 可保留，formal 未授权 |
| N31 | 两个 block RMSNorm 的 variance/gain operator | preparation 可保留，formal 未授权 |
| N32 | attention/MLP residual addition coefficient | preparation 可保留，formal 未授权 |
| N33 | GQA query-head 到 shared-KV 的映射拓扑 | preparation 可保留，formal 未授权 |
| N34 | 整合 N19 的 Q3 adapter contribution，避免重复 sweep | preparation 可保留，formal 未授权 |

这里最值得注意的是重复与膨胀：N8 与 N29 都涉及 attention--MLP composition；N12/N25 都继续拆
SwiGLU；N18/N33 都涉及 GQA；N19/N34 都涉及 Q3 adapter；N26 与 D6 重叠。后来的计划已尝试把
其中一部分定义为“整合而非重跑”，但在下一次恢复前仍应先合并问题清单，避免把同一科学问题变成
多套昂贵 family。

## 9. 计算与时间花费

从 motivation 基线提交到本次盘点约经过 **76.5 小时日历时间**。仓库 323 个相关 run metadata
中，254 个记录了 `elapsed_seconds`，合计约 **277.38 进程小时**：其中约 269.32 小时来自 metadata
标为 completed 的进程，7.50 小时来自仍保留 running/resumable 状态的进程，约 0.54 小时来自记录
了 elapsed 的机械失败。四卡和 CPU 任务大量并行，所以进程小时明显大于日历小时。

这个数字是**下界且不是纯 GPU 小时**：69 个 metadata 没有 elapsed；部分 evaluator/离线分析是
CPU；部分 retry/smoke 和 partial 也计入；训练或 shell supervisor 的全部时间不一定写入同一字段。
不能据此精确核算电费或 A40 GPU-hour。

按 run 名称粗分的已记录进程小时如下：

| 阶段 | metadata run 数 | 已记录进程小时 | 备注 |
|---|---:|---:|---|
| M0 | 9 | 0.01 | 大部分缺 elapsed，不能代表真实耗时 |
| M1 | 25 | 29.45 | 四模型多条件 scoring 是主要成本 |
| M2 | 26 | 13.43 | 含 4 个机械失败记录 |
| M3 | 14 | 3.80 | 部分训练/supervisor 时间可能未完整记录 |
| D0 | 16 | 0.00 | 多数 smoke 未写 elapsed |
| D1 | 14 | 3.72 | 全层 activation 大量成本可能记录在上游/分片方式中 |
| D2 | 77 | 159.55 | 当前最大成本；全层 patch 和 selected branch |
| D3 | 30 | 30.42 | attention observation/edge/group |
| D4 | 13 | 1.76 | 多数正式工作尚未完成 |
| D5 | 18 | 26.74 | RoPE/context，formal 尚未闭环 |
| D6 | 18 | 8.08 | Q0/Q1/Q3 大量延期 |
| D7 | 10 | 0.34 | 多数离线/缺 elapsed；optimizer replay 未完成 |
| N 系列 | 12 | 0.07 | 基本只是 smoke，说明正式 next-wave 几乎没跑 |

暂停时三个关键 partial：Q3 selected shard0 `1290/1959`、shard1 `1419/1959`，Q0 b20
`1996/8000`。metadata 仍写 `running` 是为了可恢复性；实际没有活跃进程。

## 10. 到目前为止，哪些问题已经回答，哪些没有

### 已经比较可靠地回答

1. **不是简单“模型没用历史”**：recurrence 很强，晚层 full state 能因果重现历史行为。
2. **不是简单“字段里完全没 transfer 信号”**：Q2 有局部可解码 brand/category proxy，但信号有限、
   不稳定且不自动可用。
3. **不是 relevant-history filtering 一招就能解决**：M1 不支持。
4. **不是 full surface balancing 一招就能解决**：Q2 matched control 反而伤害 target margin。
5. **不是 Q3 patch 位置不完整造成 block13/27 假反转**：all-native gate 复现了正到负 endpoint transition。
6. **Q2 的最终问题具有候选级错配**：NDCG 可改善但 target margin 变坏；common 与 relative 信息必须
   分开看。

### 尚未回答

1. attention output、MLP output、residual composition 中到底哪一个是必要且 history-specific 的根因；
2. Q2 block 15 与 Q3 block 17 是否存在跨模型共同的功能节点；
3. attention 是 formation 失败、transport 失败，还是传输了过多通用/无关内容；
4. SwiGLU/RMSNorm/RoPE/causal mask/GQA/LoRA adapter 中是否有 operator-level 必要机制；
5. AdamW moments、clipping、weight decay 后的真实 effective update 是否改变 M3 梯度故事；
6. 这些结论在第二训练 seed、forward temporal population 或其他模型规模上是否稳定；
7. factorized signed preference path 是否真的能改善 transfer——它尚未实现，也没有方法实验。

## 11. 当前最合理的项目决策边界

机制探索阶段的最终状态是：**首轮机制诊断完成；Transformer 组件 deep-dive 以部分证据快照收口；
最终组件证据矩阵未完成，且不再作为方法开发前置。** 不应称为“已经找到
attention/MLP/RoPE/LoRA 根因”，也不应把 N 系列的代码量当作证据量。

Q3 selected shards、Q0 b20、comprehensive closeout 和 N8--N34 现在统一属于非活动归档；不得因其
已有较高进度或已有代码而自动恢复。只有未来出现一个明确阻碍架构判断的问题，并获得新的用户
授权，才从中选择最小、不重复的诊断。

## 12. 新架构开发交接

用户已于 2026-07-20 明确授权进入方法实现与验证。新阶段不把未完成 deep-dive 包装成完整根因，
而使用已经足够稳定的设计约束：保留 query relevance，显式隔离 history-conditioned 更新，限制其
通过 candidate-contrast、signed、可 abstain 的路径写回 Transformer 候选状态。

执行入口为
[`../experiments/motivation/candidate_contrast_architecture_plan.md`](../experiments/motivation/candidate_contrast_architecture_plan.md)。
计划依次包含设计/数值合同、零训练机制门、参数高效原型、matched 正式微调以及冻结后的同协议
benchmark。当前尚未实现新模型、启动训练或生成新结果；状态是 `ready_for_implementation`。

## 13. 权威来源索引

- motivation 冻结观察与边界：[`../doc/motivation.md`](../doc/motivation.md)
- motivation 第一轮机器摘要：[`motivation_current_summary.json`](motivation_current_summary.json)
- M0--M4 机制计划：[`../experiments/motivation/mechanism_analysis_plan.md`](../experiments/motivation/mechanism_analysis_plan.md)
- 首轮机制诊断：[`motivation_mechanism_first_diagnosis.md`](motivation_mechanism_first_diagnosis.md)、
  [`motivation_mechanism_first_diagnosis.json`](motivation_mechanism_first_diagnosis.json)
- Transformer deep-dive 主计划与 manifest：
  [`../experiments/motivation/transformer_deep_dive_plan.md`](../experiments/motivation/transformer_deep_dive_plan.md)、
  [`../experiments/motivation/transformer_deep_dive_manifest.yaml`](../experiments/motivation/transformer_deep_dive_manifest.yaml)
- 当前衰减解释边界：
  [`../doc/dev_log/2026-07-18_transformer_signal_attenuation_interpretation.md`](../doc/dev_log/2026-07-18_transformer_signal_attenuation_interpretation.md)
- 24 小时收口与 full pause：
  [`../experiments/motivation/transformer_24h_triage_v1.md`](../experiments/motivation/transformer_24h_triage_v1.md)
- 当前批次边界：
  [`../experiments/motivation/transformer_current_batch_freeze_v1.md`](../experiments/motivation/transformer_current_batch_freeze_v1.md)
- component necessity V2：
  [`../experiments/motivation/transformer_component_necessity_extension_plan_v2.md`](../experiments/motivation/transformer_component_necessity_extension_plan_v2.md)
- final report 合同与 supplemental registry：
  [`../experiments/motivation/transformer_comprehensive_report_plan.md`](../experiments/motivation/transformer_comprehensive_report_plan.md)、
  [`../experiments/motivation/transformer_supplemental_evidence_registry.yaml`](../experiments/motivation/transformer_supplemental_evidence_registry.yaml)
- 当前架构开发计划：
  [`../experiments/motivation/candidate_contrast_architecture_plan.md`](../experiments/motivation/candidate_contrast_architecture_plan.md)
- 阶段切换记录：
  [`../doc/dev_log/2026-07-20_transfer_architecture_transition.md`](../doc/dev_log/2026-07-20_transfer_architecture_transition.md)
