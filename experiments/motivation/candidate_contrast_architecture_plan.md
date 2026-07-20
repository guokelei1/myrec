# Candidate-Contrast Personalization 架构开发与验证计划

状态：`ready_for_implementation`  
授权日期：2026-07-20  
阶段目标：把已经完成的 Motivation 机制证据转化为一个可训练、可消融、可与冻结 Q0--Q3
公平比较的 Qwen3-0.6B 个性化重排方法。

## 1. 已知起点与方法假设

冻结基线已经建立 recurrence-dominant history use；后续 M0--M4 与部分 Transformer deep-dive
进一步表明：历史并非没有进入模型，Q2 中后层也存在局部可解码的 brand/category 偏好代理，
但 task-aligned、candidate-relative 成分在深层相对衰减，最终历史状态会以错误符号或校准进入
候选比较。Q2 尤其出现很强的 request-common 历史位移；该几何尚未在 Q3 上完整复现，因此不能
把某个绝对层或单个 Transformer 组件称为普遍根因。

本阶段检验以下可证伪方法假设：

> 个性化历史更新应在 Transformer 内部显式分离为候选公共分量与候选相对分量，并只通过
> 可正可负、可关闭的 candidate-contrast 路径写回候选状态；这样可以保留 query relevance，
> 同时减少无区分或错误校准的历史更新对 strict transfer 的伤害。

当前工作名为 **Candidate-Contrast Personalization（CCP）**。名称是内部代号，不预先主张
论文命名或首创性。

## 2. V0 架构合同

V0 必须融入 Qwen Transformer，而不是在模型外另挂一个独立 rerank 小评分器。设计冻结前至少
满足以下合同：

1. **共享 Qwen backbone**：从与 Q1--Q3 相同的 `Qwen3-0.6B` 权重初始化；Q2 是主要 matched
   anchor，冻结 Q0--Q3 checkpoint 和结果不被覆盖。
2. **slate-aware candidate states**：同一请求的候选以带 mask 的候选维共同参与个性化更新；
   可使用并行候选分支或 Transformer 内部的联合候选接口，但不得依赖固定候选数。
3. **query-conditioned history state**：历史只读取允许字段，不使用 raw item ID；偏好 slot 是
   可选实现原语，不作为单独创新点。
4. **candidate-contrast writeback**：对历史产生的候选更新 `U_hist` 使用带 mask 的中心化算子

   \[
   U_{rel}=\left(I-\frac{1}{C}\mathbf 1\mathbf 1^\top\right)U_{hist},
   \]

   再通过有界 gate 写回候选残差流。实现必须对候选置换等变，并对 padding 候选严格无响应。
5. **null/abstention path**：无历史或历史无关时可以关闭 preference path；关闭后应精确暴露
   非个性化 query--candidate backbone，而不是产生另一个任意偏置。
6. **native ranking readout**：最终排序仍由项目内 Qwen 表示和共享 ranking objective 产生；
   不以外部独立 scorer 代替 Transformer 内部方法。
7. **可干预性**：history state、common/contrast 分解、gate 和 writeback 必须可独立置零、记录和
   patch，以便验证方法是否按设计工作。

内部投影只能声称阻止**直接的 candidate-common history writeback**；经过后续非线性层后仍可能
重新形成公共分量，因此不得夸大为对最终 hidden state 或 score 的全局数学保证。

## 3. 最小开发顺序

### A0：设计冻结与单元合同

- 冻结候选张量形状、mask、插入接口、loss、参数量、训练预算和 run manifest；
- 为候选置换等变、masked mean、全 padding 拒绝、null identity、有限值和梯度流添加手算测试；
- 通过原 Qwen scorer 的 identity/no-op 一致性门。

完成标准：代码 smoke 和数值合同通过，但不产生科学结论。

### A1：零训练机制可行性门

- 冻结现有 Q2 checkpoint，在预先固定的内部接口施加 candidate-contrast intervention；
- 只使用 train/internal-dev，比较 full、null、wrong-user 下的 score、target margin 与
  candidate-common/relative energy；
- 不调层、不按结果选择 gate 或投影强度。

完成标准：确认算子没有机械破坏 ranking，并判断现有表示中是否存在可保留的 candidate-relative
信号。A1 是诊断，不是论文方法；即使无提升，也不自动终止可训练版本。

### A2：参数高效原型

- 从 Qwen3-0.6B 初始化，训练新增 history/contrast/gate 参数，并在固定层组加入 LoRA；
- 使用与 Q2 相同的训练人口、字段边界和近似曝光预算；
- 只在 internal-dev 选择一次预先登记的结构候选，不访问确认 qrels。

完成标准：训练、resume、score export 和共享 evaluator 全链路完成；不存在退化常数分数、NaN、
候选缺失或 null path 漂移。

### A3：matched 正式微调

- 在架构和超参数冻结后，从原始 Qwen3-0.6B 权重重新训练，不从结果更好的 Q2 checkpoint
  warm-start；
- 主比较保持 Q2 的数据、seed、训练轮数/步数、候选合同和共享 evaluator；
- 第一 seed 完整保留，无论结果正负；第二 seed 是否启动及其固定 recipe 在看第一 seed正式
  confirmation 结果前登记。

完成标准：形成可复现 checkpoint、完整 lineage、训练曲线、参数量/吞吐与所有 benchmark
score bundle。

### A4：冻结后的 benchmark 与最小消融

统一运行与 Motivation 相同的：

- `full`、`null`、`wrong-user`；
- overall、recurrence、strict transfer、other overlap；
- graded NDCG@10、target margin、normalized-query cluster bootstrap 和人口加权贡献；
- candidate/request hash、完整有限 score coverage 与共享 evaluator 门禁。

主比较是 CCP 对 Q2 的 paired request-level 差异。安全指标包括 recurrence 保留、overall、
other-overlap、null-history absolute quality、训练稳定性和推理成本。最小消融只保留：

1. 无 candidate-contrast 投影；
2. 无 null/gate；
3. preference path 不写回 Transformer、仅接外部 head 的控制版本。

不得按结果继续增加层、slot 数、loss、seed、slice 或 endpoint。

## 4. 数据和证据边界

- 开发和调参只使用 KuaiSearch train 与 label-free internal-dev；输入字段白名单保持不变。
- legacy 2k 和 new 4k 已被观察，只能在方法冻结后按原合同做**描述性复核**，不能重新称为未见
  confirmation，也不能用于选层、选 checkpoint 或调阈值。
- source test 保持关闭；本阶段不换数据集。若未来需要论文级未见确认或 forward temporal
  证据，必须另行获得用户授权并在任何结果访问前冻结协议。
- scorer/trainer 不读取 internal-dev、confirmation 或 test qrels；共享 evaluator 只在 score
  integrity 通过后读取相应 qrels。
- 所有有效 seed、失败方向和退化结果都保留。机械失败与欠收敛必须单独标注，不能当 transfer
  结果。

## 5. 成功、失败与停止规则

本阶段目标不是保证提升，而是公平检验架构假设。正式 manifest 在 A3 前冻结数值 SESOI；至少
同时检查：

- strict-transfer `full-null` 是否改善，以及 CCP 相对 Q2 的 paired difference；
- 改善是否来自正确历史，而非 wrong-user 或 candidate-common 偏置；
- recurrence、overall、null-history relevance 是否出现不可接受退化；
- 投影、gate 与内部 writeback 的消融是否支持预期机制链；
- 代价是否仍符合 0.6B reranker 的可用范围。

若 strict transfer 不改善，按证据区分表示不足、gate collapse、contrast path 无信号、训练目标
不匹配或普通欠收敛；不得通过打开更多确认数据或 outcome-selected sweep 救结果。完成 A4 报告后
停止，等待是否扩展 seed、规模或未见人口的决定。

## 6. 当前状态

截至 2026-07-20：机制阶段证据已整理，开发授权已获得，V0 合同和阶段门已经写明；尚未创建
CCP 模型代码、配置、checkpoint 或结果。下一动作是 A0 设计冻结与代码骨架。
