# C03 阶段总结与对整体设计的启示

更新时间：2026-07-11

## 当前判断

C03 已完成设计冻结、最小实现、单元测试和数据特征准备，但训练在约 7 小时 25 分仍未产生 checkpoint。计入特征准备后，GPU 2 的保守累计占用上界约为 7.456 小时；按实测吞吐，剩余预算不可能完成训练、诊断、确定性复算和完整 dev scoring，因此已在 8 小时硬上限前主动终止。

当前建议是 **`stop`**：C03 没有在冻结预算内完成可评估的 screening。这个结论说明当前机制/实现的复杂度不可接受，但由于没有进入 dev evaluator，**不能**进一步声称它已经被真实排序指标否定，也没有任何 C03 dev 性能数字可报告。

## 已经完成的工作

- 按项目的数据隔离、候选哈希、GPU、环境和评估预算约束冻结了提案与 gate；没有读取 dev/test 标签，也没有读取其他候选的设计。
- 做了预结果文献核查。原始的“多分布 Sinkhorn + cycle consistency”表述与已有多边/多重最优传输过近，因此收窄为：query、history、candidate 三组**候选锚定的部分传输**，每条边有可学习 null/dustbin，只有三条真实质量流的交集可以更新候选表示。
- 实现了一个最小的 Transformer 排名探针：冻结文本嵌入、小型可训练 Transformer、受保护的 exact-item cost atom、部分传输、cycle 约束和 fail-closed residual；无历史时 residual 严格为零。
- 实现了 softmax、no-null、no-cycle、mean-pool 等匹配控制，以及 wrong-user、history shuffle、query mask、coarse-only 等内部破坏实验。
- 建立并验证了独立环境 `myrec-c03`。单元测试目前 8/8 通过，包括质量守恒、null 行为、exact-match 单调性、有限梯度、无历史严格零残差和 centered residual。
- 已在 GPU 2 上完成 train/label-free dev 特征准备；训练样本按 request ID 在读标签前固定抽取。训练阶段严格只使用 train labels，未读取 dev/test qrels。
- 两次实现尝试已用完：第一次暴露普通有限轮 Sinkhorn 的守恒误差；第二次改用 singleton Newton 求解后通过全部单测，但端到端训练吞吐未通过预算 stop-loss。
- 没有生成 checkpoint、scores 或 dev run，primary dev evaluator 调用次数为 0；这避免了拿不完整或超预算结果做事后解释。

## 暴露出来的问题

1. **新颖性边界偏窄。** 传输、dustbin、cycle consistency 和多边耦合本身都有较近的先例；C03 真正可主张的只剩“候选锚定、可拒绝、交集质量流作为排序信息通道”这一具体组合，目前新颖性判断仍是 uncertain。
2. **机制可能过度保守。** 三条边的质量以乘法/交集方式汇合，任何一条弱边都会把有效信号压到接近零。它可能很安全，却无法在最关键的 non-repeat 历史上产生足够增益。
3. **最小探针中的三元结构可能退化。** candidate 侧是单个候选，三次部分传输容易退化成“带 null 的 target attention”，未必需要完整 OT machinery；因此匹配控制比机制名称更重要。
4. **exact recurrence 可能继续支配模型。** exact-item cost atom 能保护 repeat，但也可能让模型只重建已经很强的 item-only 信号，而没有学到真正的跨商品迁移。
5. **训练诊断存在被目标函数教出来的风险。** corruption loss 可以让模型按构造在破坏输入时增加 null；这只证明实现按预期响应，不能证明它改善真实排序。真正的判据仍应是 label-free corruption、确定性检查和唯一一次 dev 上的 non-repeat 增益。
6. **当前仍是设计 falsifier，不是完整系统证据。** 冻结嵌入、小 Transformer 和 D2p skip connection 适合低成本否证，但如果它幸存，仍需把 base ranking 能力和该机制内化到同一个端到端 Transformer 图中，才能满足最终 LLM4Rec 系统要求。
7. **工程性价比已经触发 stop-loss。** 训练期间显存约 0.9 GB、GPU 利用率约 23%--25%，瓶颈是逐候选、四种 corruption 和三边求解造成的 CPU/调度与重复前向。最小 falsifier 尚且无法在 8 小时内走完整条评估链，复杂度本身已经构成失败证据。

实现中还遇到一个具体数值问题：普通有限轮 Sinkhorn 在严格质量守恒测试下误差约为 `3.4e-4`。在设计冻结前，已改为针对当前 singleton 侧情形求解同一熵正则缩放方程的 Newton 版本，测试误差达到要求。这说明“严格守恒”不仅是建模假设，也会成为实现与效率负担。

## 对后续整体设计的指导

- 把核心科学问题写成：**模型何时应该相信跨商品历史证据，何时应该拒绝它，并且这种选择是否真的改善 non-repeat 排名**。不要把 OT、attention 或某种模块形式本身当作贡献。
- 将“安全性”和“有效性”拆开验收。null 增加、corruption 响应、无历史零残差只证明 fail-closed；必须另设 non-repeat 正增益及置信区间门槛，防止“全部拒绝”成为伪成功。
- exact recurrence 应作为必须保护的合同和强基线，而不是主要创新点。未来机制要明确证明：repeat 不退化，同时 non-repeat 的收益不是 exact-match 泄漏或流行度重建。
- 在完整实现前优先做最小、可失败的探针。若一个机制在固定候选集、单种子和严格预算下连方向性 non-repeat 信号都没有，就不应靠更大模型、更长训练或更多调参挽救叙事。
- 在正式训练前增加一个**端到端吞吐外推 gate**：用少量真实 request 同时覆盖训练、corruption、确定性复算和全 dev scoring 路径，估算总 wall-time；不能在预算内闭环的机制应先简化，而不是训练到预算末尾才发现不可评估。
- 最近邻消融应围绕**信息流差异**设计，而不只是去掉一个 loss。尤其要检验复杂结构是否退化成 target attention、普通 gating 或固定 residual。
- 未来结构应避免让多个不稳定相似度以连乘方式决定唯一通道；可以保留“可拒绝证据”原则，但让证据聚合和置信度校准更直接、更容易审计。
- 如果多个候选最终都无法稳定超过强基线，也不应强行包装系统贡献。当前工作仍可沉淀为：严格的 PPS 基准与协议、repeat/non-repeat 的证据分解、以及“历史信息必须按可靠性 fail closed”的负结果与设计规律。

## 本轮结论

C03 以 `stop` 关闭，不追加组件、不改阈值、不做 dev evaluator 调用。若未来重新研究这一原则，应作为新的、重新冻结的候选：保留“证据不足时可拒绝”和 repeat/no-history 合同，去掉逐候选三边严格传输与多次连乘，把可评估性纳入机制 gate；不能把本轮未完成训练解释为正面或负面的排序效果。

更细的冻结定义见 [proposal.md](proposal.md)、[mechanism_fingerprint.md](mechanism_fingerprint.md)、[nearest_neighbors.md](nearest_neighbors.md) 和 [gate_protocol.md](gate_protocol.md)。
