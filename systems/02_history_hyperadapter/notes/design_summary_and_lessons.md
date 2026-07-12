# C02 阶段总结与后续设计启示

> 2026-07-11 终局修订：用户授权的单次机械续跑已修复 empty-mask NaN，并完成
> 五路固定 GPU 训练。执行有效，但 CHHT 在冻结 train-internal gate 仅通过 2/6：
> non-repeat 增益与最近控制优势均为 0，四类 corruption 的 core-norm ratio 均约为
> 1，且 loss 未下降；未生成 dev scores 或调用 evaluator。当前 C02 已形成有效
> 机制负结果并关闭，详见 `mechanical_continuation_outcome.md`。

## 一句话结论

C02 已完成从文献审计、机制设计、代码实现到真实 GPU smoke 的大部分前置工作，
但在 train-internal 阶段因一个全 no-history batch 的空掩码 NaN 停止，且两次实现
预算已经用完。因此这次是**工程执行失败，不是科学假设被证伪**：我们没有得到
有效 checkpoint，也没有运行 dev evaluator，不能判断 CHHT 是否真的有效。

## 已经完成的工作

- 文献审计后，放弃了容易归约为动态对角 LoRA 的初版方案，改为 CHHT：由
  `(query, candidate, history event)` 生成反对称低秩核，再通过 Cayley transform
  修改 Transformer 内部 FFN 的计算函数。
- 完成了 proposal、mechanism fingerprint、nearest-neighbor audit、冻结 gate、
  proposal lock，以及 CHHT 和四个 matched controls 的实现。
- 建立了 label-safe 数据与训练/打分路径：train 可读训练标签；dev 仅使用无标签
  records、冻结 D2p 分数和统一候选集；test 与 separated eval labels 均未读取。
- 13/13 单元测试通过，覆盖 skew/rank/Cayley、candidate conditioning、history-only
  隔离、no-history 精确退化、数据子集和确定性等合同。
- 完成 96,939 个 train 和 12,229 个 dev 请求的特征准备；候选集和 575,609 行
  D2p 分数全部对齐。
- 真实 GPU smoke 发现并修复了 bf16 下 Cayley 正交误差；改用 fp32 几何计算后，
  最大正交误差从 `8.76e-3` 降至 `7.42e-7`，有限梯度和 no-history 精确退化均通过。

完整执行证据见 [final_report.md](final_report.md)。

## 遇到的问题

### 1. 直接阻塞：训练损失的空集合处理不完整

冻结训练顺序中有 16 个全 no-history batch。CHHT 的 corruption loss 在这些 batch
上对空 `valid` 张量求均值，产生 NaN；首个发生在 epoch 1 的 batch 134。这个问题
本应由“所有证据形态的完整 loss smoke”在正式训练前发现。

### 2. 机制正确性验证充分，但训练系统验证不够完整

单元测试验证了算子本身，却没有覆盖完整训练目标在以下 batch 形态上的行为：

- 全 no-history；
- 全 repeat 或全 non-repeat；
- corruption 后没有有效事件；
- batch size 为 1 或被超大候选集切碎的 batch。

这说明未来的 preflight 不能只测 forward/operator，还必须枚举完整数据加载器产生的
所有结构类型，对每个 loss component 做 finite/gradient 检查。

### 3. 当前设计存在尚未被结果验证的科学风险

这些不是本次实验结论，但值得下一轮优先验证：

- **创新算子不等于有效信号。** Cayley/skew 让机制更难归约，但不能证明 4,677 个
  non-repeat 请求中存在足够可学习的跨商品个性化信号。
- **core norm 不是排序效用。** 训练目标鼓励 true history 的更新范数高于
  wrong/shuffle/coarse/query-mask；这只能证明模型“反应更大”，不保证更新方向能
  把正确候选排得更高，甚至可能奖励无用的敏感性。
- **D2p anchor 既稳定也可能遮蔽学习。** 它很好地保护 no-history，但如果内部
  residual 的监督太弱，模型可能长期停留在强 base 附近，学不到 non-repeat 增量。
- **matched total parameters 还不够。** 五个变体实例化参数量相同，但真正参与
  梯度的 active parameters、额外 corruption forward 和有效计算路径仍应单独审计。

## 对整个 proposed-system 设计的主要启示

### 1. 保留“默认 identity、证据充分才更新”的总原则

no-history 时 `Delta W=0` 的代数退化是这次最扎实的设计点，未来候选应继续保留。
更一般地，跨商品 personalization 不应默认开启；内部更新预算应随
query/history/candidate 证据一致性增加，而弱证据下保持接近 query-only identity。
这比无条件聚合用户历史更符合当前 motivation 的事实边界。

### 2. 先证明 transfer 可学，再为 transfer 发明复杂算子

下一轮在锁定新机制前，应只用 train/internal split 做一个便宜的 learnability
probe：固定同一 Transformer backbone，比较 query-only、简单 target attention 和
最小 candidate-conditioned internal update，专看 non-repeat 表面。若简单模型都无法
稳定超过 D2p，就应该先关闭或重新界定跨商品 transfer 假设，而不是继续增加
hypernetwork、router 或多粒度模块。

### 3. corruption 应约束“有用方向”，而不只是更新幅度

更好的训练信号应直接关联候选排序，例如：

- true history 相对 wrong/shuffle history 能否改善正负候选 margin；
- candidate swap 后 `Delta W` 与候选证据是否按预期变化；
- corruption 是否消除 true history 带来的正确 logit delta，而不是仅降低矩阵范数。

算子范数可以作为稳定性正则，但不宜继续承担主要机制证据。

### 4. 将 gate 分成三层，按顺序消费预算

1. **Correctness gate**：遍历完整训练 batch 结构，检查 finite、gradient、mask、
   exact fallback 和 determinism。
2. **Responsiveness gate**：在 train/internal 上验证 candidate/query/wrong/shuffle
   是否真正改变内部更新，并检查 candidate conditioning 相对 history-only 的差异。
3. **Utility gate**：只有前两层通过后，才训练完整 controls、生成 dev scores，并
   消耗唯一 evaluator call。

这样不会放宽科学标准，只是避免机械缺陷消耗宝贵的实现和 dev 预算。

### 5. 下一候选应优先“更少自由度、更直接的证据路径”

CHHT 同时包含事件组合、反对称核、Cayley rotation、recurrence preservation 和四类
corruption，首轮 probe 的自由度和故障面偏大。未来更值得优先尝试的是：单层、低
rank、单一 candidate-conditioned update，并让每个参数都能映射到一个明确可证伪的
行为。最近邻 controls 仍要保留，但可以在 pre-outcome falsifier 通过后再完整展开。

## 建议的下一步

- 若继续 C02：只修复空 `valid` mask，增加全 no-history 完整训练损失回归测试，
  重新锁定并从头训练；不要因为尚不存在的 dev 结果修改 rank、阈值或模块。
- 若启动新候选：先执行 train/internal learnability probe；只有确认 non-repeat
  transfer 存在后，再选择新的 internal modulation operator。
- 无论走哪条路线，都应保留 D2p/no-history identity、candidate conditioning、
  history-only control、repeat preservation、统一 evaluator 和 label isolation。

机械续跑的严格授权边界已经写在
[mechanical_continuation_request.md](mechanical_continuation_request.md)。
