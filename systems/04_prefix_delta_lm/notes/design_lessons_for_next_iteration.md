# C04 阶段总结与下一轮设计启示

日期：2026-07-11
状态：C04 已终止，结论为 `stop`

本文只总结 C04 已经完成的工作、失败原因及其对 PPS 后续 proposed-system
设计的约束。完整数字和协议审计见 `final_report.md` 与
`completion_integrity_audit.json`。

## 1. 已经完成了什么

C04 实现了一个完全本地的 Counterfactual Prefix-Delta Language Recommender：

- 用同一个 BGE-small Transformer，分别对 `[q,H,c]` 和
  `[q,NULL_HISTORY,c]` 产生候选 logit；
- 用 factual/null logit 差构造 candidate-order tangent，最终排序仍由该
  Transformer 路径产生，而不是把两个外部 scorer 固定混分；
- 采用静态 rank-8 LoRA、固定候选池、确定性 item identity token，以及
  train-only D2p 排序蒸馏；
- 实现 single-pass、无 tangent、普通拼接 head、static LoRA 和 exact-identity
  shortcut 五个对照；
- 完成 proposal lock、两次实现尝试记录、13 个单元测试、train/internal
  probe、确定性重评分、一次 dev evaluator 调用和最终完整性审计；
- 候选代码没有读取 dev/test qrels，也没有访问 test。总计使用
  0.207575 A40 GPU-hours。

工程上可复用的成果主要是：固定候选与哈希检查、label/split guard、paired-prefix
批处理、确定性打分、候选级 tangent 测试、对照矩阵以及 pre-outcome lock 流程。

## 2. 遇到的主要问题

| 问题 | 观察 | 含义 |
|---|---:|---|
| query-only 底座没有守住 | train/internal D2p pair concordance 只有 0.63344，阈值为 0.80 | 个性化尚未开始，基础排序就已不可靠 |
| no-history 不等价 | 4,110 个请求中有 4,098 个与 D2p 排序不一致 | “delta 为零”不等于“最终基础排序正确”；null path 本身必须先成立 |
| 没有学出跨商品迁移 | non-repeat tangent 均值为 -0.001274；描述性 dev 上相对 D2p 为 -0.015395 | 当前 history 表示没有提供可靠的 transferable evidence |
| 更像 exact-repeat shortcut | repeat tangent 为正，而 non-repeat 为负；repeat-present 描述性结果也明显退化 | 模型容易利用 item identity，却没有证明跨商品个性化机制 |
| corruption contract 失败 | wrong/shuffle/query-mask/coarse delta 比例为 1.071/1.028/0.954/0.823，门槛为 0.50 | 模型对“历史是否可信、是否与查询相关”不敏感，delta 更像输入扰动而非证据 |
| tangent 会强制改序 | 该投影专门保留与 null ordering 正交的分量 | 当 history signal 不可靠时，它会把噪声直接变成候选改序，而不是安全地保持基线 |
| 协议顺序失误 | 0.63344 未通过冻结的 pre-dev 门槛后仍调用了一次 evaluator | dev 数字只能作描述性负证据；未来必须由程序硬阻断失败后的评测 |

此外，第一次实现出现了 padding 极小值乘零后产生 NaN 的数值问题；第二次修复后
训练正常。PyTorch 的 CUDA peak-memory reset 还要求先初始化 CUDA context。
这些是工程问题，不应被解释成科学结果。

## 3. 这次失败能说明什么，不能说明什么

可以说明：

1. “共享 LM 的 factual/null 差值”本身不会自动带来可靠个性化；
2. 只靠小规模 train-only KL 蒸馏，无法同时保证 null path、no-history 和 D2p
   排序不变量；
3. 在证据质量尚未建立时，强制产生 order-changing residual 风险很高；
4. 当前训练信号更容易学习 exact recurrence，而没有学到 4,677 个 non-repeat
   请求所需的跨商品迁移；
5. raw `h-n` 在机制上接近 LM classifier-free guidance / counterfactual logit
   pairing。若新算子不能在 matched ablation 中产生不可替代的收益，仅换名称或
   prompt 不能构成创新。

不能说明：

- 不能据此证明所有 paired-prefix、Transformer 或 LLM4Rec 路线都无效；
- 不能区分失败究竟主要来自训练规模、anchor 可观测性、表示能力还是 tangent
  形式，因为对照没有获得额外 dev 调用；
- 不能把 sequencing-invalid 的 dev 数字登记成正式 paper result；
- 不能从 repeat 为正推导“identity causality”，也不能从 non-repeat 为负推导
  数据中绝对不存在 semantic transfer。

因此，被关闭的是这一个具体 primitive 和训练合同，不是整个研究方向。

## 4. 对下一轮 proposed-system 的设计指导

### 4.1 先建立强基础排序，再允许个性化

下一候选首先应在 train/internal 上证明 query-item/null-history 排序本身可靠。
如果 teacher 使用了 popularity、item identity 或其他因素，学生输入和参数化必须
能够观察并表达这些因素；否则“精确复现 teacher ordering”是欠定约束。

更稳妥的方向是：先得到一个足够强的本地 Transformer query-item ranker，再在其
内部加入 zero-initialized 的 history residual。无历史时 residual 应由结构保证为
零，并且基础路径不能因个性化训练而漂移。这仍应是 Transformer 端到端产生排名，
不能退化成外部固定分数 router。

### 4.2 把 no-history 从 loss 目标升级为结构不变量

C04 已经保证空历史时 factual/null token 和 delta 相同，但共享 LoRA/head 在训练中
改变了 null ranker，所以最终仍与 D2p 不等价。下一设计需要同时保证：

- history module 在 `history_present=false` 时严格 no-op；
- 个性化训练不能破坏已经验证的 query-only 参数或排序子空间；
- 在任何 dev 调用前，用 train/internal proxy 验证基础排序和空历史一致性。

### 4.3 先证明“有信息”，再发明“怎么用信息”

下一轮最便宜也最关键的 pre-outcome probe 应先回答：在排除 exact repeat 后，
`(q,H,c)` 是否包含能稳定区分正负候选、且在 wrong/shuffle/query-mask 下消失的
信号？如果连这个 probe 都不成立，就不应继续设计复杂 attention、adapter 或
counterfactual operator。

换言之，研究顺序应是：

```text
query-only 排序成立
  -> non-repeat history signal 可检测
  -> corruption 后信号消失
  -> repeat 安全性成立
  -> 才测试新的端到端个性化 primitive
```

### 4.4 默认应保守保持排序，而不是强制改序

C04 的 tangent 删除了与基础排序平行的分量，只保留改序分量；这个几何约束很
“干净”，但在弱证据下会把噪声放大为排序变化。未来更合理的归纳偏置是内部
trust region：只有 joint query/history/candidate evidence 通过可证伪的可靠性条件
时才允许有限改序，否则保持强 query-only 排序。

这里的“可靠性条件”必须位于 Transformer 内部并接受 matched-capacity ablation，
不能实现为外部 fixed-score gate 或手工 query-type router。

### 4.5 exact recurrence 与 cross-item transfer 要分别验收

当前证据继续支持“exact recurrence 容易利用”，但不支持“粗粒度或一般语义历史
可以迁移”。后续模型可以保留 exact-match 的安全收益，但必须单独证明 non-repeat
表面上的增益来自 candidate-conditioned evidence；不能用 repeat 请求的提升掩盖
cross-item 失败。

最重要的报告顺序应固定为：no-history、repeat-present、non-repeat-present、四类
corruption，然后才看 overall。只看整体均值很容易把机制失败隐藏掉。

### 4.6 创新点应落在 evidence fidelity，而不是 logit subtraction

`conditional logit - null logit` 已有很近的 CFG/CLP 邻居。下一设计若仍使用
counterfactual difference，创新必须来自一个可验证、不可被 single-pass/static
LoRA/普通拼接复现的 evidence-fidelity 机制，例如内部的候选条件证据约束或安全
残差结构，而不是新的命名、模板或额外 loss。

### 4.7 用程序执行 gate，而不是靠人工记忆

本次最明确的流程教训是：pre-dev gate 必须输出机器可读状态；scoring/evaluator
入口在状态不是 `pass` 时应直接拒绝执行。这样才能避免“已经看到失败，但仍顺手
跑一次 dev”的不可逆协议错误。

## 5. 建议的下一候选最小门槛

在提出新机制前，先冻结并依次检查：

1. teacher 的所有排序因素对模型是否可观测；
2. query-only/null path 是否在 train/internal 达到预注册一致性门槛；
3. no-history 是否由结构保证 no-op；
4. non-repeat 正向信号是否存在，并在四类 corruption 下显著衰减；
5. repeat-present 是否至少保持 item-only；
6. single-pass、普通 static LoRA、简单拼接和 nearest-neighbor degeneration 是否
   已定义；
7. 上述条件全部通过后，才生成完整 label-free dev scores 并调用 evaluator。

具体阈值必须在看到新候选 outcome 前重新冻结，不能从 C04 的失败数字倒推调整。

## 6. 一句话结论

C04 最有价值的负面结论不是“LM 个性化失败”，而是：**当 query-only 底座没有
守住、history evidence 尚未通过反事实检验时，强制使用 factual/null delta 改变
候选顺序只会把不可靠历史变成排序噪声。下一轮应把“强底座、结构性 no-op、先验
证据信号、保守改序”作为设计顺序。**
