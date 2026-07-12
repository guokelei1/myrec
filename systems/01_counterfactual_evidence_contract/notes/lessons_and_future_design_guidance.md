# C01 阶段总结与后续设计启示

日期：2026-07-11
状态：**当前 C01 实现停止；没有得到有效的模型胜负结论，也没有进行 dev 评测。**

## 一句话总结

C01 已经完成了从机制构思、文献审计、proposal lock、最小 Transformer 实现到
train-internal 执行的整条链路，但由于 train/internal 使用了错误的 D2p anchor，
本次训练和内部排名数字全部失效；它的主要价值不是证明 CECT 成功或失败，而是
暴露了未来 proposed-system 设计必须解决的三个核心问题：**基线语义精确复现、
可靠信号的最终排序保护，以及 certificate separation 与真实 ranking utility 的
区别。**

## 1. 已经完成的工作

### 1.1 设计与创新性审计

- 独立提出并冻结了 Counterfactual Evidence-Contract Transformer（CECT）。
- 架构由三个组件组成：
  1. Triadic Event Transformer（TET）；
  2. Counterfactual Quantile Contract（CQC）；
  3. Contracted Residual Readout（CRR）。
- 共享 Transformer 同时编码真实历史与 wrong-user、shuffle、query-mask、
  coarse-only 四类训练/诊断 twin；推理只使用真实 `(q,c,H)`。
- exact recurrence 被设计为同一 evidence contract 内的 protected atom，非独立
  router 或事后 scorer。
- 完成了 DIN、SIM、ZAM、TEM、RTM、SASRec/BERT4Rec、CFT、CARD、
  counterfactual recommendation 与 calibration 文献审计。
- pre-outcome nearest-neighbor verdict 为 `distinct-with-uncertainty`：机制不能直接
  归约成最近邻，但尚未获得经验有效性或论文级 novelty 证据。

### 1.2 实现与协议

- 建立独立环境 `myrec-c01`，只使用物理 GPU 0 和 seed `20260708`。
- 实现 96 维、2 层、4 头的 query-candidate-event Transformer。
- 实现多 twin margin、train-only quantile calibration、冻结 certificate 后的
  value/readout 阶段，以及参数完全相同的 plain-attention control。
- 实现严格 no-history fallback、候选 manifest hash 检查、split 内 wrong-history
  donor、true-input-only dev scoring 路径和 evaluator 前审计。
- proposal/config 在任何 C01 outcome 前完成哈希锁定；锁定后没有修改设计门槛。

### 1.3 测试与执行

- 候选内单测 9/9 通过。
- CPU/GPU smoke 均通过，Transformer attention、certificate head、value head 的
  梯度均有限且非零。
- attempt 2 完整复现了 attempt 1 的训练 loss 轨迹，训练路径具有确定性。
- 两次尝试累计约 `0.3261` A40 GPU-hours，未超过 8 小时预算。
- 没有生成 C01 dev scores；共享 dev evaluator 调用数为 0；test 完全未读。

## 2. 遇到的问题

### 2.1 决定性的有效性问题：D2p anchor 用错

当前 train/internal adapter 使用的是：

```text
0.6 * z(frozen-BGE cosine) + 0.4 * z(train popularity)
```

而仓库中冻结注册的 D2p 是：

```text
0.6 * z(seed-20260708 fine-tuned D2t query-tower score)
+ 0.4 * z(legal train-only popularity).
```

这不是无关紧要的尺度误差。错误 base 同时进入了训练 ranking loss、item-only
control 和 internal comparison。因此：

- attempt 2 的 repeat/non-repeat 排名差不能用于判断 CECT；
- plain control 的相对差也不能形成机制归因；
- 本轮不能声称“CECT 被证伪”，只能声称“当前实现无效”；
- 两次实现尝试已经耗尽，所以不能在当前预算下补做第三次训练。

### 2.2 工程链路问题

- 初始缓存代码错误假设 shared score 与 packed arrays 同序，后来改成按
  `(request_id, candidate_item_id)` 显式对齐和原子落盘。
- GPU deterministic mode 缺少 `CUBLAS_WORKSPACE_CONFIG`，在正式尝试前通过
  smoke 发现并补上 fail-closed 设置。
- attempt 1 在训练结束后的 reporter 中调用不存在的 `torch.flatnonzero`，消耗了
  一次允许的实现重试。
- 设计公式、README 命令名和最终代码之间仍存在少量 conformance drift。未来不应
  只靠人工阅读 proposal 判断实现是否严格等价。

这些问题说明：两次实现预算应该用于真正的数值异常或模型实现问题，而不能被
本可在全链路 dry-run 中发现的 adapter/reporter 问题消耗。

## 3. 无效运行中仍值得保留的诊断线索

下列数字**全部来自无效运行，只能生成下一轮假设，不能写入论文结果或作为模型
选择依据**。

| 观察 | 无效运行中的现象 | 可供未来验证的假设 |
|---|---:|---|
| no-history fallback | score 最大差 `0.0`，rank mismatch `0` | 结构化 evidence-empty fallback 是可靠且应保留的设计模式 |
| true vs shuffled admission mass | 相对下降仅 `0.00166` | positional embedding 并未使顺序成为 load-bearing signal；模型近似把历史当集合 |
| wrong-user admission | 相对下降约 `24.3%`，未到冻结的 `30%` | certificate 对 user identity 的区分不足，或不同用户历史仍触发通用相似性 |
| coarse/query-mask admission | 分别下降约 `96.4%/96.0%` | 模型能拒绝“容易的”破坏，但这不能证明能拒绝最接近真实分布的 hard twin |
| true-minus-pooled-twin energy | `+0.3485` | energy separation 可以很好看，但不等价于 ranking utility |
| global threshold | `Q_cf=1.6484`；各 twin false admission 约 `0.6%–18.8%` | 四类 null 分布高度异质，混成一个绝对 quantile 会让 easy/hard twin 校准失衡 |
| repeat surface | 表面 delta 为 `-0.0308` | 把 exact floor 放进 contract 不足以保证最终 item-only 排名不被 residual/归一化稀释 |
| non-repeat surface | 表面 delta 为 `-0.1088` | “通过 certificate 的事件”未必具有正的候选排序价值 |
| contract vs plain | 表面差 `+0.0557` | 相对优于更差的 control 不代表超过合法 anchor；control 和绝对水线必须同时过门 |

这里最重要的线索是：**模型确实可以学出一个非恒定 certificate，但 certificate
区分 corruption 的能力与它是否能改善产品排序是两回事。**

## 4. 对整个 proposed-system 设计的指导

### 4.1 把 baseline fidelity 变成系统级输入契约

未来所有候选都不应自行“近似重写” D2p、item-only 或其他注册 control。建议在
训练前统一完成：

1. 从冻结 checkpoint 物化 train/calibration/internal 的注册 base score arrays；
2. 记录 checkpoint、config、score array 和 candidate manifest 的 SHA256；
3. 对随机请求逐候选比较 shared scorer 与本候选 adapter，要求最大误差为 0 或
   预注册浮点容差；
4. 将 parity test 作为 G0，未通过时禁止创建模型 optimizer；
5. train 与 dev 只更换合法 popularity 统计作用域，score 语义必须完全相同。

这比单纯检查 candidate hash 更重要：candidate universe 正确，并不代表 baseline
score 的语义正确。

### 4.2 “protected exact recurrence”必须是最终排序不变量

C01 只给 exact recurrence 一个内部 floor，但后续 non-exact residual 和 request
内 z-score 仍可能改变 repeat 请求的最终顺序。未来设计应把保护写到最终 logit
约束，而不是只写到中间特征：

- repeat-present 请求上，主模型相对 item-only 的扰动应被显式约束；
- 可使用 item-only ranking distillation、受限 residual、单调性或 non-degradation
  loss，但仍须位于同一 Transformer contract 内，不能退化成 fixed-score router；
- 在训练早期就逐 batch 检查 repeat ranking parity，不要等完整 internal gate 才
  发现可靠信号被冲淡。

设计目标应从“exact signal 被加入”升级为“exact signal 在最终排名中不可被未经
证明的 transfer 覆盖”。

### 4.3 不要把异质 counterfactual null 粗暴合成一个全局阈值

coarse/query-mask 是容易识别的强破坏，wrong-user/shuffle 更接近真实数据分布。
把它们的 raw energy 放入一个全局 quantile，会产生三个风险：

- easy twin 让整体拒绝率看似很好；
- hard twin 决定阈值，却仍可能和 true 大量重叠；
- 不同历史长度、候选类型和 corruption family 的 energy scale 不可比。

更值得下一轮预注册验证的是：condition-wise standardized margin、每类 null 单独
上界后取 worst case，或把相对 twin margin 蒸馏成 true-input-only certificate。
无论选哪一种，都必须用 matched ablation 证明收益来自校准方式，而不是更多
loss/容量。

### 4.4 先证明顺序真的有信息，再设计 sequence primitive

true 与 shuffled 的 admission mass 几乎相同，说明“放了 positional embedding”
并不等于“模型使用了顺序”。后续候选若声称 temporal/order mechanism，应先做
一个廉价 pre-outcome probe：

- shuffle 后 hidden state、certificate 和 ranking logit 是否稳定改变；
- 改变是否在多种非身份 permutation 上方向一致；
- 去掉 position/order 后 matched model 是否等价。

若这些检查不通过，应承认该数据作用面更像 evidence set，而不是继续堆更复杂的
sequence architecture。

### 4.5 certificate gate 必须同时检查“可辨别”与“有用”

未来 gate 至少需要两个互不替代的轴：

```text
fidelity: true evidence 是否显著区别于 hard counterfactual evidence？
utility: 通过 certificate 的 residual 是否改善合法 anchor 上的排序？
```

仅有 energy gap、variance 或 admission rate 只能排除全零/全一坍缩，不能证明
ranking value。建议增加以下 train-internal 诊断：

- admitted residual 对 clicked candidate 的增量是否显著高于 unclicked candidate；
- certificate 分桶后，增量 ranking utility 是否单调；
- 去掉 value head、只保留 certificate 时是否仍出现伪 separation；
- hard twin rejection 与真实 ranking gain 是否在 request level 正相关。

### 4.6 最危险的不是 easy corruption，而是“近分布 hard twin”

coarse-only 和 query-mask 的大幅下降并不足以支持机制。真正约束设计的是
wrong-user 与 order-shuffle，因为它们保留了大量文本、长度和行为统计。未来 common
gate 应逐类通过，不能用 pooled average 掩盖任何一个 hard twin 失败。

### 4.7 no-history 结构化退化值得跨候选复用

C01 中最清晰的工程成功是：没有历史时 personalized path 结构上为空，输出逐位
等于 base，而不是依赖模型“学会忽略空历史”。这个模式应成为所有候选的统一
接口约束：

- `history_present=false` 时不执行伪历史、不生成 learned default user state；
- score/rank parity 在模型层直接断言；
- 不允许用训练损失期待模型近似学出 fallback。

## 5. 建议的下一轮执行顺序

建议将未来候选的 cheap gate 拆成四层：

1. **G0 — 输入与基线一致性**：candidate、D2p、item-only、split、label isolation
   全部逐位/逐哈希通过；不训练。
2. **G1 — 机制敏感性**：在小样本和随机初始化/短训练下验证 exact invariant、
   shuffle sensitivity、wrong-user sensitivity 和 no-history fallback；不看 dev。
3. **G2 — utility falsifier**：只在冻结 train-internal 上测试 repeat non-degradation、
   non-repeat gain、hard twin rejection 和 matched control。
4. **G3 — blind dev screening**：仅当 G0–G2 全部通过时生成 dev scores、做
   deterministic rescore，并调用一次 shared evaluator。

正式 attempt 计数应从 G0 和 reporter 全链路 smoke 通过后开始。smoke 必须覆盖
真实 packed adapter、score alignment、checkpoint reload、internal reporter 和
最终 JSON serialization，而不只是模型 forward/backward。

## 6. 对未来架构方向的凝练判断

当前证据仍支持继续研究 **candidate-conditioned history-evidence fidelity**，但不
支持简单的“一个全局 counterfactual threshold + 一个正值 residual”作为充分
答案。更有希望的设计原则是：

> 在同一 Transformer 排序核心中，把 exact recurrence 写成最终排名层面的受保护
> 不变量；对 cross-item transfer 使用相对、局部、hard-null-aware 的证据契约；
> 只有当 certificate fidelity 与 residual ranking utility 同时成立时才允许它
> 改变 item-only 排名。

这仍然是一个可证伪 primitive，而不是 router。它也比“增加更多语义/协同特征”
更具体：任何新模块都必须支付三份 rent——保护 repeat、改善 non-repeat、拒绝
hard twin；少一项都不应进入 dev。

## 7. 当前应如何处理 C01

- 当前 checkpoint 和 internal 数字只保留作实现审计，不进入论文或候选比较。
- 不应在当前 C01 上追加模块、改 threshold 或补做 dev。
- 若协调者希望重启，必须新授权、重新 lock，并首先物化正确的 seed-20260708
  train D2p score arrays；否则应永久关闭本实现。
- 详细执行证据见 `notes/final_report.md` 和
  `artifacts/c01_cect_probe/internal_gate_report.json`。
