# N21--N24 training-boundary plan (pre-registered, inactive)

状态：2026-07-20。本文件扩展 Transformer 深层探索到训练动力学与参数化边界。它不
修改冻结 checkpoint、第一轮结果或统一数据接口；在 N8--N20 机制 closeout 前不启动
GPU 训练，不读取效应值选择控制，也不把诊断控制称为新方法。

## 共同不变条件

- 只用 KuaiSearch train 和 train-only internal-dev；confirmation/test qrels 对训练与
  scorer 不可见。每个控制保持 backbone、初始化函数、可见字段、记录顺序、总 optimizer
  step、batch token budget、evaluator 和 checkpoint lineage 不变。
- 所有训练对照至少两个预注册 seed；原始 matched control 与诊断 control 同时运行，保留
  divergent、under-converged 和 mechanical failure，不按 strict-transfer 结果挑 seed。
- 报告 raw loss、梯度、effective parameter/function update、full/null/strict-transfer
  frozen utility，使用 request-level paired evaluator；训练控制改善不等于架构证据。
- 每个 family 先做 tiny deterministic gradient/recomposition smoke，再做可 resume 的短
  诊断训练；连续 GPU job 不超过 13,500 秒，独立 checkpoint/output 目录。

## N21：Q3 FP32-adapter / BF16-base cast boundary

### 问题

Q3 使用 BF16 frozen base 与 FP32 LoRA factors，PEFT 在 adapter 输入和 scaled contribution
处存在 dtype cast。现有 LoRA path geometry 不能说明 cast-local error 是否改变了有效函数
更新或 transfer response。

### 固定对照

- native BF16-base/FP32-adapter cast；dtype-aligned BF16 adapter；FP32 base reference；
  三者初始化时匹配 `B@A` function delta、dropout mask、optimizer state 和 update budget。
- 每 step 记录 base result、adapter result、cast-back result、per-row cast residual、
  raw gradient 与 applied delta；不只比较最终 loss。

### 证伪门

只有 cast residual 在多个 seed、多个 fixed batch family 上稳定，并且 integrated function
update 与 frozen utility 方向一致，才能标为 `dtype-boundary candidate`；单次 dtype 数值差异
或最终 NLL 变化不足以支持 transfer 归因。

## N22：LoRA input-dropout boundary

### 问题

Q3 q/v LoRA 的 `p=0.05` input dropout 只在训练时启用。它可能改变低秩分支的有效 rank、
梯度方向或 q/v 不平衡，但现有 final A/B geometry 没有隔离这一随机算子。

### 固定对照

- 原生 dropout、固定-mask replay、dropout=0 reference；保持相同 seed、batch 顺序、总
  update 和 final evaluation path。
- 按 q/v、early/middle/late 预注册相对深度汇总所有 28 blocks，不选择单个 path；记录
  dropout 后 rank-input covariance、gradient cosine、B@A function update 与 loss share。

### 证伪门

要求至少两个 seed、固定-mask replay 与 native stochastic path 的差异可复现，并通过 q/v
对称 negative control 和 frozen utility；否则只记为随机优化噪声，不能设计成 adapter 结构。

## N23：checkpoint gradient bridge / recomputation boundary

### 问题

Q3 训练使用 `enable_input_require_grads` bridge 和 non-reentrant gradient checkpointing。
现有参数更新记录无法区分 bridge、重算与真实 LoRA 梯度路径。

### 固定对照

- native bridge + non-reentrant checkpoint；checkpoint-safe explicit bridge reference；
  no-bridge gate-stop control（只用于确认梯度是否缺失，不作为性能方法）。
- 同一 microbatch 上逐参数比较 q/v LoRA raw gradient、recomputed gradient、effective
  update、embedding output gradient coverage 和 optimizer state；前向 logits 必须 exact
  identity 到机械容差。

### 证伪门

先过完整 q/v gradient coverage 与 forward identity，再看多 seed 下的梯度/更新差异；只要
差异不能在重复运行中稳定，N23 保持 unresolved，不声称 checkpoint 是 transfer 根因。

## N24：objective / optimizer effective-update boundary

### 问题

Q2 同时有 listwise 与 pairwise ranking loss，Q3 有 alignment NLL；Adam moment、global
  clip、decoupled weight decay 和学习率缩放又会重新组合有效更新。现有 objective conflict
  观察尚未把“loss share”与“实际参数函数更新”分离。

### 固定对照

- Q2：原生 listwise+pairwise、matched listwise-only、matched pairwise-only；Q3：原生
  alignment NLL 与 token-balanced NLL reference。总 step、初始化、batch token、negative
  contract 和 optimizer 超参固定，诊断 runs 不改变 evaluation population。
- 同步记录 per-family loss/gradient share、Adam preconditioned direction、clip/decay
  contribution、`B@A` effective delta 和 frozen full/null/strict-transfer utility。

### 证伪门

只有 loss-family 梯度方向、effective update 与 behavior effect 在两个 seed 和 Q2/Q3 对应
边界上共同一致，才允许把 objective/optimizer 标为 training-mechanism candidate；单纯
重新加权后 strict-transfer 上升不能证明 Transformer 内部瓶颈。

## 排程与停止点

N21--N24 不抢占当前 N15/N16 或 N17--N20 队列。只有在 inference operator 的 N8--N20
closeout 后仍存在 training-mechanism unresolved debt，才按 N21 -> N22 -> N23 -> N24
启动；若前向 operator 已足以解释现象，则这些 family 保留为 preregistered negative
boundary，不继续 outcome-driven 扩大训练 sweep。

