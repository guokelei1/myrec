# C05 successor design authorization

日期：2026-07-11

## 决定

在 C01--C04 全部结束并完成横向失败审计后，协调者明确授权在 `systems/` 下启动
一个串行后继候选 C05。该决定修订了 doc 15 原先只允许四路盲化并行候选的阶段
边界，但不放宽其科学、数据或评测规则。

## 为什么不是“四路拼装”

C05 不继承 certificate head、Cayley hyperadapter、三边 transport 或 paired-prefix
delta。它只提出一个新假设：跨商品历史若有用，应当在当前 query 和 candidate set
下产生候选可区分的排序证据；对所有候选近似相同的历史相关性不应改变排序。

相应的唯一 primitive 是 candidate-contrastive evidence budget：在一个
Transformer personalization block 内，先对每个历史事件的候选支持做候选间中心化，
再把超过 dead zone 的 signed evidence 归一到严格小于 1 的更新预算，并注入
candidate residual stream。

## 当前授权边界

下列内容是首次设计授权时的边界；同日后续的
`20260711_c05_prerun_review_and_g2a_amendment.md` 已显式增补 GPU 0 上的 G0/G2a
权限，并以该后续修订为准。

- 允许：proposal、mechanism fingerprint、最近邻审计、gate protocol、配置、CPU
  unit/synthetic tests，以及在再次锁定后的 train-internal signal probe。
- 暂不允许：占用任一物理 GPU、调用 dev evaluator、读取 test、完整模型训练、
  多 seed 或跨数据集扩展。
- 下一授权点：G0 输入/base parity、G1 synthetic contracts 和 train-internal
  non-repeat learnability gate 全部通过，并登记新的 GPU/run prefix/budget。

## 终止条件

若最小 probe 不能在 non-repeat 表面稳定改善正确候选 margin，或该改善可由普通
target attention / Denoising Attention 退化版复现，则停止 C05，不增加模块挽救。
