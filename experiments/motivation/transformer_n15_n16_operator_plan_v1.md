# N15/N16 residual-composition and RMSNorm operator plan

状态：2026-07-20。N15/N16 是在 N11（QK logits）、N12（SwiGLU gate/up）、N13（Q/K/V
projection）和 N14（embedding）之后登记的下一组内部 operator 诊断。它们不修改冻结
ranker、训练目标或 evaluator，也不把 patch 直接升级为 transfer architecture。

## N15：branch residual composition

### 要回答的问题

现有 post-block state patch 能看到一个状态变化，但它把 branch increment、残差相加和
后续 MLP 混在一起。N15 在同一次 forward 中保留 incoming residual `r` 和已计算的 branch
increment，只改变 decoder 的系数：`r + alpha*a`（attention）或 `u + alpha*m`（MLP）。
这能把“attention/MLP 产生了不同张量”与“该张量在 residual composition 中被放大、抵消或
翻转”分开。

### 固定设计

- Q2/Q3，blocks 13/20/27，full/null，固定 native readout rows；不按结果挑层或 token。
- 分别干预 attention branch 和 MLP branch，模式为 identity、0.5x、2x、sign-flip、zero。
- hook 只在 self-attention 或 MLP 的 branch 输出处执行；其余 Q/K/V、RoPE、mask、SwiGLU、
  RMSNorm、下游权重和 readout 保持 native。
- identity 必须返回原 tensor，避免 BF16 clone/write 漂移；active 条件记录 branch 增量
  的最大实际变化与 block fire count。

### 判定边界

N15 的主要量是 operator perturbation 对 full/null gap 与 target margin/NDCG 的改变；它不
声称 residual 加法本身是唯一原因。只有 alpha=1 identity、same/wrong-history、random
direction、反向 removal、Q2/Q3 复制和 shared evaluator 全通过，才可把 residual composition
列为候选瓶颈；否则保持 unresolved。

## N16：RMSNorm variance/gain

### 要回答的问题

input RMSNorm、post-attention RMSNorm 和 final RMSNorm 的状态观测不能说明是 variance
rescaling 还是 learned gain 在改变传递方向。N16 从模块输入重算 native RMSNorm，只改一个
因子：variance inverse-scale 或 gain；另一因子和所有下游参数固定。

### 固定设计

- scope 固定为 block 13/20/27 的 input/post-attention norm，以及最终 norm；不增加层、head
  或 hidden coordinate。
- 模式为 identity、variance 0.5x/2x、gain 0.5x/2x/sign-flip、zero；每个 scope 都保留
  full/null 和 same/wrong-history controls。
- variance 与 gain 使用同一 hidden input、同一 epsilon、同一位置集合；FP32 重算只用于
  operator 结果，identity 仍交给原生输出。
- 除 score 外记录 selected-row RMS、输出 norm、方向 cosine 和低精度重组误差，作为机械
  audit，不把范数变化本身当作机制效果。

### 判定边界

必须先通过 norm 重组、identity、完整有限 coverage、Q3 shared-prompt 和 wrong-history
controls。只有 variance-only 与 gain-only 的 direction/utility 对比在 Q2/Q3 同向且通过
预注册 BH family，才能说 normalization operator 参与 transfer gap；单纯的 post-norm
state response 仍只能是 localization。

## 四卡排程与停止点

N15/N16 不抢占当前 D2/D5 四卡波次。N14 evaluator 完成后，使用两个四卡波次：第一波
GPU0/1/2/3 分别跑 Q2-N15、Q3-N15、Q2-N16、Q3-N16 的 block-13/20 shard；第二波复用四卡
跑 block-27 和 final-norm cells。每个 shard 独立 resume 目录，连续 job ≤13,500 秒；等待
队列必须先做 GPU ownership audit，禁止同一物理卡并发两个模型。

N15/N16 只补齐“operator 是否导致信号被抹平/反转”的证据，不授权实施补偿层、转移架构或
打开 source test。若两波仍无法在 branch tensor、composition 和 normalization 之间形成
一致方向，最终报告明确写 `unresolved`，不继续 outcome-driven 地增加组件。

## N17 之后的固定扩展边界（先登记，不与当前波次抢卡）

架构审计仍显示若干接口只有几何/配置证据，不能被当前 N15/N16 覆盖：

1. `q_head_rmsnorm_variance_rescale_and_gain` 与 `k_head_rmsnorm_variance_rescale_and_gain`：
   在 q_norm/k_norm 输出上分别做 variance-only、gain-only 和 identity，保持 RoPE、另一侧
   Q/K、V、mask、softmax、o-proj 不变；这是 attention 内部 normalization 的下一优先级。
2. `gqa_query_to_kv_grouping`：在 attention interface 的 repeat-KV 边界做预注册 group
   permutation/identity，保持 Q/K/V 数值、head 数、mask 和 o-proj 不变；只在 Q2/Q3 都通过
   identity、wrong-history 和 permutation negative control 后才解释为 GQA 拓扑证据。
3. `q3_q_lora_scaled_adapter_injection`、`q3_v_lora_scaled_adapter_injection` 与
   `kv_cache_phase_boundary`：分别隔离完整 adapter contribution 和 Q1 prefix/continuation
   cache boundary；现有 LoRA rank/path 与 Q1 trajectory 只能作为参数/路径描述，不能替代这些
   operator tests。

这些方向暂不启动，直到 N8–N16 的 evaluator 结果和 H0–H5 矩阵写入报告；若 N15/N16 已能把
   transfer failure 定位到 composition 或 norm，N17 只作为预注册的交叉验证，不按结果继续
   扩大层、head 或 position family。
