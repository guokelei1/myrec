# N17--N20 Transformer boundary plan (pre-registered, inactive)

状态：2026-07-20。本文件只登记 N15/N16 之后仍未完成的内部 operator
边界；在 N8--N16 的 shared evaluator 终态和 closeout audit 之前不占用
GPU、不读取效应值、不启动正式 scoring。它不是 transfer 架构，也不允许
按结果选择层、head、group、position 或 LoRA path。

## 共同边界

- 继续使用 `full_confirm_preceding40k_v11` 的 8000 条 label-free
  internal-dev 请求；scorer 不读取 qrels，统一 evaluator 在完整性审计后读取
  冻结 qrels。
- Q2/Q3 使用同一冻结 checkpoint、native readout、token/position/mask、候选
  集合和 full/null/wrong-user 对照；Q3 额外保留三条 shared-prompt readout path。
- 所有干预必须有 exact identity、完整 finite coverage、same/wrong-history
  specificity、反向 removal、随机/范数匹配负控；机械失败不算科学效应。
- 每个 family 预先固定 endpoint 为 target-vs-best-competitor margin 与
  NDCG@10，request-level paired difference、normalized-query bootstrap 和
  family 内 BH；不把单一模型或单一层的结果升级为架构结论。

## N17：query/key head RMSNorm operator

### 问题

N15/N16 只能分解 block branch composition 与 block-level RMSNorm。它们不能回答
在 `q_proj/k_proj -> q_norm/k_norm -> RoPE` 之间，head-dimensional normalization
是否把可传递的历史信号重新缩放或改变方向。

### 固定干预

- Q2/Q3，blocks 13/20/27，分别在 `q_norm` 和 `k_norm` 输出做 variance-only
  `0.5x/2x`、gain-only `0.5x/2x/sign-flip`、zero 和 identity；Q 与 K 分开，
  不做 joint outcome-selected 组合。
- 保持 q/k projection 输出、另一侧 norm、RoPE phase、V、mask、softmax、o-proj
  和 native readout 完全不变；记录 pre/post norm RMS、方向 cosine、RoPE 后 Q/K
  几何和 score effect，范数变化只作 mechanical audit。
- 预注册 same-history、wrong-user、random-direction 与 output-norm-matched
  controls；Q3 对 shared prompt 的 yes/no 两条路径分别检查 identity。

### 证伪门

只有 variance 与 gain 的方向/效用在 Q2/Q3 同向、identity/recomposition/wrong-user
均通过，并且与 post-RMSNorm/N15 的结果区分开，才允许把 head norm 标为
`routing-normalization candidate`；否则保留为 geometry-only 或 unresolved。

## N18：GQA repeat-KV grouping boundary

### 问题

已有 attention edge/group 实验改变了边或 group 的内容，但没有隔离 16 query heads
映射到 8 shared KV heads 的拓扑。N18 检验是 grouping 造成的 transport 问题，还是
KV 内容本身的问题。

### 固定干预

- 在 repeat-KV 边界只改变 query-to-KV group mapping；Q/K/V 数值、head 数、mask、
  RoPE、softmax、o-proj 和所有 token/position 均固定。
- 条件固定为 identity、seeded cyclic permutation、seeded reverse permutation、
  deterministic random permutation；覆盖全部 8 个 KV groups 和其对应的 2 个 query
  heads，不按 group 结果挑选。
- 每个 permutation 同时保留 same-history、wrong-user 与 permutation-reversal
  controls。若 runtime 使用 cache，先限制为非 cache full-sequence Q2/Q3，另行登记
  cache 版本，不把两个 phase 混为一个结论。

### 证伪门

identity 必须 exact；只有 permutation effect 在 Q2/Q3 同向、非 permutation negative
control 不矛盾、且不被 q/k norm 或 RoPE family 完全解释时，才标记为
`gqa-topology candidate`。绝不能由单个 group 的异常推导 head 数或新的 attention 结构。

## N19：Q3 complete LoRA branch injection

### 问题

已有 LoRA A/B、B@A 几何和 rank-path 描述不能说明完整的
`base(x) + (alpha/r) B(A(x))` 分支在推理时是否真正改变 transfer score。N19 将
adapter contribution 与 base projection、q/k norm、GQA transport 分开。

### 固定干预

- 仅 Q3；全部 28 blocks 的 q_proj 与 v_proj adapter path 都登记，按固定 block
  shards 并行执行，不按效应值筛选；每条 path 的 contribution 在合并前插入。
- 条件为 identity、zero、0.5x、2x、sign-flip 和 output-norm-matched random
  direction；base projection、adapter input、另一条 q/v adapter、q/k norm、RoPE、
  mask、o-proj 和 native readout 固定。
- 记录 base、adapter、sum 三个张量的 RMS/cosine/recomposition residual；full/null/
  wrong-user 和三条 Q3 shared-prompt path 全部保留。

### 证伪门

必须先通过 exact re-add 和 all-28-path coverage。只有完整 contribution 的 removal/
scale 方向在 Q3 的 transfer margin 与 native utility 上稳定，并不能被 base 或 q/k
family 的负控解释，才可称为 `q3-adapter-branch candidate`；A/B 范数或 B@A 几何本身
不能通过此门。

## N20：Q1 KV-cache phase boundary

### 问题

Q1 的 prefix cache、candidate continuation 和 answer-token likelihood 不是 Q2/Q3 的
单次 forward。现有 trajectory 记录了状态变化，但还没有把 cache materialization、
cache reuse 与 continuation phase 分开作同请求因果对照。

### 固定干预

- 只在 Q1 的冻结 prefix/continuation serialization 中做 phase-matched controls：
  exact cache identity、同请求 cache rebuild、wrong-user prefix cache、prefix-cache
  replacement、continuation-only replacement；不改变 token IDs、cache positions、
  causal mask、candidate slate 或 answer labels。
- 分别报告 prefix hidden/cache RMS、continuation first-token state、完整 response
  mean log-likelihood 与 candidate order；cache replacement 不得跨请求借用未审计状态。
- 先做 full-sequence no-cache vs native-cache identity audit，再运行 phase interventions；
  identity 不通过时整个 N20 标记 mechanical non-result。

### 证伪门

只有同一 Q1 请求的 cache-preserving 与 phase-matched replacement 方向一致，并通过
wrong-user、token/position/cache-key integrity 和重复运行，才可作 Q1-scoped cache
boundary evidence；不得外推到 Q2/Q3，也不得把 cache workaround 当 transfer 方法。

## 排程与停止点

N17/N18/N19/N20 是固定的后续边界，不与当前 N15/N16 抢卡。若 N15/N16 已经在
composition 或 block RMSNorm 形成明确、跨模型且通过负控的 G/N 级定位，则先做最小的
N17/N18 交叉验证；若仍 unresolved，则按 N17 -> N18 -> N19 -> N20 顺序各跑一次完整
预注册 family，不追加新层/head/seed。所有结果写入新的 mechanism-stage report，不能
覆盖冻结的 first-round 或 N15/N16 manifest。

