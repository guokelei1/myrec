# 2026-07-11 架构组合经验与下一步

状态：**C01--C16 的历史终局；后续搜索已重启。C17 纸面拒绝，C18--C20
synthetic 均 0/3 失败，C21 的真实 train-only 路径信号门和 C22 的 filtration
matched-control 门也失败，C23/C24 的真实 label-free load-bearing 门继续失败，当前没有通过
门槛的候选。** 本文区分三类结论：有效科学失败、工程/协议无效运行、尚未完成；
不能把它们统称为“所有假设都被证伪”。

## 当前证据账本

| 候选 | 当前结论 | 能否判断机制有效性 | 主要原因 |
|---|---|---:|---|
| C01 counterfactual contract | 实现运行无效 | 否 | 使用了错误的 D2p anchor；数字不能用于模型胜负 |
| C02 history hyperadapter | 有效 train-internal 负结果，停止 | 是，限于冻结 CHHT primitive | 单次机械续跑完成五路训练；non-repeat 与控制 margin 均为 0，四类 corruption core-norm ratio 均约 1；2/6 gate 通过，dev 未打开 |
| C03 triadic transport | 复杂度/预算停止 | 否 | 7h25m 仍无 checkpoint；严格三边传输尚未支付计算租金 |
| C04 prefix-delta LM | 负结果，停止 | 是，限于该 primitive | query-only/null base 未守住；non-repeat delta 为负；corruption 不敏感 |
| C05 target-attention probe | 负结果，停止 | 是 | 训练后几乎只产生 candidate-common translation，内部 NDCG 改变为 0 |
| C06 local-Hodge flow | 真实 label-free A0 负结果，停止 | 是，限于该 primitive | 1,200 请求中 order/top-10 change 均为 0；非平凡 delta-range 比例为 0；A 标签未打开 |
| C07 signed-kernel attention | 学习型合成负结果，停止 | 是，限于该 primitive | S mean 0.654，低于 TARGET_NULL 0.750 与 CENTER0 0.744；corruption specificity 失败 |
| C08 reversible memory | 学习型合成负结果，停止 | 是，限于该 primitive | 0/3 seed；repeat 未保护；2/3 seed 输给普通 attention；corruption 保留过高 |
| C09 cross-view agreement | 学习型合成负结果，停止 | 是，限于该 primitive | 0/3 胜简单控制；constant-value 等价；corruption 保留约 92%–95% |
| C10 predictive evidence write | 学习型 GPU 合成负结果，停止 | 是，但绝对难度有生成器 caveat | 3/3 non-repeat gain 约为 0；比 matched centered attention 低约 0.049--0.052；全局预测分布压缩丢失 event-level interaction |
| C11 eventwise predictive write | 锁前代数拒绝 | 无需运行 | tied linear decoder 的 log-ratio 经 candidate centering 后精确退化为 hidden similarity，只差固定温度 |
| C12 candidate-prefix predictive write | 实现前复杂度拒绝 | 无需运行 | candidate-specific normalizer 避开 C11 归约，但精确代价为 `Omega(C(H+1)TVd)`；去掉该项又退回 hidden similarity |
| C13 set-relative RMS write | 实现前机制拒绝 | 无需运行 | scalar RMS 只是 request-adaptive rescale；full whitening 优先放大弱/噪声模，破坏 abstention，且 novelty 不足 |
| C14 abstaining support-mass attention | 实现前代数/新颖性拒绝 | 无需运行 | 任意非负 subprobability write `w=rho p` 与 zero-NULL softmax/attention×gate 在 forward 与 Jacobian 上同构；强近邻已覆盖 |
| C15 candidate-conditioned value write | 实现前代数/新颖性拒绝 | 无需运行 | 双线性/低秩 pair value 可移出聚合并退化为 pooled-value FiLM/hyperadapter；不归约的 joint nonlinear 形式就是通用 edge-conditioned message passing |
| C16 mixed-gradient energy write | 实现前代数/新颖性拒绝 | 无需运行 | candidate gradient 是 tied cross-attention/Hopfield/ET；candidate-axis competition 是 Slot Attention；保守 mixed Hessian 又退化为标量势梯度 |
| C17 evidence ledger | 实现前代数/新颖性拒绝 | 无需运行 | learned ledger 是 Edge Transformer；chain-rule ledger 是 attribution；score feedback 是 gate/C06 flow |
| C18 evidence-constrained order | GPU synthetic 负结果，停止 | 是，限于冻结 operator | repeat 3/3 为 1.0 且约束精确，但 non-repeat 3/3 与 base 等价，安全投影不能产生 transfer direction |
| C19 oriented lag | GPU synthetic 负结果，停止 | 是，限于冻结 operator | successor/repeat 可学，但仅 1/3 稳定胜结构控制；一 seed corruption 几乎全保留，且生成器暴露 base shortcut |
| C20 history-transition cone | GPU synthetic 负结果，停止 | 是，限于冻结 operator | 多转移系数与正反符号见证均活跃，但 supported 仅 0.226--0.358、三 seed 均输 pooled/span，shuffle 保留 63%--84% |
| C21 contiguous path closure | 真实 train-only 负结果，停止 | 是，限于冻结 path operator | 写入活跃并改序，但相对 D2p `-0.0000643`，显著输 one-step `-0.0006272`，wrong/shuffle 不特异；未进 dev/test |
| C22 evidence filtration | GPU synthetic 负结果，停止 | 是，限于冻结 filtration | 零 Jacobian/utility/corruption 全过，但 repeat 与控制全平，supported 未稳定胜 dense/parallel/final projection；未读真实数据 |
| C23 recurrence-reset survival | real train-only label-free A0 负结果，停止 | 是，限于 reset-suffix primitive | 写入活跃且安全，但 49.3% 请求被非平凡 suffix shuffle 时三 seed 最多仅 0.42% correction 改变；internal-A 标签未开 |
| C24 multi-recurrence competition | real train-only label-free A0 负结果，停止 | 是，限于 generic set-attention competition | primary 相对 item-only 改变 38.0% 排序，但删除跨候选边后三 seed 均 0/600 改序；排序来自独立 recurrence calibration，internal-A 标签未开 |
| C25 anchored Möbius interaction | real train-only label-free A0 负结果，停止 | 是，限于 pooled-state pure-three-way primitive | 只改变 2/1,200 top-10；wrong history 三 seed 只改 1.25%--1.83% 排序且 0 top-10；internal-A/delayed-B 标签未开 |

因此，当前没有一个 proposed architecture 得到可用于论文的正面真实排序证据。
C01 仍是 anchor-invalid，C03 只关闭复杂度可行性，不能写成排序机制证伪；C02 的
机械缺陷则已在不改科学设置的续跑中闭环，现可作为冻结 CHHT 的有效内部负结果。
其余最明确的科学负结果来自 C04--C10；其中 C10 的相对控制失败有效，但其 base
绝对难度因 positive-variant shortcut 不能作为干净的定量估计。

## 跨方向已经稳定下来的经验

### 1. “数学上不同”不是有效归纳偏置

C02 的 Cayley update、C03 的 transport、C06 的 Hodge flow、C08 的可逆群交换子都
有清晰数学结构。C08 进一步给出了 endpoint-memory 无法复现的构造见证，但学习后
仍不稳定地输给普通 attention；C02 则完成了真实内部训练，却把几何半径和 score
residual 同时推到上界而没有形成候选相对方向。以后每个新算子必须同时支付三份 rent：

1. 相对最近邻控制的可学习增益；
2. exact-repeat 或已知可靠证据不退化；
3. wrong/shuffle/query-mask 等 hard corruption 不复现 clean gain。

只通过代数性质、参数移动、能量差或非归约见证，不再进入真实数据。

C21 又补上真实状态层面的反例：多步路径算子有足够幅度、改变 46.0% 请求排序且
数值合同全通过，但方向仍不与标签稳定对齐；one-step 显著更好，说明增加路径长度
是在累积未经支持的方向，而不是恢复组合偏好。以后不得把“真实写入活跃”替代
“相对简单控制有稳定增益”。

### 2. “模型对历史有反应”不等于“历史改变了候选相对顺序”

C01 学到了非恒定 certificate，C02 可产生内部矩阵变化，C05 参数明显移动并略降
训练 loss，但 C05 最终是几乎完全的 candidate-common translation。以后必须在看
NDCG 前先检查：

- 每请求 score delta 的零和/中心化与最终上界；
- within-request delta range 和真实 order/top-k change；
- clicked-minus-unclicked delta，而不是 update norm；
- 同一 checkpoint 去掉关键 primitive 后，候选顺序是否真的改变。

C06 又补充了另一种 collapse：严格零和、有限、有界、deterministic、no-history exact
都可以通过，但 learned residual 的最大幅度只有 `1.83e-5`，最终仍是 `0/1200`
order change。以后 A0 必须同时设置**下界**（delta range/order/top-k change）和安全
上界；仅靠 conservation 不能证明模块是 load-bearing。

只读 checkpoint 诊断进一步限定原因：四路 global residual scale 已从零打开到
`0.003--0.010`，但 delta RMS 仅 `1.5e-7--1.0e-6`，而 D2p 相邻分差中位数为
`0.01746`，相差约 1.7 万到 9.4 万倍；投影层几乎停在 Xavier 初值，fit loss 与
冻结 base 的差异小于 `3.5e-7` 且略差。因此正确结论是“zero-init 单标量造成梯度
饥饿，加上算子/中心化幅度塌缩，C06 没抽取出证据”，而不是“数据中已证明没有
历史信号”。

C02 的有效续跑展示了相反的幅度失败。所有 history-present candidate 的 skew core
都达到固定 Frobenius 上界 `0.35`，score residual 约为共同的 `+1.5`，但 non-repeat
请求的平均 within-request delta range 只有 `2.44e-6`，共享 tie-break 下仅
3/1,112 改序且 0 个改变 top-10。true/wrong/shuffle/coarse/query-mask 的 core norm
又全部相等到约 `1.0` 比率。因此必须同时检查**绝对幅度、候选相对幅度和证据特异性**：
太小会像 C06 不承重，太大但 common-mode 会像 C02/C05 只饱和而不排序。

进一步的 checkpoint 诊断表明这是三重饱和链，而不是参数未学习：`rho` pre-tanh
绝对值均值 `16.32` 并 100% 饱和，raw skew norm 均值 `5.6407` 又使全部 core 被
投到固定半径 `0.35`，raw score residual 绝对值均值 `5.834` 最后被 `1.5*tanh`
压成共同的 `+1.499974`。corruption loss 比较的是 post-cap norm，所以 true 与四种
twin 都落在同一球面；listwise/preservation 又对请求常数平移不敏感。以后所有
门槛必须报告 pre/post activation 饱和率、candidate-centred energy 与有效导数下界，
fidelity loss 必须直接约束 margin/write direction，而不是 post-cap norm。

### 3. 强 base 与 no-history 必须是结构输入合同

C01 的 anchor 语义错误、C04 的 null ranker 漂移说明“delta 为零”不等于“回到合法
base”。所有后续门槛固定为：

```text
candidate hash + base checkpoint/config/score hash
  -> key-alignment bitwise parity
  -> no-history pointwise score/rank parity
  -> 才允许创建 optimizer
```

候选不能自行近似重写 D2p，也不能期待 loss 学出 fallback。

### 4. exact recurrence 与 cross-item transfer 必须分开验收

exact identity 容易学习，但中间层“看见 exact”不能保护最终排名。C04、C08 都说明
复杂 semantic path 会冲淡可靠 recurrence。下一完整架构应有同一 Transformer 内部
的单调 exact path，并要求最终 repeat 排名相对 item-only non-inferior；non-repeat
收益必须单独超过 base，不能由 repeat 均值掩盖。

### 5. corruption 必须约束正确的排序方向

降低 attention mass、certificate energy 或 memory norm，只证明反应强弱不同。
真正的门槛是：true history 改善正负候选 margin，而 wrong-user、event replacement、
query-mask 和真实需要顺序时的 shuffle 不能产生同方向增益。easy corruption 不能和
hard twin 池化后平均过关。

### 6. 在证明真实 transfer 可学前，不再增加自由度

历史模块越复杂，越容易出现三种假成功：容量收益、common-mode 响应、或合成生成器
特调。新候选必须先在固定表示、固定 base、单 seed train-internal 机制门槛上胜过
普通 centered attention/direct gate；未过门槛时不实现完整 LM、不访问 dev。

### 7. 完整执行链本身也是实验对象

正式 attempt 前必须覆盖最大真实 batch、全 no-history/全 repeat/全 non-repeat、空
corruption mask、checkpoint reload、report serialization、标签分阶段打开以及 evaluator
硬阻断。否则工程缺陷会消耗科学预算，C01/C02 的结论会重复发生。

### 8. token-likelihood 路线存在结构性三难

C10--C12 给出了一条比单次负结果更清楚的边界：

1. 先把 history 池化成一个预测分布，计算便宜，但 C10 丢失 event-level interaction；
2. 保留 event 维度、使用 candidate-independent predictor，C11 的 log-partition 在
   candidate centering 后被精确消掉，只剩 hidden similarity；
3. 用 candidate prefix 让归一化真正依赖候选，C12 在代数上成立，但 exact full-vocab
   normalizer 的代价随 `C(H+1)TVd` 增长，且这个最贵的项正是唯一不归约的项。

因此，不再把“token likelihood”本身视为免费创新。若没有新的、精确且通用的低成本
normalized decoder，下一路线应回到 Transformer 的 candidate/history residual-write
接口，直接验证局部交互，而不是用昂贵的 vocabulary normalization 绕一圈。

### 9. 不能用无条件归一化把“没有方向证据”变成大残差

C06 证明 residual 太小，C13 则证明直接做 candidate-set RMS/whitening 不是合格修复。
scalar RMS 对每个请求严格只是正标量重缩放，不改变 evidence direction；full whitening
虽然能转动/均衡子空间，却对最弱奇异模给最大增益，小 `epsilon` 会把 wrong-user/噪声
方向强行放大，大 `epsilon` 又退回固定 scalar。安全上界只能限制总能量，不能证明方向
正确。下一候选必须在 **event/head value-write 之前**验证方向一致性，同时保留明确零点，
而不是在候选集合输出端把任何非零响应归一成“有用证据”。

### 10. “允许 attention 拒写”是必要合同，但本身不是新 primitive

C14 把 real-event 总 support `rho` 与 allocation `p` 分开后，任意非负
subprobability write 仍精确等于 `rho * Attention_p(V)`，也等于追加 zero-value
NULL 的普通 softmax；平滑坐标变换后 Jacobian/HVP 也一致。sigmoid attention、
head-specific gated attention、Multiscreen absolute screening、sparsemax/entmax 以及
C03 dustbin 已覆盖相同或更强的可拒写机制。因此 abstention/no-history zero 应作为
所有后续架构合同和强对照，而不能单独承载创新 claim。下一 primitive 必须改变
history write **携带的信息或方向**，而不只是其非负质量参数化。

### 11. 改写 value 方向仍需要一条新的结构定律

C15 检查了在 attention 聚合前令每个 history value 依赖 candidate 的路线。若
`Phi(z,h)=M(z)h` 是线性或双线性映射，则
`sum_j p_j Phi(z,h_j)=M(z) sum_j p_j h_j`，严格等价于对普通 pooled value 做
candidate-conditioned FiLM/hyperadapter；低秩分解不改变这个结论。加入
`Phi(z,h)=U sigma(Az+Bh)` 一类联合非线性虽能产生相同 aggregate、不同输出的见证，
但它已经是通用 dynamic-filter/edge-conditioned message passing。无约束 pair MLP
只增加 `C x H` 容量与计算，无法形成可证伪的新归纳偏置。因此下一候选若仍修改
`V/W_O`，必须先提出一条从 motivation 导出的 pairwise value 结构定律；不能把
“candidate-conditioned”本身当作创新。

### 12. “能量梯度提供方向”也不是新的逃生口

C16 对最后一个自然后继做了纸面归约。双线性 compatibility 的 candidate gradient
精确等于 value 与 score derivative 绑权的 cross-attention/modern Hopfield；一般
非线性保守写入仍是 Energy Transformer/Hopfield--Fenchel--Young 的标量能量梯度。
把 softmax 改到 candidate 轴只是 candidates-as-slots 的 Slot Attention 分配。
mixed Hessian 与 candidate-independent event direction 收缩时，等于另一个标量势的
candidate gradient；允许 candidate-dependent direction 时，要么补全乘积法则后仍
回到标量势，要么不再保守。`softmax-uniform` 又已由 Differential Transformer 与
ZeroS 覆盖。以后不再把 gradient、Hessian、energy、softmax-axis change 或 zero-sum
命名本身当作新 primitive。

## 本轮执行终局

1. **C07**：已按双层 hash lock 唯一执行并失败；关闭，不进入真实数据/GPU/dev/test。
2. **C09**：已按双层 hash lock 唯一执行并失败；关闭，不进入真实数据/GPU/dev/test。
3. **C06**：使用完全排除 C05 的新 cohort；先冻结 selection、D2p/base、fit-only
   labels 和代码 hash，再比较 local-Hodge、`t=1`、direct learned gate、matched
   centered cross-attention。G0、四路 GPU smoke、数值修复与四路固定两轮训练均完成；
   label-free A0 因 `0/1200` order change、`0/1200` top-10 change 和零非平凡
   delta-range 比例而终止。A 标签、A1、B、escrow、dev/test 均未打开；C06 关闭。
4. C08 已终止，不重跑、不放宽 permutation/corruption/repeat 阈值，也不进入真实数据。
5. **C10**：GPU 3 上 3 seeds、6 modes 的冻结闸门已唯一执行并失败；不重跑、不调参、
   不进入真实数据。其后继只能是新的 fingerprint，例如把条件预测增益保留到
   candidate-token × history-event 接口，而不是再次压成一个全局 vocabulary 分布。
6. **C11**：虽保留了 event 维度，但在锁前 reduction review 中被证明与
   eventwise-hidden control 代数等价，因此永久拒绝且不运行 GPU。
7. **C12**：candidate-prefix normalization 通过非归约证明，却因 load-bearing 的
   full-vocabulary 复杂度在实现前拒绝；没有 model、runner 或 GPU 结果。
8. **C13**：set-relative RMS/whitening 在纸面 gate 被拒绝；没有实现或 GPU 结果。
9. **C14**：support-mass/null-attention 与已有机制精确归约且强近邻密集，纸面拒绝；
   没有实现或 GPU 结果。
10. **C15**：candidate-conditioned value write 的可计算形式退化为 pooled-value
    FiLM/hyperadapter，非退化形式则是通用 edge-conditioned message passing；纸面
    拒绝，没有实现或 GPU 结果。
11. **C02 机械续跑**：14/14 测试和双锁通过；GPU 1 上五路各两 epoch、3,942 steps
    完成。冻结 internal gate 仅通过 repeat 与 no-history 两项，另外 4 项失败；按新
    authorization 未生成 dev scores、未调用 evaluator，C02 关闭。
12. **C16**：mixed-gradient/energy family 被 tied attention、Hopfield/ET/HFY、
    Slot Attention、Differential Transformer/ZeroS 精确覆盖；纸面拒绝且未运行。
13. **C21**：先于架构实现执行真实 train-only 信号门；3 seed × 5 modes 全部完成，
    primary 与 D2p 等价、显著输 one-step，且 wrong/shuffle 不特异。关闭多步路径闭合，
    不调参、不选子组、不进入 Transformer/dev/test。
14. **C23**：原始 delta-rule 因 DeltaNet/SinkRec 同构在实现前否决；替代 RRST 完成
    3 seed × 4 mode GPU 门。13/14 个 A0 检查通过，但 post-anchor suffix 并不
    load-bearing，故在 internal-A 标签前关闭，不调层数/顺序损失、不进入 dev/test。
15. **C24**：generic candidate-set attention 只作为 signal-existence gate 执行，
    3 seed × 3 mode GPU 门完成。15/16 个 A0 检查通过，但同 checkpoint 删除
    candidate-candidate edges 后三 seed 都是 0/600 改序，故在 internal-A 标签前关闭；
    不调 attention 深度/残差幅度，也不把 set attention 包装成最终创新。
16. **C25**：共享势函数三阶离散导数精确删除低阶 residual bypass，并与 joint、
    candidate-history pairwise、direct trilinear 做参数/算力匹配。12 个 GPU fit 完成，
    但 top-10 activity 与 wrong-history ranking sensitivity 均失败；在 internal-A 前关闭，
    不修容差重跑。后继必须改变 token/representation 粒度。

## 下一架构的启动条件（C02 续跑与 C10--C16 之后）

用户于 2026-07-11 明确要求继续从 Transformer 内部架构出发并用 GPU 验证，因此
C06 的最小 real train-internal gate 与 C10 的冻结 synthetic falsifier 已执行。二者
都失败：C06 的安全结构成立但 learned write 不承重，C10 的全局 candidate-token
预测增益又稳定输给普通 centered attention。C02 的未决工程失败也已完成机械续跑：
CHHT 产生饱和 common-mode write，未得到 non-repeat 增益或 corruption specificity。
C11--C16 随后把明显的低成本后继逐一
关闭：eventwise likelihood 归约、candidate-prefix likelihood 过贵、输出端 RMS 缺乏
方向证据、support mass 是已有 gating、一般 pair value 是 FiLM 或 edge-conditioned
message passing，而 mixed-gradient energy write 是已有 tied attention/Hopfield/ET/
Slot/Differential attention。

因此不再通过连续枚举 gate、normalizer、scalar、NULL token 或通用 pair MLP 启动
GPU。新候选只有在结果前同时满足以下条件才可登记：

1. 在 `history-event -> candidate` 的信息或 value 方向上提出一条 motivation-derived
   结构定律，而不是一般可学习函数；
2. 给出不能被 ordinary centered attention、FiLM/hyperadapter、gated attention 或
   edge-conditioned message passing 重参数化的见证；
3. 用 label-free lower bound 先证明写入相对 base score gap 足以承重，同时保持
   no-history exact、候选中心化与 corruption specificity；
4. 训练初始化必须让 primitive 与其内部投影从第一步都能收到梯度，不能再依赖
   zero-init 单一全局门。

C10 冻结生成器存在 positive-variant shortcut，故其绝对 base 难度不能作为干净估计；
但同数据、同容量 centered attention 仍稳定多出约 `0.05` NDCG，因此相对失败不被
该 caveat 推翻，也不得通过后验修改生成器或阈值救回。

上述条件不是一个待执行的 C17 配方。**当前探索到此关闭**：没有通过门槛的 primitive，
也没有一个可直接实现且通过不可归约审计的新结构律。后续若重新启动，必须来自新的
motivation/data fact 或新的不可归约数学见证，而不是继续排列本轮已经关闭的组件。

## 决策原则

本轮不以“至少找一个正结果”为目标，而以找到最小、可复现且能胜过最近邻的机制为
目标。允许全部失败；不允许用新增模块、后验阈值、更多 dev 调用或 test 访问把失败
包装成成功。
