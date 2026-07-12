# C01--C80 架构搜索终局复盘

日期：2026-07-12（本轮执行跨越 Asia/Shanghai 午夜）
状态：**终局；C80 已关闭；不授权 C81、机械续跑或同机制 rescue。**

权威终局：[`reports/pps_c80_amazon_real_gate.json`](../../reports/pps_c80_amazon_real_gate.json)。
逐候选账本见 [`systems/README.md`](../../systems/README.md)。

## 一、直接回答

如果必须在“motivation 不够”和“design 原则有问题”之间选一个首因，我的判断是：

> **首因是 motivation/observability 没有建立一个需要新 Transformer 架构才能解决的、
> 可构造且跨数据集成立的缺口；design 原则随后把这个欠定问题变成了一个过度严格的
> 单次证伪流程，放大并固化了搜索困难。**

原始 motivation 足以说明“现有 PPS 流程有问题”：KuaiSearch 上稳定收益主要来自
exact recurrence，粗类别迁移无效，错误历史可能污染排序。但它不足以推出“应修改
Transformer 的哪一个计算部件”。它提供的是一组**负约束**——不能无条件混合、
不能破坏 recurrence、没有历史必须回到 base——而不是一个能产生正确 non-repeat
排序方向的充分统计量、监督对象或结构定律。

更决定性的是，我们直到 C76 前后才运行最基本的 positive control：普通 full-token
history-aware BGE cross-encoder。它在 Amazon 的未打开 1,200-request reserve 上已经有
`true-null +0.025298` 和 `true-wrong +0.035944`，而 pooled text 只有
`+0.001661`。冻结权重的 edge attribution 又表明，多层双向 Q--H、C--H 和
history-read-context 边都承重。这个结果首先证明的是：**早期 item/pool 表示接口丢掉
了信号，普通稠密 Transformer 已经能观察到它。** 它没有先证明普通 Transformer
存在一个必须靠新 primitive 修复的架构缺陷。此时继续被要求“必须做架构创新”，很
容易变成先有解法、后找问题。

因此，80 版仍未形成 CCF-A 级 motivation→design 链条，不是因为缺少足够多的算子
想法或 GPU 训练，而是因为：

1. **问题可观察，但架构缺口未被识别；**
2. **早期表示接口把真正信号预先压没；**
3. **搜索协议擅长排除伪阳性，却不适合发现和调好模型；**
4. **Kuai/Amazon/JD 的证据对象并不一致，跨域共同 claim 没有数据基础；**
5. **连续局部修补造成 protocol-level meta-overfitting，版本数增长快于有效搜索深度。**

这五项中，前两项是决定性根因，第三、四项是主要放大器，第五项解释了为什么会形成
长期循环。工程失误消耗了不少版本，但不是最根本原因。

## 二、“到 C80”实际完成了什么

仓库有 79 个候选目录：C01--C78 加 C80；C79 按终止预算有意未使用。它们不是 79 次
完整、独立、同等有效的架构实验。候选编号同时包含纸面归约、无效运行、机械
canonicalization、确认实验、signal probe、synthetic gate 和真实 A0/A1。更准确地说，
**C80 是 79 个审计分支的终点，而不是 79 个成熟模型的迭代终点。**

| 阶段 | 主要探索 | 稳定结论 |
|---|---|---|
| C01--C17 | contract、hyperadapter、OT、prefix delta、Hodge/signed/reversible/predictive/value/energy | C01 anchor 无效、C03 复杂度停止；有效运行主要出现 common-mode、饱和、vanishing write、corruption 不特异；C11--C17 多数被已知 attention/FiLM/message-passing/energy 归约 |
| C18--C30 | 最终排序约束、时序/路径、multi-repeat、Möbius、token bridge、pairwise contest、authenticated mediation | 安全约束能保护 recurrence，却不能创造 transfer direction；rank activity 不等于 utility；认证路径活跃但 A1 不增益 |
| C31--C44 | query transport、tangent/cone/barycentric、Amazon transfer、metric-coupled routing、partial logit flow | 首次出现小而真实的正方向，但 tangent/metric/halfspace 等具体 primitive 总被更简单控制追平或击败；Amazon utility 无法在 Kuai 保持 user specificity |
| C45--C55 | prequential/behavioral representation、ridge/influence/memory/covariance、token concept、joint context | 行为预测误差和 frozen pooled readout 不是缺失的价值表示；raw-BGE 活性经常来自弱 base，保留强 base 后优势消失 |
| C56--C69 | token competition、candidate budget、base-edge flow、memory slots、adaptive LM、counterfactual state、fast weight/free energy、behavior relation | 增大表示/排名活动仍多学成 generic query-candidate reranker；wrong-history dependence和唯一机制 rent 不稳定；C59 有强 true-wrong 但严重伤 base |
| C70--C75 | logged choice information object、query relay、semantic-conservative/frozen carrier | Kuai 可恢复的信息对象在 Amazon/JD 不存在；C74 token relay 很活跃但训练合同失败，C75 冻结 carrier 后几乎学不动 |
| C76--C80 | full-token observability、edge attribution、layer trajectory、frozen authentication、event-set symmetry、terminal real gate | 普通 full-token Transformer 首次明确观察到 user-specific signal；C76 仍含 candidate-only shortcut；C77 authentication 有效但位置顺序无依据；C78 set symmetry 有效且 triadic control 最强；C80 的真实预训练实现在标签前数值置换合同失败 |

版本数很大，但真正接近论文核心的独立正面节点只有少数。其余大量工作仍有价值：它
系统性排除了“数学形式不同就会有效”“响应历史就等于个性化”“改序就等于增益”等
错误推理；但这些负结果不能自动累积成一个正面架构。

## 三、motivation 到底缺了什么

### 3.1 它足以提出问题，不足以识别设计

M3/M4 最初的 oracle/headroom 论证被 Random channel 完整复现并超过：Random oracle
`0.4325` 高于原 oracle `0.4232`，Random-label AUC `0.6952` 高于 `0.6688`。因此
“按请求可预测地选择个性化通道”不是可靠正面事实。C5-R3 随后得到的可靠事实是：
exact item recurrence 强，category-only 近零，完整混合反而弱于 item-only；正式结果
也明确写了 `architecture_ready=false`、`authorized_primitive=null`。

“history evidence fidelity 不均等”是一个合理研究问题，但它并没有指定：

- 哪类 non-repeat history 含有可学习偏好；
- 正确排序方向来自 token、event、transition、choice set 还是 user identity；
- 哪个现有 Transformer 部件系统性失败；
- 需要的是新 attention、训练目标、数据字段，还是仅仅不做 pooling。

所以这一 insight 在机制层面其实**过宽**：几乎任何 gate、transport、memory、routing、
counterfactual delta 都能事后映射到“evidence fidelity”。版本不断增加，是因为
motivation 没有把假设空间压到一个可识别对象。

### 3.2 “motivation complete”被宣布得过早

在进入 C01 前，我们没有先回答最便宜的三个问题：

1. strict non-repeat 历史在强 base 上是否可观察；
2. signal 在 pooled item state 还是 full WordPiece interaction 中；
3. ordinary full-token Transformer 是否已经能区分 true、null、wrong。

直到 C76 阶段，前两个 pooled HSO 给出 Kuai negative / Amazon 微弱结果，随后 full-token
HSO 才在 Amazon 给出明确正面。这说明 motivation 实验不是“数量不够”，而是**顺序
错了**：先做了几十个利用机制，后补 source observability 和 layer/edge localization。

### 3.3 最新 motivation 指向 interface repair，不天然指向 architecture novelty

full-token HSO 与 edge attribution 最直接的设计后果是保留普通多层双向 joint
self-attention。删除 C--H 边后 `true-null=-0.037232`，完全隔离 history 时精确等于
null；Q--H 也贡献显著。最朴素结论是“不要预池化，不要只做 query relay，不要切断
joint graph”。这支持一个强标准 cross-encoder，却没有暴露一个标准 cross-encoder
的结构性 failure。

因此 C80 被要求同时胜过 ordinary full-token HSO，本质上是在没有先观察到 ordinary
model 缺陷的情况下，要求一个更受限的 sparse/authenticated graph 必须更好。这是
motivation 与 architecture-innovation 目标之间最根本的错位。

## 四、design 原则哪些正确，哪些用错了

### 4.1 应永久保留的原则

- train/dev/test 和 qrels 隔离、candidate hash、共享 evaluator；
- 强 base fidelity 与 no-history exact；
- repeat 与 strict non-repeat 分开报告；
- wrong history、query mask 等 provenance intervention；
- matched-capacity / nearest-neighbor control；
- outcome 前冻结 claim，失败后不调阈值救回；
- utility、specificity、mechanical safety 和 novelty 分开表述。

正是这些规则阻止了把 C32、C42、C59、C74 等局部正面包装成完整论文，也阻止了 C80
在核心合同失败后偷看标签。没有这些规则，我们可能早就得到一篇表面正面、但归因
错误的文章。

### 4.2 问题在于把确认规则当成发现算法

当前协议通常要求一个刚提出的 primitive 在一次固定配置中同时满足：所有 seed、
所有 hash fold、所有 corruption、所有 control、loss trend、数值 exactness、强 base、
跨域与 novelty。并且不允许常规学习率、训练时长、初始化或容量调优。这个制度适合
**最终确认**，不适合**模型发现**，因为它把下面三种情况混成同一个 `closed`：

1. primitive 确实错误；
2. 实现/数值合同错误；
3. primitive 可能有效，但固定优化配置没有学好。

典型例子包括：C32 总体 CI 为正却因一个约 200-request fold 为负关闭；C74 的
label-free activity 和 wrong-history sensitivity 很强，却因 all-mode loss trend 关闭；
C29/C58/C65 分别因微小 permutation 数值误差消耗 C30/C59/C66；C80 的所有训练、
梯度、fallback 和 candidate permutation 都通过，但 bf16 event permutation 为
`0.0319/0.0684/0.0394`，float32 诊断仍为 `3.34e-6`，高于 `2e-6`，所以标签完全未开。

这些关闭在协议上正确，却说明“候选编号”经常衡量的是工程合同分支，而不是架构
效用分支。

### 4.3 原则同时过度约束和约束不足

这是本轮最重要的 design-process 矛盾：

- **机制选择约束不足：**“candidate-conditioned evidence fidelity”允许太多数学实现；
- **单候选验收过度约束：**一个未经正常调优的实现必须一次通过论文级联合门槛。

结果是我们在大量 exotic operator 之间快速跳转，却很少对一个有明确 source signal
的强模型做正常 representation/objective/optimization co-design。“一个 falsifiable
primitive”也把现实中可能必须共同设计的 tokenization、attention graph、objective 和
base preservation 人为拆开。很多候选要么有独特形式但无 utility，要么有 utility
却与简单控制等价。

### 4.4 部分 corruption/gate 的 construct validity 不足

- shuffle 长期被当作应摧毁收益的 corruption，但 full-token HSO 后来显示 order
  不是承重变量；此前的时序/路径路线因此建立在未验证前提上。
- wrong-user 是 identity specificity 控制，不是纯 null；共享语义/popularity 可能仍
  有用。它适合限制 personalized claim，但不能单独定义全部 utility。
- order/top-k activity 只能排除全零，C28/C59 证明高 activity 可以显著伤害 NDCG。
- synthetic teacher 有时包含捷径或与机制过度匹配；通过/失败不能稳定预测真实 ranking。
- arbitrary loss-window 和 `1e-6/2e-6` 数值阈值有时比真实排序问题更先决定候选生死。

这些不是放宽科学标准的理由，而是要求把 gate 分层：先验证可学性，再验证真实 utility，
再做机制归因，最后做论文级 confirmatory audit。

## 五、表示、数据与统计为什么构成硬上限

### 5.1 早期 pooling 不是中性工程选择

C01--C51 的大部分搜索围绕 frozen/pooled item states、profile 或最终 residual 展开。
连续负结果一度被解释为“没有稳定 history direction”。但 Amazon 的量化对照是：

| 接口 | strict non-repeat true-null NDCG@10 |
|---|---:|
| pooled text HSO | +0.001661 |
| ordinary full-token HSO | +0.025298 |

差异超过一个数量级。换言之，很多架构不是在“寻找正确方向”，而是在已经删除
cross-token composition 后试图从压缩状态恢复它。任意新 transport/gate 都无法恢复
输入接口没保留的信息。

### 5.2 三个数据轨不是同一个机制表面

- KuaiSearch repeat-present 约 23.44%，Amazon-C4 约 0.51%；原 motivation 的 exact
  recurrence 核心在副轨几乎不存在。
- logged historical slate 在 Kuai 可恢复 96.56%，Amazon 为 0%，JD 公共格式也没有；
  C70 因此无法形成共同 operator。
- full-token 正面 source 目前只在 Amazon 被严格确认；Kuai pooled HSO 为负，没有对应
  的 full-token confirmatory result。
- JD 尚未标准化且没有 plaintext text，C80 的 frozen WordPiece semantic anchors
  无法直接支持原定 anchor claim。

所以“同一架构必须跨 Kuai/Amazon/JD”在当前数据合同下不只是标准高，而是输入对象
不一致。反过来为每个数据集设计不同 evidence object 又会违反 anti-dataset-tuning。

### 5.3 fresh-role 过度碎片化降低了学习与判断能力

大量正式 A 只有 600 或 1,200 requests，固定三 fold 后每 fold 约 200--400；C80 只剩
365 个 user-disjoint fresh requests。严格 role isolation 很好，但每个微小 successor
都消费新 cohort，使模型训练、机制选择和最终确认共享同一有限数据预算。

这并没有消除全局自适应多重尝试：后继 hypothesis 仍然根据前一 cohort 的失败形态
选择。它只是把 meta-overfitting 从显式 dev 调参变成了“跨 cohort 的 protocol 调参”，
同时让每次 CI 更宽。正常研究流程应允许一个明确记录预算的 development surface 做
探索，把一个大 validation/test 留给最终模型，而不是为每个 primitive 切一个小型
“新鲜”角色。

## 六、搜索过程为什么会形成死循环

本轮没有写 `if dataset == ...` 的显式数据集规则，但存在明显的**规格层过拟合**：

- C29→C30、C58→C59、C65→C66 修 numerical/canonicalization；
- C31→C32→C33 修 query-parallel/tangent 与确认；
- C34→C35→C36→C37 依次修 always-on、centering、shrinkage；
- C73→C74→C75 依次修 relay identifiability、semantic carrier drift；
- C76→C77→C78→C80 依次修 candidate shortcut、query authentication、event order。

每一步都合理地修复最近一次失败，但连续局部 pivot 会把模型越来越贴合**我们自己
定义的 gate surface**，不等价于越来越贴合真实标签函数。预注册能防止在同一候选内
偷改阈值，却不能防止 79 个候选之间的 adaptive hypothesis selection。

同时，“必须是架构创新”的压力奖励形式上不可归约的算子。C11--C17 的纸面审计说明
很多名字最终仍是 attention、FiLM、message passing、Hopfield/energy；真正不归约的
形式则通常计算过重或难优化。我们在 novelty 审计上投入了大量深度，但没有先建立
一个普通 Transformer 的明确、可复现 failure mode。这使搜索越来越数学化，却没有
更接近可预测目标。

工程问题进一步消耗有效深度：C01 错 base、C02 empty-mask NaN、C03 无 checkpoint、
C04 gate 后错误调用 evaluator、C47 label-scope incident、C71 fresh role 零正例、
C74 状态 flag 缺陷，以及多次 bf16 permutation 问题。它们不是科学反例，但占用了
候选编号、时间和未打开 cohort。

## 七、最接近正面的结果为什么仍不是论文核心

| 结果 | 正面部分 | 缺失部分 |
|---|---|---|
| C32 tangent query transport | Kuai +0.004268，overall CI 正，3 seed 正，true>wrong | 一个 fold 负；C33 fresh confirmation CI 跨零，且只比 unprojected +0.000583 |
| C42 coupled-content control | Amazon 相对 C38 +0.010250、true>wrong +0.035234，CI 均正 | 对 semantic/asymmetric routing 的唯一 rent 不稳定；它本来是 control，不是已冻结 primary |
| C43 same operator on Kuai | +0.004124，CI 正 | true-wrong 仅 +0.000487 且 CI 跨零；shifted/single-wide/selection-only 追平或更好 |
| C59 semantic candidate budget | true-wrong +0.027817，CI 正 | 比强 base -0.070103，controls 基本等价 |
| C74 semantic relay | synthetic 强，真实 A0 rank/wrong-history activity 强 | all-mode training trend 失败，validation label 未开；C75 冻结 carrier 后几乎不学习 |
| ordinary full-token HSO | Amazon true-null/true-wrong 均稳定正，edge attribution 清楚 | 是标准 dense Transformer evidence，不是新的架构 primitive；Kuai/JD 未确认 |
| C80 terminal graph | 15 fits 全完成，训练/梯度/anchors/no-history/candidate permutation 全通过 | event permutation 3/3 失败，fresh 标签未开，utility 完全未知 |

跨全部阶段反复出现同一个“三角缺口”：

1. 有 utility 时，简单控制常常同样有效；
2. 有 unique mechanism 时，utility 或 learnability 不成立；
3. 有 utility+specificity 的 Amazon 结果时，Kuai specificity/generalization 不成立。

CCF-A 级核心至少需要强 base 上的稳定 utility、user-specific evidence、相对最近控制
的独特 rent、跨数据集/切分复现和最终 test。当前没有一个候选同时满足这些条件；
C80 甚至未获准进入 utility gate。因此不能把“接近”累加成一个论文级模型。

## 八、反事实责任判断

| 如果当时改变一件事 | 对避免 80 版循环的预期影响 | 判断 |
|---|---|---|
| C01 前先做 pooled vs full-token HSO 和 edge attribution | 会直接淘汰大多数 pooled transport/gate，并把起点放在 ordinary joint Transformer | **最高影响；主要缺失实验** |
| 只放宽 all-fold/数值/loss gate | 可能让 C32/C42/C74 更早被称为正面，但仍无 unique rent 或跨域证据 | 中等；能减少假阴性，不能创造 CCF-A claim |
| 再想更多数学架构 | 已覆盖 attention/residual/memory/FFN/transport/fast-weight 等广泛空间，缺少 target 时边际价值低 | 低；不是主要瓶颈 |
| 给少数强候选正常 dev tuning 和更大 cohort | 可区分“优化失败”与“机制失败”，提高统计功效 | 高，但必须在 source observability 之后 |
| 先完成 JD 和跨域输入可用性审计 | 会更早发现统一 text/token/history primitive 的 scope 不成立 | 高；会迫使论文缩窄或更换数据 |
| 取消 label/test 隔离 | 会更快得到漂亮数字，但无法可信归因 | 负面；不应做 |

这张表说明：**motivation/数据观测顺序是首因，design/search protocol 是第二原因；
不是简单“再设计三个更聪明的模块”就能解决。**

## 九、哪些结论值得保留

即使没有 proposed architecture，以下结论足够稳健：

1. KuaiSearch 被测静态历史增益主要是 exact recurrence；粗 category transfer 不独立
   有效，完整混合会稀释 item-only。
2. no-history 必须结构性回到强 base，repeat 与 strict non-repeat 必须分开验收。
3. update norm、attention mass、rank activity、energy separation 都不能替代排序 utility。
4. pooled history state 在当前 strict non-repeat 问题上不足；Amazon 的 useful signal
   位于 full WordPiece joint contextualization。
5. Amazon ordinary full-token 模型需要双向 C--H、Q--H 和 history-read-context；
   event order 当前不是已建立的承重因素。
6. C77/C78 的 data-free 结果支持 frozen provenance admission 与 event-set symmetry 能
   阻断特定 shortcut，但 C80 没有提供真实 utility 证据。
7. query transport / learned routing 是目前最可重复的弱方向，但 tangent、metric
   coupling、halfspace 等命名 primitive 没有支付独特 rent。

不能保留的结论包括：“所有 semantic personalization 都无效”“所有 Transformer 都
失败”“C80 utility 为负”或“再也不存在可行架构”。被终止的是当前证据、数据合同和
搜索协议下的 architecture search，不是一个数学不可能性证明。

## 十、论文与后续边界

当前工作不能诚实包装成 CCF-A 级 proposed-architecture 论文。最接近可写的是一个
**measurement/negative-design study**：exact-repeat shortcut、pooled-vs-token
observability、attention-edge attribution，以及哪些 evidence-safe 结构仍无法带来
unique utility。但仅凭当前 Amazon 1,200-request HSO 和未完成的 Kuai/JD full-token
验证，它是否足够成为顶会论文也不能预先保证。

按用户终止指令，本轮动作是：

- 停止 C80 和全部架构搜索；
- 不打开 C80 fresh labels；
- 不创建 C81，不做 fp32/canonicalization/tolerance rescue；
- 不用 dev/test 挽救；
- 将本文作为 C01--C80 的正式经验总结。

如果未来另立项目重新开始，它不应叫 C81，也不应沿着当前局部链继续。新的 Phase 0
至少应先完成：

1. 在主轨 KuaiSearch 和至少一个可比副轨上复现 full-token true/null/wrong
   observability；
2. 先找出 ordinary full-token Transformer 的具体失败，而不是假设必须修改它；
3. 统一或显式缩窄跨域信息对象，先决定 JD/no-text 是否仍属于同一 claim；
4. 用一个有预算记录的 development surface 正常调好强模型，再用大而唯一的 holdout
   做机制和论文确认；
5. 将 learnability、utility、specificity、mechanism attribution、numerical safety、
   novelty 六个 gate 分阶段，而不是一次合取。

这不是下一架构计划，只是解释如果不改变研究问题和流程，为什么继续编号只会重复
同一循环。

## 终局一句话

> **我们不是“设计了 80 个好架构都被数据否定”，而是在一个尚未识别标准
> Transformer 架构缺口的问题上，用确认级规则快速审计了 79 个候选分支；早期
> pooling 隐藏了真正信号，晚期 positive control 又说明普通 joint Transformer 已经
> 能利用它。motivation 的构造性不足是首因，design/search 原则的使用顺序错误是主要
> 放大器，跨域数据与碎片化角色则让任何弱正面都无法长成 CCF-A 级证据。**
