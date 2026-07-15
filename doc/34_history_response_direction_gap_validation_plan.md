# LLM4Rec 历史响应—排序方向缺口验证计划

日期：2026-07-14

状态：**开放探索已获授权；本文中的冻结、门槛与停止条件只约束后续确认性证据，
不把 Lite 或任一先导实验的阴性结果外推为 Full/其他数据上的终局。test、独立
confirmation 与新架构仍未授权。**

探索结论（2026-07-15）：上述“方向缺口”作为普遍命题已被 Amazon-C4 与 JDsearch
收窄，不能再作为当前最终 claim。现有实测支持的是 base retention 与
candidate-relative history utility 无法被普通 joint full-token ranker 稳定共同控制；
详见 `doc/35_controlled_history_composition_motivation.md`。本文保留为原始预结果验证计划和
证据边界，不应被重新解释为已经通过的 Failure Card。

本文要验证的不是某个新模型，而是一个可能成为后续架构工作的前置事实：

> LLM4Rec 对历史产生响应，但不能稳定把这种响应转化为
> query-conditioned、candidate-relative 的正确排序方向。

这句话目前**没有被现有实验完整证明**。仓库已有的 full-token observability、
true/null/wrong-history 差异和少量 context-collision 信号，只能说明普通 Transformer
有时会读取历史、历史有时会改变输出。它们还没有排除以下更简单的解释：模型本身不够强，
变化只是所有候选共同平移，历史响应其实方向正确但效应小，数据中没有足够可预测的
个性化方向，或点击标签中的位置偏差让方向看起来错误。

因此，本计划把上述句子视为**待证伪的工作假设**，而不是新的项目结论。只有它在强模型、
真实搜索数据和严格反事实控制下成立，才允许从现象进入 Failure Card；即使成立，也还没有
自动推出任何具体 architecture。

### 0.1 探索层与确认层分开

本项目先运行一个可纠错的探索层，再把其中稳定、值得检验的规律转写成确认协议。

- 探索层允许依次或交叉观察 Lite、Full、KuaiSAR、JDsearch 与 Amazon-C4；允许根据字段、
  覆盖、统计功效、模型行为和实现问题增加诊断。Lite 结果无论正负都只描述 Lite，不能关闭
  Full；一个数据集没有信号也不禁止继续理解另一个信息对象。
- 探索层允许普通 baseline/full-token 模型训练和 development 诊断，但所有调用都登记，
  结果标记为 exploratory，不能进入最终显著性或论文主表。
- 确认层才冻结数据角色、population、primary endpoints、阈值、模型选择规则和统计检验。
  探索中看过的请求不能伪装成未观察的独立确认样本。
- 两层共同遵守历史时间因果、候选集合一致、评分代码不读取 qrels、共享 evaluator、test
  封存和架构前 Failure Card 等证据边界。

探索不是无记录地试到结果好看。每一步都记录：问题、动作、直接观察、至少两个可能解释、
尚未排除的不确定性、是否纠正此前判断、以及下一项最便宜的区分性探针。探索可以改变路线，
但不能改写已经发生过的观察。

---

## 一、先把论文动机改写成可验证命题

### 1.1 不验证无限范围的“Transformer 不行”

实验无法证明所有 Transformer、所有训练方法和所有数据上都不能正确利用历史。可被证据
支持的目标应收缩为：

> 在至少两个经过正常调优、query--candidate base 能力合格的 ordinary full-token
> Transformer 家族上，在两个具有真实 query、严格先验用户行为、曝光候选集和候选级标签
> 的产品搜索数据中，true history 会造成显著的候选相对排序变化；但在预先定义的
> strict-nonrepeat/context-demand 请求上，这些变化与正确标签方向的对齐度低，不能达到
> 预先冻结的最小有用幅度，也不能稳定优于 matched wrong-user history；与此同时，一个
> 不依赖新架构的历史信号 witness 能在同一信息边界内恢复部分正确方向。

如果这条有边界的命题成立，后续才可以讲：问题不是 Transformer 完全看不到历史，而是
ordinary joint modeling 缺少把历史证据转成候选间相对决策的稳定机制。

### 1.2 四个概念必须分开

| 概念 | 本计划中的含义 | 不能用什么替代 |
|---|---|---|
| 历史可观测 | true history 与 null history 导致输出不同 | attention 不为零、hidden state 变化 |
| 候选相对响应 | 历史改变同一候选集内部的分数差或次序 | 所有候选共同加一个 bias |
| 方向正确 | 历史诱导的候选间位移与候选标签优劣一致 | true-history 总分高、loss 下降 |
| 用户特异 | true history 的正确方向优于匹配的其他用户历史 | true 优于空历史但 wrong history 同样有效 |

只有四者按顺序成立，才存在“看见了，但没有稳定转成正确方向”的现象。

### 1.3 本计划不是 base-degradation 计划

历史模型在空历史下是否损伤 query--candidate base 会被测量，但只作为 adequacy/confound
检查。核心反事实使用**同一个训练完成的 checkpoint**，在同一请求上替换 true、null 和
wrong history，从而隔离运行时历史响应。另设独立训练的无历史强 base，用来判断 personalized
checkpoint 是否本来就没有学好基本排序。

如果主要失败只是 personalized checkpoint 的 null-history 能力很差，应先修训练和 base
preservation，不能把它包装成本计划所说的方向缺口。

---

## 二、证据链：要同时证明什么

```text
数据里存在可恢复的 query-conditioned 个性化方向
                 |
                 v
强 query--candidate base 与 ordinary LLM4Rec 均已充分调优
                 |
                 v
true history 引起非平凡的 candidate-relative 响应
                 |
                 v
响应与正确候选方向弱对齐，且 true 不稳定优于 null/wrong
                 |
                 v
该规律跨模型家族、split 和至少一个独立搜索数据复现
                 |
                 v
形成 Failure Card，之后才问是哪种计算缺口、如何修复
```

这条链中任意一环失败，都必须改变结论，而不是继续寻找更有利的切片：

- 没有候选相对响应：只说明模型未用历史，不能支持“响应转化失败”。
- 响应方向正确且达到有用幅度：核心动机被证伪，这是好结果，应停止为它设计架构。
- 没有简单 witness 能恢复正确方向：优先判断数据监督不足，而不是结构缺陷。
- 只有一个弱模型失败：是 baseline/recipe 问题。
- 只有一个数据集成立：是 dataset-specific finding，不能直接升级为通用 LLM4Rec story。

---

## 三、数据集选择：按“能否制造方向性证据”选择，而非按熟悉程度选择

### 3.1 数据准入条件

一个数据集要成为 binding evidence，必须在打开模型结果前通过以下 source audit：

1. 有真实 request query，而不是只有用户或 session。
2. 有目标行为之前、时间边界清楚的 user/session history。
3. 有真实曝光或召回候选 slate，至少包含可比较的正负候选，而不是为每个正例临时随机采样。
4. 有候选级点击、购买或分级反馈，且能保留原始曝光位置用于偏差控制。
5. 能形成不读取评价标签的 `history_present`、`strict_nonrepeat` 和
   same-query/overlapping-candidate eligibility surface。
6. 有足够多的重复 query 和候选重叠，使不同用户面对相同或相近信息边界时可能需要不同排序。
7. 用户、session、timestamp 和 item 标识足以防止历史越过目标时刻，且能做独立时间/用户切分。
8. 训练、development、confirmation 的有效请求量能够支持预先指定的最小效应检验。

不满足真实 query 的 MovieLens、KuaiRec、普通 next-item recommendation 和 MIND 类数据，
不进入核心证据。它们最多证明 context-conditioned recommendation，不能证明
query-conditioned personalized ranking。

### 3.2 推荐的绑定数据组合

#### 主数据：KuaiSearch Full，而不是只依赖 Lite

KuaiSearch 最适合做主验证，因为其官方数据同时提供真实搜索 query、产品标题/品牌/类目、
用户近期点击与购买、曝光候选及点击/购买标签；Full 版本的规模也更有希望形成自然的
same-query/candidate-overlap collision。官方字段说明见
[KuaiSearch repository](https://github.com/benchen4395/KuaiSearch)。

本计划需要为 Full 数据单独建立新 standardized version，不从旧 Lite 结果外推。主轨承担：

- plaintext 条件下的 ordinary LLM reranker 结论；
- 全量 strict-nonrepeat 请求上的响应—方向分析；
- same-query/overlapping-candidate 的自然 context-demand 分析；
- click 与 purchase 两种强度标签下的一致性检查。

#### 行为复制：优先 KuaiSAR Full

KuaiSAR 同时记录同一批用户的搜索与推荐行为；搜索记录包含 query token sequence、搜索
session、展示 item 和点击，推荐流可提供更长的严格先验行为。其 query 和 caption 是一致
映射的匿名 token sequence，字段与统计见
[KuaiSAR schema](https://kuaisar.github.io/detailed_statistics.html)。

它适合验证“候选相对方向缺口是否是一种行为建模规律”，但不承担“预训练语言语义没有被
利用好”的结论。也就是说，KuaiSAR 的作用是跨数据复制 functional failure，而不是假装
匿名 token 等同于自然语言 LLM 输入。

#### 预注册替补：JDsearch

如果 KuaiSAR 无法稳定重建搜索展示 slate、时间严格的历史或独立 confirmation，替换为
JDsearch。JDsearch 官方格式包含 query、候选列表、分级候选反馈、历史 query、历史商品、
行为类型和时间，但文本主要是匿名 term ID，且下载与 split 可用性要先审计。来源见
[JDsearch repository](https://github.com/rucliujn/JDsearch)及其
[dataset paper](https://arxiv.org/abs/2305.14810)。

替换规则只允许依据数据可获得性、字段完整度、污染风险和 outcome 前的功效估计；不能因为
KuaiSAR 上模型结果“不好看”再换 JDsearch。

### 3.3 非绑定的补充数据

- **Amazon-C4**：query 和商品文本丰富，适合做英文/语义 stress test；但官方 query 由
  长正向评论经模型改写，每条 query 只有一个正商品，常用候选池是同域随机负例。因此它
  不能作为自然搜索方向缺口的主证明。来源见
  [Amazon-C4](https://huggingface.co/datasets/McAuley-Lab/Amazon-C4)及
  [temporal user-history companion](https://huggingface.co/datasets/zhiyuanpeng/amazon-c4-user-purchase-history)。
- **Coveo SIGIR e-commerce challenge**：有 session query、返回商品和点击，适合在用户级
  数据失败时复制 short-term session-context 规律；但缺少稳定用户身份和原始文本，只能支持
  较窄结论。来源见 [dataset paper](https://arxiv.org/abs/2104.09423)。

这些数据不得与绑定数据混在一起做“多数投票”。它们只回答各自边界内的附加问题。

### 3.4 为什么这个选择最可能看见真实信号

要观察“方向错误”，不能只找历史很长的数据，而要找同一个 query 下多个候选确实存在用户
条件性反转的数据。KuaiSearch Full 提供自然语言和商品文本，便于强 LLM ranker；KuaiSAR
提供大规模真实搜索曝光与长行为上下文，便于提高 context-demand 事件的统计功效。两者的
互补性比再选一个“一个正例加随机负例”的文本推荐数据更重要。

---

## 四、数据构造与防止后验挑切片

### 4.1 三层人群，角色不同

1. **全体可评估搜索请求**：判断模型的基本 ranking adequacy 和 overall 代价。
2. **history-present strict-nonrepeat**：候选不直接出现在历史中，用于排除 exact item
   recurrence 对方向结论的垄断；这是主要 population。
3. **自然 context-demand population**：先按 normalized query、候选重叠、非空历史等
   label-free 条件配对不同用户请求；评价时再报告其中标签确实要求不同候选方向的 pair。

repeat 请求只作为正控制：如果模型连历史中直接出现的候选都不响应，说明 observability 或
实现有问题；但 repeat 成功不能替代 nonrepeat 的个性化迁移证据。

### 4.2 natural collision 的构造

对于两个用户请求，先在不看候选标签的情况下要求：

- normalized query 相同；若 exact-query 功效不足，只允许使用训练期冻结的 query cluster；
- 两个请求共享至少一对曝光候选；
- 两个用户的历史都严格早于目标请求且非空；
- 被比较候选不在任一用户历史中；
- source position、候选流行度和 session stage 能够匹配或分层控制。

评价器随后判断共享候选对是否在两个用户标签中呈现相反优先方向。核心指标不是只挑一个
“漂亮反转案例”，而是：context-demand 的发生率、普通模型能否随用户历史完成正确翻转、
以及 true history 是否比 exchanged history 更容易同时排对两个请求。

### 4.3 split 和标签隔离

- 以时间为第一边界，所有历史必须严格早于 target request。
- 同一 session 不能跨 split；对 collision 分析还要避免同一请求或同一候选事件重复进入两侧。
- training labels 用于训练、训练内功效估计和 witness 构造。
- development records 对训练/打分代码保持 label-free；所有评价只经过共享 evaluator。
- confirmation 在数据选择、模型选择、threshold、primary endpoint 和 checkpoint-selection
  rule 冻结后才运行；test 继续封存，本文不提出打开条件。
- 数据集是否入选，只由 schema audit、label-free eligibility count、训练标签分布和预先功效
  判断决定，不能依据 development 模型结果。

---

## 五、对比模型：先保证失败不是“用了一个弱 baseline”

### 5.1 两个主 ordinary Transformer 家族

#### Encoder reranker family

以 [BGE reranker v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
这类 multilingual ranking-pretrained cross-encoder 为起点。建立完全相同 backbone、优化器
预算和 candidate scoring head 的两种输入：

- `E-QC`：query + candidate，不输入历史；
- `E-FULL`：query + 严格先验 history events + candidate，所有 token 进入普通 dense
  self-attention，不加新 primitive。

#### Decoder/LLM reranker family

以 [Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)
为主可执行尺度。它是 ranking-pretrained、支持中文/多语言和长上下文的 decoder reranker。
同样建立：

- `D-QC`：query + candidate；
- `D-FULL`：同一 instruction 下加入结构一致的 history events。

如果资源允许，只在两个主家族结论已冻结后增加 Qwen3-Reranker-4B capacity check；不能把
不断换更大模型变成无上限的反证逃生口。

两个家族的价值在于：一个 encoder cross-encoder 和一个 decoder LLM reranker 共享失败，
比两个相近的 MiniLM 配方更能支持“ordinary full-token modeling 的共同困难”。

### 5.2 强 base 与传统个性化 control

至少保留以下角色，而不是把所有方法堆进一张 leaderboard：

| 角色 | 方法 | 回答的问题 |
|---|---|---|
| 强无历史 base | E-QC、D-QC | 基本 query--candidate 排序是否已经学好 |
| ordinary LLM4Rec | E-FULL、D-FULL | full-token Transformer 是否响应且方向正确 |
| 低复杂度历史 control | pooled/late-fusion history、item recurrence | 简单融合或重复记忆是否已足够 |
| 经典 query-aware personalized control | TEM/ZAM、DIN-style 或仓库中公平实现的最接近方法 | 缺口是否只发生在某一种序列化方式 |
| train-only signal witness | 交叉拟合的 query-conditioned residual/pairwise predictor | 同一信息边界是否存在可恢复方向 |

signal witness 不是 proposed method，也不进入最终方法比较。它可以使用训练内交叉拟合和较强
监督，但必须遵守与被测模型相同的历史、query 和 candidate 信息边界；它的唯一作用是排除
“这个数据根本没有可预测的用户方向”。

### 5.3 objective 与 candidate-wise confound

每个主家族在冻结的小预算内至少比较标准 pointwise 与 pairwise/listwise ranking objective。
如果一种常规 objective 已解决方向问题，结论应是 objective mismatch，而不是新 attention
必要性。

如果两个 candidate-wise full-token 家族均出现方向缺口，再运行一个普通 all-candidate
list-context Transformer control：它允许候选在同一 forward 中相互比较，但不加入任何新
历史 primitive。若该 control 消除问题，failure locus 应收缩为 candidate-independent
scoring interface，而不能泛化成 ordinary Transformer 整体失败。

---

## 六、核心反事实：同一 checkpoint，只替换历史

对每个 `E-FULL` / `D-FULL` checkpoint 和同一 request slate，固定 candidate、query、
token budget 与打分代码，只构造：

- `T=true`：真实且严格先验的用户历史；
- `N=null`：结构合法的空历史；
- `W=wrong`：从同 query/近 query 用户中选择的匹配 donor history；
- `R=repeat-control`：只在诊断中保留/移除 exact repeated candidates；
- `S=shuffle`：只有当后续 claim 涉及 event order 时才成为 binding control，否则只报告。

wrong-history donor 必须在看评价标签前匹配 history 长度、行为类型构成、时间跨度、item
流行度和 query cluster，并排除 target candidate 泄漏。不能使用随便抽一个用户制造容易的
对照。

独立训练的 `E-QC/D-QC` 与 `FULL-null` 有不同作用：

- `FULL-true` vs `FULL-null`：同 checkpoint 的历史响应与方向；
- `FULL-true` vs `FULL-wrong`：响应的用户特异性；
- `FULL-null` vs 独立 `QC`：历史训练是否破坏 base；
- `FULL-true` vs 独立 `QC`：个性化系统的实际 overall payoff。

---

## 七、怎么测“响应”和“方向”

### 7.1 候选相对响应，而不是原始 logit 变化

对请求 `q` 的候选分数向量记为 `s_T(q)` 和 `s_N(q)`。先在每个 slate 内去掉均值：

```text
P(s) = s - mean(s)
delta_TN = P(s_T - s_N)
```

候选相对响应至少报告：

- `||delta_TN||` 相对 base score spread 的规范化幅度；
- candidate pair score-margin 的变化；
- Kendall/rank correlation 变化、pair flip rate 和 top-k membership 变化；
- true/null 与 wrong/null 响应幅度的比较。

同时增加一个**解释性而非 binding** 的 common-mode/differential energy 分解。对请求 `q`
定义：

```text
mu_q = mean_c delta_qc
r_qc = delta_qc - mu_q
rho_common = sum_q C_q * mu_q^2 /
             (sum_q C_q * mu_q^2 + sum_qc r_qc^2)
```

`rho_common` 高，表示模型的大部分 raw-score 响应没有进入候选间 margin；它可以帮助发现
request-level bias 或校准漂移。但 common-mode 本身不改变排序，raw logit 的平移也不具有
跨参数化可识别性，所以它**不是核心 thesis 的必要条件或充分条件**，不能单独用于 Failure
Card。该比率只在同一 checkpoint、同一 scoring parameterization 和相同 score-scale
normalization 内解释，不跨模型直接比较绝对值。

重复 deterministic rescore 给出数值噪声底线；repeat-positive-control 和训练 split 给出
最小有意义响应幅度。只有超过 outcome 前冻结阈值的请求才称为 `active response`。阈值不能
用 confirmation 标签调节。

### 7.2 方向正确性的主指标

方向判断必须看历史**引起的增量**，而不只是最终排名：

1. **增量排序效用**：同一 checkpoint 的
   `NDCG@10(true) - NDCG@10(null)`，并报告 purchase/click 两种标签定义。
2. **signed delta alignment**：对标签有严格优劣的候选对 `(a,b)`，判断
   `(delta_a - delta_b)` 是否把更优候选向上推；按 label gain 差加权，并先在请求内平均，
   防止大 slate 支配结果。
3. **active-response precision**：在 label-free 定义的 active requests 中，历史位移带来
   正向、零向和负向 ranking gain 的比例。
4. **true-over-wrong advantage**：true history 相对 matched wrong history 的增量效用和
   signed alignment 差。
5. **response-to-direction curve**：按训练期冻结的响应幅度分箱，检查响应越大是否真的更
   可能方向正确；若大幅响应只意味着更激烈而非更正确，这是最直观的 failure law。

另报告一个易解释的 secondary metric：在 active 且标签有严格优劣的候选对中，
`delta_a > delta_b` 的 request-averaged pairwise directional accuracy。它与 signed delta
alignment 相互校验，但不能把所有 pair 当独立样本，也不预设 `50%/60%/65%` 这类通用
阈值：曝光构造、ties、每个请求的正例数和 gain weighting 都会改变机会水平与实际价值。
门槛仍应由训练期 null/randomization、最小有用效应和 confirmation 功效共同冻结。

primary endpoint 在冻结时只保留一个排序效用指标和一个增量方向指标，其余作为解释性结果，
避免多指标择优讲故事。

### 7.3 context-demand 指标

在 label-free 构造的 same-query/overlapping-candidate population 上报告：

- 标签要求相反候选方向的自然发生率；
- 对这类 pair，模型能否为两个用户同时排对，不能只报告其中一个请求；
- `true history`、`swapped history`、`null history` 下的 joint pair accuracy；
- 历史交换后预测方向是否随用户上下文合理翻转；
- 对 query cluster 做 cluster bootstrap，不能把同一 query 的大量相关请求当独立样本。

这个实验直接逼近 `query-conditioned candidate-relative`：query 和候选保持不变，只有用户
历史变化，正确排序方向也随用户变化。

### 7.4 不能用“不显著”证明不会

“p 值不显著”不能证明方向不稳定。confirmation 前必须从训练/预留功效分析冻结：

- 最小非平凡响应幅度 `m_response`；
- 最小有实践意义的排序增益 `m_utility`；
- 最小用户特异优势 `m_specificity`；
- paired bootstrap / user bootstrap / query-cluster bootstrap 的使用边界。

核心 failure 需要同时满足：响应幅度的置信下界高于 `m_response`，而方向效用或对齐度的
置信上界低于预设的“可接受转化”门槛，或 true-over-wrong 的上界低于
`m_specificity`。也就是用有功效的 equivalence/non-inferiority 逻辑证明“低于有用程度”，
而不是把宽置信区间解释成失败。

多 seed 用于模型训练不确定性，request/user/query bootstrap 用于样本不确定性；两者不能
互相代替。

---

## 八、分阶段实验与每一步的停止条件

### E-1：现有 checkpoint 的零训练 instrumentation pilot

在 E0 之前，可以复用现有 Amazon/KuaiSearch development checkpoint 或合法 score artifact，
用 true/null 两次推理快速验证 activity、pairwise directional accuracy、common-mode
decomposition、request/user/query cluster bootstrap 和数值噪声实现。若 score artifact 不含
候选级 true/null 分数，只允许在既有 development 边界内补推理，不读取 fresh/test labels。

这一阶段的价值是尽早发现 evaluator 设计、pair dependence、score calibration 和功效问题，
并判断后续正式验证是否值得投入训练成本。它不产生论文结论，不决定绑定数据集，不把旧的
弱/自适应 checkpoint 算作 adequate family，也不能因为 pilot 结果有利就跳过 E0--E2。

### E0：source、collision 与功效审计

只检查字段、时间边界、label-free eligibility、训练标签分布和预期功效，不训练新模型，
不读 dev/test labels。

产物是 dataset admission card。若 KuaiSearch Full 无法形成足够的 strict-nonrepeat 或
same-query/candidate-overlap population，这个数据集不能证明核心 thesis；按预注册顺序进入
KuaiSAR/JDsearch，而不是先跑模型再寻找有利数据。

### E1：建立强 query--candidate base

分别正常调优 E-QC 与 D-QC，并与 source-order、popularity/BM25、已有最强公平 baseline
比较。检查候选 hash、文本覆盖、截断、seed 稳定性和目标指标。

如果两个家族都无法形成合格 query--candidate ranker，停止。弱 base 上观察到的历史方向
混乱没有架构解释价值。

### E2：ordinary full-token LLM4Rec adequacy

在与 QC 家族对称的预算内训练 E-FULL、D-FULL。先检查 overall、repeat positive control、
history token coverage、null-history base 和 true/wrong observability。

如果模型因为历史序列化、超长截断或训练干扰而整体失效，先在既定 ordinary recipe 空间内
修复；仍不合格则停止该 family，不把它计入 shared failure。

### E3：label-free 响应确认

在不看方向标签的 response instrumentation 中比较 true/null/wrong，确认变化是候选相对而非
common-mode，并超过数值噪声和最小活动阈值。

若没有 active candidate-relative response，核心句子的前半句失败。下一步应研究
observability/serialization，而不是声称 response-to-direction conversion failure。

### E4：方向性与用户特异性确认

冻结 active definition 后，由共享 evaluator 计算增量 NDCG、signed delta alignment、
active-response precision 和 true-over-wrong advantage；先看全 strict-nonrepeat population，
再看 natural context-demand population。

若 true history 的响应方向稳定正确且达到 `m_utility/m_specificity`，核心 thesis 被证伪，
不得继续寻找更窄失败切片。如果只在一个预定义 population 失败，结论必须收缩到该 population。

### E5：证明数据里有可恢复方向

运行传统 query-aware personalized controls 与 train-only cross-fitted signal witness。它们应
在相同输入边界、同一候选集上证明至少一部分 history-conditioned direction 可被恢复。

如果所有方法和 witness 都失败，最合理结论是公开日志没有提供稳定监督或标签噪声过大；这会
否定架构论文入口，而不是加强它。

### E6：排除简单解释

按预冻结顺序检查：

- pointwise vs pairwise/listwise standard objective；
- candidate-wise vs ordinary all-candidate context；
- history length/truncation 和 recent/coverage policy；
- exact recurrence、query masking、history identity-only/text-only/action-field ablation；
- source position、item popularity、search entrance/session stage 分层；
- matched wrong-user donor 质量；
- click 与 purchase/graded label 的一致性。

标准 objective control 之后，还可以加入一次冻结的 **directional-delta objective stress
test**：在训练 pair 上显式奖励更优候选的 history-induced delta 更大，并保持模型结构不变。
它是“最近简单答案”的诊断，不是默认训练配方：

- 若同时改善 directional alignment 和 overall ranking，缺口主要属于监督/目标，可形成
  objective contribution，但不支持新架构必要性；
- 若改善 alignment 却不改善或损伤 overall/base，说明局部方向目标与整体排序存在张力，
  可用于 localization，但**不能单凭 tradeoff 证明结构限制**；
- 若它不能改善 alignment，也仍可能来自优化、权重、label bias 或 delta construction，
  不能把一次 objective failure 当作 architecture necessity。

任何一个普通修复消除缺口，都应把 locus 精确收缩到 objective、interface、field loss 或
bias，而不是继续使用宽泛的“Transformer direction failure”。

### E7：独立模型家族与数据复制

在 KuaiSearch 上冻结现象定义和 decision thresholds 后，才在 KuaiSAR（或 outcome 前已替换的
JDsearch）做同构复制。匿名 token 数据只复制功能规律，不复制语言语义机制。

如果现象跨模型但不跨数据，写成 KuaiSearch-specific analysis；如果跨数据但只在一个模型
family 出现，写成该 recipe 的 failure；只有两条轴都复现，才进入 Failure Card。

### E8：Failure Card 或终止

通过时的 Failure Card 至少包含：

- 哪个 label-free population 有多大覆盖和 overall ranking 后果；
- 两个 adequate ordinary full-token families 的共同响应—方向规律；
- true/null/wrong 的反事实证据；
- signal witness 的可恢复性证据；
- 标准 objective、容量、candidate interface、truncation、recurrence 和 position confound
  被如何排除；
- 独立数据复制支持的最窄共同 claim；
- 哪一层/哪一种计算行为产生 failure 的下一步 localization，而不是直接给 architecture 名字。

未通过则登记明确终局：`no response`、`ordinary model succeeds`、`data insufficient`、
`recipe-specific`、`dataset-specific` 或 `underpowered`。这些状态都比再开一个模糊 round 更有
研究价值。

---

## 九、最重要的替代解释与对应控制

| 替代解释 | 若不排除会误讲成什么 | 对应控制 |
|---|---|---|
| 所有候选共同 logit 平移 | 模型在 candidate-relative 地使用历史 | slate centering、pair margin、rank flip |
| personalized model 的 base 很差 | 历史把方向弄错 | 独立 QC base、FULL-null adequacy |
| wrong donor 太容易识别 | true history 有用户特异性 | query/length/action/time/popularity matched donor |
| exact recurrence 垄断收益 | 模型能做跨商品偏好迁移 | strict-nonrepeat primary、repeat positive control |
| 点击位置偏差 | 历史导致错误排序 | source-position 分层、purchase/graded labels、非因果措辞 |
| candidate-wise scorer 不能比较 slate | Transformer 本身不会相对决策 | ordinary all-candidate context control |
| pointwise loss 与 NDCG 不匹配 | 需要新 attention primitive | 标准 pairwise/listwise objective control |
| 历史被截断或序列化损坏 | 模型读到但不会转方向 | token coverage、length policy、recent/semantic selection audit |
| 数据里没有稳定个性化监督 | 架构有盲点 | traditional control + cross-fitted signal witness |
| 只挑反转成功/失败案例 | 普遍 failure law | label-free population、冻结 evaluator、cluster bootstrap |

---

## 十、何种结果才足以支撑后续 architecture 优化

### 10.1 可以推进的结果

后续架构研究需要看到一个类似这样的稳定规律：

> 当 query 与 candidate evidence 固定时，ordinary full-token models 会随 true history
> 大幅改变候选间 margin；但这种 margin update 对 label gain 的 signed alignment 很低，
> 大响应并不提高正确方向概率，且交换为 matched wrong-user history 后表现近似。与此同时，
> 一个使用相同输入的简单 residual/pairwise witness 能预测部分方向。该规律在 encoder 和
> decoder reranker、plaintext 和匿名行为搜索数据上重复出现。

这时优化空间才清楚：不是“让模型更关注历史”，而是寻找为何 history-conditioned update
缺少 candidate-relative direction、为什么 provenance 没有约束更新符号，以及哪个最小计算
改变能提高 response precision，同时不损伤 query--candidate base。

### 10.2 不足以推进的结果

以下结果即使好看，也不能授权架构：

- attention map 显示模型看了历史；
- true 与 null 的 hidden state 或 raw logit 显著不同；
- 某个很窄、事后定义的 slice 上 true-null 为负；
- 一个弱 family 失败、另一个强 family 成功；
- wrong history 更差，但 donor 没有匹配；
- 没有方法能从数据恢复个性化方向；
- 只有 Amazon-C4 随机负例上的结果；
- only significance，没有最小有用幅度与置信上界。

### 10.3 成功后也不要立刻设计模块

现象通过后，下一轮首先应做 failure localization：响应错误主要产生在 candidate-independent
history summary、query--history binding、history--candidate margin update、listwise
competition，还是训练梯度归因。只有 localization 指向稳定的计算缺口，才写 architecture
consequence。否则会重演“先有新算子，再寻找它修什么”的旧循环。

---

## 十一、建议的最小可执行版本

为了避免又变成一个巨大 pipeline，review 通过后的第一批工作只应包含：

1. KuaiSearch Full 的 outcome-free admission/collision/power audit；
2. E-QC/E-FULL 与 D-QC/D-FULL 两个对称家族；
3. true/null/matched-wrong 三个同-checkpoint 反事实；
4. strict-nonrepeat primary population 与 natural context-demand population；
5. candidate-relative response、增量 NDCG、signed delta alignment、true-over-wrong；
6. 一个传统 personalized control 和一个 train-only signal witness；
7. 得到主轨 survivor 后才启动 KuaiSAR/JDsearch 复制。

暂不做新 Transformer primitive，不做 C81，不复活 C01--C80，不把 Amazon-C4 设为 binding
主证据，也不提前打开独立 confirmation/test。

---

## 十二、review 时需要决定的事项

本草案需要人工 review 的不是某个阈值小数，而是以下研究边界：

- 是否接受“KuaiSearch Full 主验证 + KuaiSAR 行为复制，JDsearch 为 schema 替补”的组合；
- 是否接受将 Amazon-C4 降为非绑定语义 stress test；
- 是否接受以 BGE encoder reranker 与 Qwen decoder reranker 作为两个 ordinary family；
- 是否同意把 base degradation 只作为 adequacy/confound，而不是本文主 thesis；
- 是否同意“数据存在可恢复方向”是 Failure Card 的硬前提；
- 是否同意只要 ordinary objective/listwise control 解决问题，就主动收缩或关闭 architecture
  动机；
- 是否同意主张必须有跨 family、跨 dataset 的有功效复制，而不是依赖一次显著性。

review 通过后，下一份文档应只是 E0 的短 protocol：字段映射、split、label-free eligibility、
power/MDE 规则和 dataset admission decision。此时仍不应写 proposed architecture。
