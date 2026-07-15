# 受控历史组合：LLM4Rec 的实证动机与论文逻辑

状态：**CCF-A 级论文 motivation 的中文论证稿**。本文重新组织当前已有的
探索性证据，不新增实验事实，也不把问题提前归因到某个 Transformer 模块。
正式证据综合见
[`35_controlled_history_composition_motivation.md`](35_controlled_history_composition_motivation.md)。

## 论文式摘要

个性化商品搜索要求排序器同时理解两类信息：当前 query 表达的即时需求，以及
用户历史反映的长期或阶段性偏好。近年来常见的 LLM4Rec 做法，是把 query、用户
历史和候选商品共同序列化为 token，交给一个 Transformer/语言模型端到端打分。
这种 full-token 联合建模直观、通用，也具备足够强的表达能力。因此，一个自然的
判断是：只要真实历史相对空历史能够提升排序指标，模型就已经有效利用了历史。

我们的观察表明，这个判断并不充分。真实历史带来的收益混合了两个性质不同的
过程：联合历史训练是否保住了原有的 query–candidate 基础排序能力，以及历史在
这个基础上是否产生了方向正确、候选相对的额外价值。跨 KuaiSearch、JDsearch 和
Amazon-C4 的分析显示，普通联合 Transformer 确实能够读取历史，也能在部分场景
产生正确的历史方向；但它不能稳定地同时控制上述两个过程。在 KuaiSearch 的
非重复请求上，历史响应广泛存在，却很少有效转化为正确排序；在 Amazon-C4 上，
更丰富的历史能够强化正确方向，却同时造成更大的基础能力损失；在 JDsearch 上，
非重复历史效用真实存在，但不足以偿还联合训练造成的基础损失，而重复商品带来的
简单收益又会使 overall 指标保持为正。

因此，当前证据支持的核心问题不是“Transformer 看不懂历史”，也不是“历史一定
有害”，而是：

> **现有 LLM4Rec 联合融合把 query–candidate 基础相关性与历史个性化增量纠缠在
> 同一个打分过程中。模型虽然会对历史产生响应，却缺乏对两项排序义务的稳定
> 控制：一方面保留当前 query 所要求的基础排序能力，另一方面只把历史增量分配给
> 真正受益的候选。因此，正的 `true-history − null-history` 收益可能只是在修复
> 模型自己造成的 base 损失，而 recurrence 又可能在 overall 指标中掩盖非重复
> 偏好迁移的不足。**

这个发现把研究目标从“让 Transformer 更关注历史”推进为一个更明确、可测量、
可证伪的问题：**如何使 Transformer 排序核心对基础相关性与候选相对的历史增量
进行受控组合，并在非重复候选上创造真正的净排序收益。**

## 1. 必要背景：个性化搜索不是单一的信息建模问题

### 1.1 当前意图与历史偏好承担不同职责

在普通推荐中，用户历史通常是主要条件；但在个性化商品搜索中，当前 query 具有
更强的约束性。一个用户过去经常购买摄影器材，当他搜索“儿童防水鞋”时，系统
首先必须识别哪些候选是儿童防水鞋，然后才应在满足当前需求的候选之间使用历史
做个性化区分。

因此，一个合格的个性化搜索排序器至少承担两项义务：

1. **基础相关性义务**：保留 query 与 candidate 之间的语义和任务相关性；
2. **历史增量义务**：在基础相关性之上，利用与当前 query 有关的历史偏好，调整
   候选之间的相对顺序。

可以用一个仅用于分析、并不限定具体架构的概念式来表示：

```text
最终排序 = query–candidate 基础排序 + query-conditioned 历史增量
```

这里的关键不是一定采用加法结构，而是两类信息具有不同的因果和评价角色。历史
应该提供“增量”，而不应通过破坏当前 query 的基本含义来换取表面上的个性化。

### 1.2 什么是本文讨论的 full-token LLM4Rec

本文所说的 full-token LLM4Rec，是指 query 文本、历史事件中的文本或行为字段、
候选商品文本共同进入 Transformer/语言模型，由 Transformer 本身作为端到端排序
核心输出候选分数。例如：

```text
[query tokens] + [history-event tokens] + [candidate tokens]
                              ↓
                    Transformer / LM ranker
                              ↓
                         ranking score
```

它不是“Full 数据集”，也不要求重新预训练一个大语言模型。当前实验中的 BGE
encoder reranker 和 Qwen decoder ranker 都属于这个建模范畴：我们微调的是排序
任务，而不是从零训练基础语言模型。

这种方案之所以常见且合理，是因为自注意力原则上可以让 query、历史和候选充分
交互。也正因为如此，本文并不把“Transformer 完全读不到历史”作为前提。真正的
问题是：**有表达能力，不等于训练后会形成可控、排序有效的组合方式。**

## 2. 现有评价逻辑中的隐藏缺口

LLM4Rec 实验常用下面的比较来说明历史有效：

```text
同一个模型输入真实历史，比输入空历史取得更好的 NDCG / MRR
```

这个比较本身没有错，但它只回答“历史能否帮助这个已经接受过联合历史训练的
模型”，不能回答“最终个性化模型是否超过了一个正常训练的无历史排序器”。

为了区分这两件事，我们使用四个互补角色：

| 角色 | 定义 | 诊断目的 |
|---|---|---|
| `QC` | 独立训练的 query–candidate 排序器，训练和推理都不使用历史 | 给出加入历史前应当保有的基础排序能力 |
| `FULL-true` | 联合历史训练模型，推理时输入真实用户历史 | 给出个性化系统的最终表现 |
| `FULL-null` | 与 `FULL-true` 完全相同的权重，推理时移除历史 | 隔离联合历史训练后剩余的基础能力 |
| `FULL-wrong` | 同一权重，输入其他用户或匹配控制历史 | 检查收益是否来自正确的历史来源 |

这四个角色揭示出一个精确的指标账目：

```text
FULL-true − QC
= (FULL-null − QC) + (FULL-true − FULL-null)
= Base Retention      + History Utility
```

其中：

- `Base Retention` 衡量联合历史训练是否保住了 query–candidate 基础能力；
- `History Utility` 衡量真实历史相对同一 FULL 模型的空历史状态带来多少效用；
- `FULL-true − QC` 才是个性化相对正常基础排序器创造的净价值。

如果 `History Utility` 为正，但 `Base Retention` 更负，那么“历史有效”的结论只
说明历史帮助 FULL 模型恢复了部分能力，并不说明它创造了净个性化收益。这不是
指标解释上的细枝末节，而是会改变论文核心结论的评价缺口。

## 3. 我们如何观察这个问题

### 3.1 从“分数是否变化”推进到“排序方向是否正确”

对于同一个请求和候选 (c)，定义历史引起的分数变化：

```text
Δ_h(c) = score_FULL-true(c) − score_FULL-null(c)
```

如果所有候选都得到相同的增量，模型虽然对历史有强烈响应，排序却不会发生任何
变化。因此，我们进一步减去请求内候选的平均增量，只观察 candidate-relative
部分。对于一个正候选 (c^+) 和负候选 (c^-)，历史方向正确意味着：

```text
Δ_h(c+) > Δ_h(c−)
```

这使“利用历史”被拆成三个逐级增强的问题：

1. 历史是否让模型产生响应；
2. 响应是否改变候选之间的相对关系；
3. 相对变化是否沿着能够改善真实排序的方向发生。

此外，我们使用 fixed-response direction intervention：保持一个请求内已有历史
响应的幅度分布不变，只改变这些增量在候选间的分配，并比较随机分配、模型实际
分配和标签对齐分配。它衡量的不是模型能否制造更大的历史响应，而是**已有响应中
有多少被有效转化为排序收益**。标签对齐版本仅是 dev 诊断上界，不是可部署方法，
也不证明该方向能由训练数据直接学得。

### 3.2 区分 recurrence 与真正的偏好迁移

如果目标商品已经出现在用户历史中，模型只需学会“历史中出现过的候选应当上升”，
就可能获得很大收益。我们把这类请求称为 `repeat`。它是重要的真实流量，也是
检验模型是否具备基本历史学习能力的正控制。

但论文若要主张模型理解了用户偏好，还必须观察 `strict-nonrepeat`：正确候选没有
直接出现在历史里，模型需要结合当前 query，将过去的行为迁移到一个新候选上。
这一分面更接近 query-conditioned personalization，而非候选记忆或 exact match。

### 3.3 选择互补数据集，而不是要求所有数据集呈现相同故障

我们没有让任意一个数据集“一票否决”整个研究问题，而是让不同信息条件的数据集
承担不同证据角色：

| 数据集 | 信息特点 | 在论证中的角色 |
|---|---|---|
| KuaiSearch Lite 与 Full-source scout | 自然语言搜索 query、真实行为历史、repeat 与 nonrepeat 共存 | 观察自然搜索中的方向分配和 recurrence 掩盖 |
| JDsearch | 独立商品搜索源，语义字段匿名但排序功能完整 | 验证账目分解和功能性规律是否跨数据源存在 |
| Amazon-C4 | 商品文本和历史语义丰富，query 构造相对容易 | 作为正向边界，检验 Transformer 在信息充分时能否获得正确方向 |

KuaiSearch Lite 和 Full-source scout 是同一来源的不同人口样本，不能冒充两个独立
数据集；Amazon-C4 的 query 具有构造性，也不能冒充自然搜索复现。恰当的论文逻辑
不是抹平这些差异，而是利用它们解释：**同一个受控组合问题会在不同信息条件下
暴露出不同侧面。**

## 4. 核心观察：模型会读历史，但不会稳定地组合历史

### 4.1 观察一：历史通路是活跃的，问题不是“模型完全忽略历史”

在主要实验面上，真实历史相对空历史会改变大量请求和候选对的分数。KuaiSearch
和 JDsearch 的 repeat 请求还提供了更强的正控制：同一模型能够识别历史中的直接
重复候选，并取得明显的排序收益。

这排除了一个最简单的解释——模型只是没有读取到历史 token、历史接口完全失效，
或者训练没有让历史进入打分函数。模型具备历史响应，问题发生在响应如何被组织和
转化，而不是历史通路是否存在。

### 4.2 观察二：在自然搜索的非重复请求上，广泛响应不等于有效方向

KuaiSearch Lite 上，无论使用 encoder 型 BGE reranker 还是 decoder 型 Qwen
ranker，无论采用 pairwise 还是 pointwise 目标，历史都能广泛改变候选间分数；但
在 strict-nonrepeat 请求上，候选相对方向接近随机或在人群间不稳定，模型只转化了
很小一部分固定响应幅度下本可获得的方向收益。

KuaiSearch Full-source scout 进一步恢复了历史事件中的先前行为 query。结果并未
修复 strict-nonrepeat 的方向转化问题。因此，当前现象不能简单归咎于 Lite 数据
缺少历史 query，或历史文本过于贫乏。

这一观察说明：**Transformer 对历史敏感，与 Transformer 能把历史变成当前 query
下正确的候选相对偏移，是两种不同能力。**

### 4.3 观察三：正确的历史方向也不保证最终获得净收益

Amazon-C4 给出了一个对原始“方向盲”假设非常重要的反例。普通 Transformer 在
该数据集上能够产生显著正确的 candidate-relative 历史方向；增加历史预算后，
方向质量和 `FULL-true − FULL-null` 都继续增强。这证明 Transformer 并非在原理上
不能学习正确的历史方向。

但与此同时，更长历史使 `FULL-null − QC` 的基础损失增长得更快。以当前探索性
长历史实验为例：

```text
Base Retention  = −0.1343
History Utility = +0.1132
Net Value       = −0.0211
```

也就是说，历史确实被正确利用了，却仍不足以偿还联合历史训练造成的基础能力损失。
如果只报告 true-over-null，这个实验会被描述成“历史非常有效”；如果比较最终模型
与 QC，结论则是“个性化没有创造净价值”。

Amazon-C4 因而不是应该忽略的负面数据集。它否定了过强的普遍方向失效故事，同时
把问题收紧为更稳健的研究命题：**历史方向提取成功与历史安全组合成功不是同一件
事。**

### 4.4 观察四：recurrence 可以使 overall 指标看起来成功

JDsearch 展示了这类误判如何发生。当前探索性 dev 结果的账目如下：

| 请求面 | Base Retention | History Utility | FULL-true 相对 QC 的净值 |
|---|---:|---:|---:|
| strict-nonrepeat | −0.0407 | +0.0289 | −0.0118 |
| repeat | −0.0512 | +0.2401 | +0.1888 |
| overall | −0.0432 | +0.0796 | +0.0364 |

在 nonrepeat 上，历史效用是真实的，但小于基础损失；在 repeat 上，直接复现带来
的巨大收益足以覆盖同量级的基础损失；最终 overall 仍然为正。若只看 overall，
可以写成“个性化模型优于 QC”；分面后才能看到，成功主要集中在容易的 recurrence，
偏好向新候选的迁移并没有获得同等可靠的净收益。

这里并不是说 repeat 流量不重要，而是说：**repeat 上的成功不能作为模型已经解决
非重复个性化排序的证据。**

### 4.5 跨数据集结果不是相互矛盾，而是共同定位了一个组合问题

三个数据集并没有呈现完全相同的表面症状：

- KuaiSearch：基础排序仍可用，但非重复历史响应的方向分配和转化较弱；
- Amazon-C4：历史方向高度正确，但更强历史依赖造成更大的 base erosion；
- JDsearch：处于两者之间，非重复历史有真实效用，却不足以抵消基础损失，且
  recurrence 掩盖了这一点。

如果研究命题是“所有 Transformer 都方向盲”，这些结果确实互相冲突；如果研究
对象是“基础相关性与历史增量缺乏受控组合”，它们恰好构成互补证据。普通联合
Transformer 可以在任一轴上成功，却没有稳定地把两个轴共同推向更优。

## 5. Baseline 代表性与当前证据强度

当前核心现象不是只从一个 Qwen checkpoint 得出的。KuaiSearch Lite 的方向分配
症状已经跨越：

- BGE encoder 与 Qwen decoder 两类 Transformer 排序核心；
- pairwise 与 pointwise 两种训练目标；
- Lite 与独立 Full-source 人口 scout；
- 仅商品历史与恢复历史 query 的两种信息接口。

此外，我们还实现并运行了推荐原生的 HSTU、经典 SASRec，以及前沿论文路线的
LLM-SRec 代表性适配。它们在 Lite 上也表现出“历史响应广泛、strict-nonrepeat
方向接近机会水平”的支持性症状，使问题不再只是某一种文本序列化的偶发现象。

但这些结果目前只作为**支持性证据**，还不能作为跨架构的 binding confirmation：
HSTU 和 SASRec 的 QC 能力尚未超过 BM25，LLM-SRec 仍是有限训练预算下的探索性
适配，且尚未完成独立数据集上的同等确认。因此，当前最准确的证据等级是：

> 普通 full-token Transformer 上的跨数据集问题动机已经形成；跨推荐架构的结果
> 与该动机一致，但模型充分性和冻结确认仍需补齐。

这一区分很重要。CCF-A 级 motivation 需要代表性证据，但不能为了故事完整而把
弱 baseline 包装成强证据。

## 6. 为什么这个结果重要

### 6.1 它揭示了主流评价中的潜在假阳性

`FULL-true > FULL-null` 只能证明历史帮助了同一个联合训练模型，不能证明最终
个性化系统优于一个正常的 query–candidate 排序器。overall 提升也可能主要来自
repeat 流量。若不同时报告 QC、null、repeat 和 strict-nonrepeat，论文可能把“恢复
自身损失”误写成“创造个性化价值”，把“候选复现”误写成“偏好迁移”。

因此，本工作首先贡献的是一个更严格的问题定义和评价视角：历史响应、历史效用、
基础能力保留和最终净收益必须被分别测量。

### 6.2 它对应真实系统中的相关性安全问题

商品搜索的首要约束是当前 query。一个系统即使更懂用户，也不能因为历史依赖而
降低基本相关性。基础能力损失并不只是某个离线指标的小波动，它意味着历史可能
在信息不足、与当前需求无关或用户意图发生切换时把候选推向错误方向。

因此，“历史越丰富越好”不是可靠的设计原则。更合理的目标是：模型应当知道何时、
对哪个候选、以多大幅度使用历史，并在历史无效时保住当前 query 的排序能力。

### 6.3 它把宽泛的“优化 Transformer”变成可验证的研究对象

“让 Transformer 更好地建模历史”过于宽泛，很容易退化为增加 attention、memory、
gate 或 loss 的模块堆叠。受控历史组合则给出了两个独立、可量化的义务：

```text
义务 A：Base Retention
义务 B：Directional History Utility
```

新方法必须改善二者的联合前沿，而不能只提高历史响应幅度、attention 可视化或
true-over-null 收益。这使后续架构设计拥有明确的失败对象、对照实验和证伪条件，
也避免再次进入“先想模块、再寻找故事”的循环。

## 7. 从观察中得到的 CCF-A 级 motivation

### 7.1 一句话问题定义

> **LLM4Rec 的核心缺口不是无法读取历史，而是无法稳定地将历史作为一个
> query-conditioned、candidate-relative 的增量，安全地组合到已有的
> query–candidate 排序能力之上。**

### 7.2 完整论文论证链

```text
个性化商品搜索必须同时满足当前意图与历史偏好
                         ↓
现有 full-token 联合 Transformer 假定一次联合打分即可自然完成融合
                         ↓
传统 true-vs-null / overall 指标无法区分 base 损失、历史增量与 recurrence
                         ↓
跨数据集反事实分析发现：模型会读历史，但两个组成部分不能稳定共同改善
                         ↓
KuaiSearch 暴露方向转化不足，Amazon 暴露 base erosion，JD 暴露 overall 掩盖
                         ↓
研究机会：让 Transformer 对基础相关性与候选相对历史增量进行受控组合
```

这条逻辑比“Transformer 不会利用历史”更强，原因在于它经受住了正面反例：即使
Amazon 已证明方向可以学对，组合问题仍然存在；即使 JD overall 为正，分解后仍可
发现非重复净价值不足。一个好的 motivation 不应依赖所有实验都失败，而应能解释
模型何时成功、何时失败，以及不同成功与失败为何属于同一研究矛盾。

## 8. 这个结果启发我们做什么

当前证据启发的不是立刻增加一个具体模块，而是为后续 LLM4Rec Transformer
提出五项设计要求。

### 8.1 保住基础排序能力

历史建模不能通过牺牲 query–candidate competence 获得收益。未来方法需要让历史
为空、无关或发生意图切换时，模型仍能保有一个充分训练的基础排序器的能力。
这不预设一定采用双分支或残差结构，但要求 `FULL-null − QC` 成为正式约束，而
不是事后补充指标。

### 8.2 让历史作用成为候选相对、query-conditioned 的增量

历史不应只产生 request-level 的共同偏移，也不应脱离当前 query 提供泛化用户
兴趣。真正有价值的更新必须区分同一 slate 中的候选，并优先改变当前 query 下
确实受益的候选相对位置。

### 8.3 控制历史的选择性，而不只是增强响应强度

Amazon 的结果说明，更强历史方向和更大 true-null 收益仍可能伴随更差的最终
排序。因此，方法目标不应是“让模型更依赖历史”，而应是提高每单位历史响应转化
为净排序收益的效率，并抑制无关、错误来源或与当前意图冲突的历史作用。

### 8.4 把 recurrence 与 preference transfer 分开优化和评价

一个方法可以保留 repeat 上的实用收益，但必须单独证明它能在 strict-nonrepeat
上将偏好迁移到历史中未出现的新候选。否则，改进仍可能只是更强的候选记忆。

### 8.5 优化联合前沿，而不是单个局部指标

后续方法至少需要联合报告：

1. `FULL-null − QC`：基础能力是否保留；
2. `FULL-true − FULL-null`：历史是否产生额外效用；
3. `FULL-true − QC`：最终是否创造净价值；
4. strict-nonrepeat 上候选相对方向是否正确并转化为实际排名提升；
5. repeat 与 nonrepeat 的分面结果，避免 recurrence 掩盖。

只有同时改善这些量，才能说明模型解决了受控组合问题，而不只是把损失从一个分面
转移到另一个分面。

## 9. 从 motivation 到架构贡献之间还缺什么

当前结果足以支持一个值得继续投入的论文问题，但尚不足以断言“必须发明新
Transformer 架构”。在进入架构设计前，仍应完成以下判别：

1. **简单训练修复**：history dropout、base anchoring、loss reweighting 或接口
   控制能否同时恢复 base 与历史效用；
2. **可学习性验证**：strict-nonrepeat 的正确方向能否从训练信息中恢复，而不是
   只存在于使用 dev 标签的诊断 oracle 中；
3. **强架构复现**：在充分训练的 HSTU 或前沿论文架构上重复同一账目，而不是只看
   历史响应；
4. **冻结确认**：固定命题、指标和控制后，在独立确认人口上验证，而不是继续根据
   dev 结果移动定义。

如果标准训练修复即可关闭差距，论文贡献应诚实地定位为 objective 或 training
interface；如果这些控制仍无法同时改善两个义务，才形成进入 Transformer 内部结构
设计的合法入口。届时架构工作的目标也不是泛化地“增强历史建模”，而是：

> **设计一个以 Transformer/LM 为端到端排序核心、能够协调 base relevance 与
> candidate-relative history update 的 LLM4Rec 模型。**

这条路线保留了 LLM4Rec 和 Transformer 优化的研究初心，同时让架构创新由已经
观测到的失败推出，而不是由模块新颖性倒推 motivation。

## 10. 当前可以写什么，不能写什么

### 已被探索性证据支持

- 普通 Transformer 会读取历史并改变候选排序，不是完全 inactive；
- 历史响应、正确方向、历史效用与最终净收益是不同概念；
- 联合历史训练可能侵蚀 query–candidate 基础排序能力；
- recurrence 会使 overall 指标高估非重复个性化能力；
- KuaiSearch、JDsearch 和 Amazon-C4 从不同侧面共同支持受控组合问题；
- “普遍方向盲”已经被 Amazon-C4 否定，应使用更准确的组合命题。

### 尚未被证明

- 不能声称所有 LLM4Rec 模型、数据集或请求都会失败；
- 不能声称 Transformer 在原理上无法解决这一问题；
- 不能把故障直接归因于 attention、表示空间或某个内部 primitive；
- 不能声称只有新架构能够修复，简单训练和接口控制尚未全部排除；
- 不能把 HSTU、SASRec、LLM-SRec 的当前 Lite 结果写成强 baseline 的冻结确认；
- 当前仍是 exploratory dev 证据，不是最终 test 上的论文主表结论。

因此，现阶段最严谨的论文措辞是“我们观察到一个跨数据集、跨普通 Transformer
实现重复出现的受控组合缺口”，而不是“我们已经证明 Transformer 架构存在不可
修复的缺陷”。

## 11. 可直接用于论文 Introduction 的五段逻辑

**第一段：任务价值。** 个性化商品搜索需要同时建模当前 query 的即时意图和用户
历史中的个体偏好。前者决定候选是否满足当前需求，后者负责在相关候选之间进行
个性化区分。二者缺一不可，因此历史利用不能以破坏基础 query–candidate 相关性
为代价。

**第二段：现有范式及其隐含假设。** 现有 LLM4Rec 方法通常将 query、历史和候选
共同输入 Transformer，并用真实历史相对空历史或整体排序提升证明个性化有效。
这种评价隐含地假设：联合训练已经保留基础排序能力，true-over-null 的差值就是
历史创造的净价值，overall 提升也代表了偏好向新候选的迁移。

**第三段：关键观察。** 我们通过 QC、FULL-true、FULL-null 和 FULL-wrong 的匹配
反事实比较，将最终收益分解为 base retention 与 history utility，并进一步区分
repeat 与 strict-nonrepeat、响应幅度与候选相对方向。跨三个互补数据集的结果显示，
普通 Transformer 能够读取历史，但基础能力和历史增量不能稳定共同改善：自然搜索
中存在低效的非重复方向分配，语义丰富数据中存在随历史增强而加剧的 base erosion，
独立搜索数据中 recurrence 又会掩盖非重复净损失。

**第四段：为什么现有结论不够。** 因而，正的 true-over-null 收益不必然意味着
有效个性化；它可能只是在偿还联合训练造成的基础损失。正的 overall 收益也不必然
意味着模型能够迁移用户偏好；它可能主要来自历史中的直接候选复现。历史响应本身
不是最终能力，响应是否保留基础相关性、是否具有候选相对的正确方向、是否产生净
排序价值，才是关键。

**第五段：研究机会。** 这些发现提出了“受控历史组合”问题：下一代 LLM4Rec
排序器应当在保留 query–candidate competence 的同时，将历史表示为选择性的、
query-conditioned、candidate-relative 更新，并在非重复场景中提高历史响应到净
排序收益的转化效率。这为后续 Transformer objective、interface 或 architecture
设计提供了统一、可验证的目标。

## 12. 证据入口

- 正式英文综合：
  [`35_controlled_history_composition_motivation.md`](35_controlled_history_composition_motivation.md)
- C3 探索性 motivation 审计：
  [`../reports/pps_c3_motivation.json`](../reports/pps_c3_motivation.json)
- KuaiSearch 代表架构支持性验证：
  [`../reports/history_response_gap_lite_representative_architecture_decision.json`](../reports/history_response_gap_lite_representative_architecture_decision.json)
- JDsearch 的分面、账目和格式控制：
  [`../reports/history_response_gap_jdsearch_motivation_decision.json`](../reports/history_response_gap_jdsearch_motivation_decision.json)
- Amazon-C4 的正向边界与历史预算权衡：
  [`../reports/history_response_gap_amazon_c4_motivation_contrast_decision.json`](../reports/history_response_gap_amazon_c4_motivation_contrast_decision.json)
