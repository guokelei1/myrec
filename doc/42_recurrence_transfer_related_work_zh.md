# 商品重排中 recurrence–transfer 失衡的相关工作

状态：2026-07-16 文献调研记录。本文服务于 Motivation V1.1 的学术定位，记录与
`recurrence–transfer` 观察最相关的原始论文、已有优化路径和当前未被覆盖的证据空缺。
本文只整理相关工作，不授权新的方法扩展或训练范围。

当前实验结论和数值以
[`41_motivation_v11_current_conclusion_zh.md`](41_motivation_v11_current_conclusion_zh.md)
为准；冻结 V1 证据见
[`40_transformer_recurrence_transfer_motivation_v1_zh.md`](40_transformer_recurrence_transfer_motivation_v1_zh.md)。
实验、文献与下一步假设的完整推理链见
[`43_llm_rerank_recurrence_transfer_research_logic_zh.md`](43_llm_rerank_recurrence_transfer_research_logic_zh.md)。

## 1. 术语与调研结论

本文使用的 `transfer` 是一个任务内概念：

> 模型从用户历史中的商品、查询和交互中抽取偏好，并把这种偏好用于排序历史中没有
> 出现的当前候选商品，特别是 `target-nonrepeat/no-candidate-overlap` 请求。

它不同于文献中常见的跨数据集、跨平台或跨领域 transfer learning。后者只作为表示学习
的相邻证据，不能直接证明本项目的 history-to-unseen-candidate transfer。

本轮调研得到的最重要结论是：

1. 序列推荐文献已经直接发现 repetition shortcut、repeat/explore 难度失衡，以及平均
   指标掩盖 explore 失败的问题；
2. 个性化商品搜索和重排文献已经分别提出选择性个性化、语义历史建模、对比增强、
   query-aware 历史筛选、外部关系和偏好记忆等 transfer-oriented 方法；
3. 但是，截至本轮检索，尚未找到一篇工作在自然查询的商品候选重排中，同时使用
   same-checkpoint `full-history − null-history` 干预，并显式比较 exact recurrence 与
   candidate-disjoint transfer；
4. 因此，已有工作支持“这个问题真实且可优化”，但没有替代本项目对收益来源的诊断。

调研覆盖三层：个性化商品搜索/重排、repeat/explore 问题发现、未见商品与可迁移历史
表示。以下判断优先依据论文原文；“未找到”只表示本轮检索结果，不构成绝对首创声明。

## 2. 最直接的问题发现：repeat shortcut

Li et al. 的 SIGIR 2023 论文
[Repetition and Exploration in Sequential Recommendation](https://staff.fnwi.uva.nl/m.derijke/wp-content/papercite-data/pdf/li-2023-repetition.pdf)
是目前与本项目现象最接近的直接先例。论文在 Diginetica 和 Yoochoose 上分析
GRU4Rec、Caser、SRGNN、BERT4Rec、SASRec 和 RepeatNet，得到以下结论：

- 模型在 repeat-next 用户上的准确率明显高于 explore-next 用户；
- 更高的整体准确率可能来自牺牲 explore 用户，平均指标和总体显著性检验会隐藏这一点；
- 训练中的 repeat-next 样本形成 accuracy shortcut，使模型对 explore-next 用户也更倾向
  推荐历史商品；
- 一些模型在只含 explore target 的训练人口上仍会输出 repeat item；
- 输入层和预测层共享 item embedding 会加剧 repetitive bias；独立预测 embedding 能缓解
  但不能消除该问题；
- 论文在纯探索场景提出 3R（remove repeat items from results）后处理并获得显著改善。

该论文直接证明了 recurrence/exploration 难度失衡和 shortcut，但不是 query-conditioned
product reranking。3R 只适用于已知的纯探索场景，不能直接用于 repeat 与 transfer 混合的
自然商品搜索流量。

另一个相邻方向是显式增强 recurrence。RepeatNet 的
[Repeat Aware Neural Recommendation Machine](https://ojs.aaai.org/index.php/AAAI/article/view/4408)
将预测拆成 repeat/explore 两种模式；个性化搜索中的
[Enhancing Re-finding Behavior with External Memories](https://www.zhouyujia.cn/attaches/WSDM2020.pdf)
则专门建模重新查找历史文档。这些工作说明 recurrence/re-finding 是一个已知且有独立
结构的子任务，但它们的目标是利用 recurrence，而不是诊断它是否掩盖 transfer。

## 3. 个性化商品搜索与重排的直接邻居

| 工作 | 场景与主要方法 | 与本项目 transfer 的关系 | 是否显式拆分 repeat/transfer |
|---|---|---|---|
| [ZAM, CIKM 2019](https://assets.amazon.science/37/52/3425ce394654af4687c7feba6b0f/a-zero-attention-model-for-personalized-product-search.pdf) | 商业商品搜索日志；zero attention 允许模型完全关闭历史 | 发现个性化收益依赖 query–history 关系，无关历史可能带来噪声 | 否 |
| [TEM, SIGIR 2020](https://arxiv.org/abs/2005.08936) | Transformer 联合编码 query 与购买历史，动态调节个性化强度 | 本项目直接 baseline；建立 history interaction，但只报告总体搜索指标 | 否 |
| [RTM, SIGIR 2021](https://arxiv.org/abs/2004.09424) | query、用户历史评论和候选商品评论做细粒度 Transformer 匹配 | 用文本属性把历史偏好匹配到不同商品，是直接的语义 transfer 路线 | 否 |
| [CAMI, WWW 2022](https://dou.playbigdata.com/publication/2022_WWW_Multi_Interest_Product_Search.pdf) | 按类别拆分多兴趣，当前 query 和候选决定兴趣聚合 | 减少单一用户向量混合不相关类别偏好的问题 | 否 |
| [CoPPS, KDD 2023](https://doi.org/10.1145/3580305.3599287) | BERT 历史编码器；序列、规则和知识图谱增强后的对比预训练 | 主动用同类或 KG 相似的序列外商品替换历史商品，最接近 transfer-targeted 训练 | 否；仅总体 Amazon 指标 |
| [BATA, TOIS 2025](https://doi.org/10.1145/3726864) | 将 item–item 与 query–query 外部关系作为 Transformer attention bias，并加入辅助解码任务 | 在 JDsearch 上用当前 query 定位相关历史，直接改善历史到相关候选的推断 | 否 |
| [HMPPS, 2025](https://arxiv.org/abs/2509.18682) | top-K 商品 MLLM reranker；多模态内容、query-aware 历史筛选与 hard negatives | 明确针对低频/unseen 样本、无关历史和内容噪声 | 否 |
| [MemRerank, 2026 preprint](https://arxiv.org/abs/2603.29247) | 1-in-5 商品重排；偏好记忆抽取器由下游 rerank reward 训练 | 直接发现 raw history 受噪声、长度和 relevance mismatch 限制 | 否 |

### 3.1 ZAM：先判断历史是否应该生效

ZAM 在商业商品搜索日志上发现，个性化并不总能改善搜索：其潜力依赖 query 特征以及
query 与购买历史的联合关系；用户第一次搜索某个类别时可能没有任何相关历史。普通
attention 被迫至少选择一个历史商品，ZAM 通过 zero vector 允许历史权重接近零。

这与本项目的启示一致：历史响应率高不等于历史产生了有用 transfer。模型需要同时学习
“使用哪段历史”和“什么时候不使用历史”。但 ZAM 没有 target-aware overlap surface，
不能判断其总体收益来自 recurrence 还是 transfer。

### 3.2 CoPPS：最接近 transfer-targeted 的训练方法

CoPPS 针对真实搜索日志稀疏、含噪声、用户序列表示不稳定的问题，先对 BERT 历史编码器
做自监督对比预训练，再用于商品排序。其正样本视图包括：

- 随机 mask 历史 query 或商品；
- 重排 query–product 行为对；
- 将历史商品替换为同类别、但不在原序列中的商品；
- 用知识图谱 embedding 找到相似商品进行序列外替换；
- 删除与当前 query 最不相关的历史 query–product 对。

其中“把历史商品替换成语义/类别相似但 ID 不同的商品，并保持用户表示一致”直接抑制
exact-ID memorization，理论上最有可能提高本项目定义的 transfer。其局限是实验使用
Amazon 5-core 和自动构造 query；论文没有报告用户历史与 target/candidate 的 overlap
审计，所以只能证明总体 personalized product search 改善，不能确认改善落在 transfer。

### 3.3 BATA：最接近 JDsearch 的现有证据

BATA 认为历史通常很短且含随机购买，单靠序列内 attention 容易得到不稳定用户画像。
它把品牌、类别等 item–item 关系和 query 语义相似度作为 attention bias，并用历史商品
序列重建和全局商品相似度预测提供辅助监督。

在其 JDsearch 设置中，CoPPS 的 `MRR/NDCG/Precision` 为
`0.225/0.220/0.108`，BATA-v 为 `0.230/0.220/0.120`。作者的消融结论是，JDsearch
提升主要来自 query dependency：真实 query 能帮助模型找出与当前意图相关的历史；由于
JDsearch 的商品关系较稀疏，完整 KG 初始化反而不如不初始化的 BATA-v。该结果说明
query-aware history selection 值得验证，但 NDCG 没有超过 CoPPS，而且论文没有
repeat/no-overlap 拆分，因此不能作为“transfer 已解决”的证据。

### 3.4 HMPPS 与 MemRerank：LLM rerank 中的历史压缩和筛选

HMPPS 把已有 PPS 模型的 top-10 结果交给 MLLM pointwise reranker。它先用 query 相关
视角压缩商品描述，再用第一阶段模型筛选与 query/candidate 相关的历史商品，并使用初排
结果中的 hard negatives。论文明确指出 ID 模型难以学习低频和 unseen 样本，也指出无关
历史会误导 MLLM；但实验仍使用 Amazon 自动构造 query，没有 overlap-conditioned 指标。

MemRerank 更直接地比较 no-memory、raw-history 和 preference-memory：作者报告 raw
history 因噪声、长度和 relevance mismatch 往往无效，而由下游 1-in-5 rerank 正确性
奖励训练的偏好记忆最高带来 `+10.61` 个绝对准确率点。它是 2026 年预印本，且没有
recurrence/transfer surface，因此只能作为方法邻居，不能作为已确认结论。

## 4. 一般 rerank 与可迁移表示的相邻方法

### 4.1 历史序列的鲁棒对比学习

[COCA, CIKM 2021](https://arxiv.org/abs/2108.10510) 面向 context-aware document
ranking，对历史 query/document 序列使用 term mask、query/document deletion 和行为
重排，再通过对比学习使 BERT 历史表示对这些变化保持稳定。它证明了该方法可以改善
文档重排，并报告 term mask 单独带来约 2.5% MAP 提升。它不在商品搜索中，也没有
repeat surface，但提供了一个较便宜的训练控制。

[UR4Rec, COLING 2025](https://aclanthology.org/2025.coling-main.45/) 使用候选商品作为
query，从 LLM 生成的用户偏好和商品知识中检索与当前候选相关的信息；随后使用
InfoNCE 和 preference–item matching 对齐语义表示与推荐模型表示。它是真正的候选集
rerank，但数据是 MovieLens、Amazon-book 和 Steam，不包含自然商品搜索 query。

[PEAR, WWW 2022](https://arxiv.org/abs/2203.12267) 和
[MIR, 2022](https://arxiv.org/abs/2204.09370) 则直接建模初排候选集合与历史点击列表之间
的 set-to-list interaction。它们证明“候选条件化地读取历史”优于简单 history encoder，
但同样没有区分 exact repeat 和 unseen-candidate transfer。

### 4.2 文本语义与跨场景 transfer

[UniSRec, KDD 2022](https://arxiv.org/abs/2206.05941) 指出显式 item ID 序列难以迁移到
新场景，使用商品描述、whitening/MoE adaptor 和多域对比预训练学习通用序列表示。
[RecFormer, KDD 2023](https://arxiv.org/abs/2305.13731) 将商品属性展平为文本句子，结合
语言理解与推荐预训练，在低资源和 cold-start 商品上取得改善。

这些论文的 transfer 主要指跨域或 cold-start，不等于本项目的 same-dataset history
transfer；但它们共同支持一个机制判断：要改善未见候选，模型需要依赖可组合的商品内容
和属性语义，而不能只依赖 item ID 或历史中的精确表面匹配。

## 5. 本项目相对已有工作的证据空缺

当前文献已经覆盖“repeat shortcut 存在”和“历史 transfer 可以被优化”，但本轮没有
发现同时满足以下条件的工作：

1. 自然用户 query；
2. 初排候选集上的商品 rerank；
3. 同一 checkpoint、同一请求和候选集上的 full-history/null-history 干预；
4. exact target-repeat 与 target-nonrepeat/no-candidate-overlap 互斥 surface；
5. 分别报告两类 surface 的每请求历史增量及 `repeat − no-overlap` contrast；
6. 在 KuaiSearch/JDsearch 这类真实搜索人口上复现。

因此，已有论文报告“加入历史后总体 NDCG/MRR 提升”，并不能确定提升来自 recurrence、
transfer，还是二者的流量加权。这个 attribution gap 是本项目最清楚的学术位置。

结合 V1.1，当前应使用的有界表述是：

> Target recurrence can strongly amplify a ranker's response to history. Transfer
> to candidate-disjoint targets is possible—as demonstrated on JDsearch—but its
> per-request effect is much smaller and is less stable across datasets and models.

不能声称 transfer 不存在，也不能声称所有 Transformer 都有固定比例的失衡。KuaiSearch
冻结 V1 和 InstructRec 多 seed 提供弱/不可靠 no-overlap 证据；TEM 扩展没有稳定复现；
JDsearch 则提供可靠但远小于 recurrence 的 no-overlap 正增量。

## 6. 对 Motivation V1.2 后续诊断的文献启示

在提出新架构前，最有信息量的顺序是：

1. 将 CoPPS/BATA 一类强历史模型接入同一 target-aware evaluator，先判断其总体提升是否
   真正提高 no-overlap history delta；
2. 在冻结训练人口上使用 repeat/no-overlap 平衡采样或分层 loss，检查容易的 repeat 样本
   是否主导梯度；
3. 使用 query-relevant history selection 和 query-matched hard negatives，区分“筛掉噪声”
   与“学会跨商品偏好”两个机制；
4. 以同类别/同属性但不同 ID 的历史替换做最便宜的可逆 probe，判断模型是否保持预测；
5. 先完成当前多方法与 witness 的实测，再依据结果决定是否需要新的机制研究；本文不预先
   授权后续架构。

候选技术来源可以归纳为四类：CoPPS/COCA 的对比增强，BATA/HMPPS/UR4Rec 的
query/candidate-conditioned 历史读取，CAMI/MemRerank 的兴趣分解或记忆压缩，以及
UniSRec/RecFormer 的内容语义表示。它们目前都是待判别的机制假设，不是已经证明适用于
本项目的解决方案。
