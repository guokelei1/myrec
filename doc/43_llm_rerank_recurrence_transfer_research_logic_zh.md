# Motivation：LLM 个性化重排中的 recurrence–transfer 失衡

状态：2026-07-16 论文动机工作稿。本文只保留研究问题、核心证据和待完成的统一验证；
冻结实验边界与完整相关工作分别见
[`40_transformer_recurrence_transfer_motivation_v1_zh.md`](40_transformer_recurrence_transfer_motivation_v1_zh.md)、
[`41_motivation_v11_current_conclusion_zh.md`](41_motivation_v11_current_conclusion_zh.md) 和
[`42_recurrence_transfer_related_work_zh.md`](42_recurrence_transfer_related_work_zh.md)。

## 1. 背景与研究问题

Ranking 是推荐系统中的核心任务：系统先从大规模商品库中召回候选，再根据用户当前需求
和长期偏好决定候选的展示顺序。本文关注的 reranking，是对已召回的小规模候选集合做
最后一阶段的精细排序。近年来，LLM 的文本理解、上下文建模和指令遵循能力使其逐渐被
直接用于这一阶段。[InstructRec](https://arxiv.org/abs/2305.07001) 将推荐与搜索任务统一
为指令学习，[TALLRec](https://arxiv.org/abs/2305.00447) 通过轻量参数高效训练使 LLM
适配推荐，[RecRanker](https://doi.org/10.1145/3705728) 则面向 top-k recommendation
系统研究 pointwise、pairwise 和 listwise 的 LLM 排序。它们代表了同一个基本方向：让
语言模型读取当前需求、候选商品以及用户信息，并直接产生候选排序分数或次序。

用户历史是个性化排序最重要的信息来源之一。即使两个商品同样符合当前 query，不同用户
仍可能偏好不同品牌、价格带、材质、功能或风格。因此，一个
history-conditioned LLM-based reranker 理应能够从历史行为中识别与当前 query 相关的
偏好，并利用这些偏好提高候选排序质量。已有 LLM 推荐与个性化商品搜索工作也普遍将
“加入历史后总体指标提高”视为个性化有效的证据。

然而，总体 NDCG 或 MRR 的提高并没有说明模型如何利用历史。如果历史中的旧商品恰好
再次成为当前目标，模型只需识别重复商品就可能得到较大收益；这并不等价于模型已经学会
从旧行为中抽取偏好，并把偏好迁移到一个历史中从未出现的新商品。因此，本文首先追问：

> LLM-based reranker 的历史收益，究竟有多少来自对历史商品的直接 recurrence，
> 又有多少来自跨商品的 preference transfer？

## 2. Recurrence 与 transfer 的可检验分解

我们把历史可能产生的收益分为两类。

**Recurrence** 指当前正例商品已经出现在用户历史中。模型可以通过 exact-item identity、
名称或高度相似的文本重新找到该商品。**Transfer** 指当前正例从未出现在历史中，模型
必须先从其他历史商品中抽取偏好，再将偏好对齐到当前候选。后者要求模型同时完成
query-aware history selection、偏好抽象和跨商品匹配，不能仅依赖目标商品复现。

为分别估计两种能力，我们对同一个 checkpoint、同一个请求和同一个候选集合输入真实
历史与空历史，并计算逐请求 graded NDCG@10 的 `full-history − null-history` 差值。核心
诊断 surface 为：

- `target-repeat`：当前正例商品出现在历史中，用于测量 recurrence；
- `target-nonrepeat/no-candidate-overlap`：正例不在历史中，且整个候选集合与历史商品没有
  ID 交集，用于测量 strict transfer；
- `target-nonrepeat/other-candidate-overlap`：正例不重复，但其他候选与历史重叠，单独报告
  为 overlap surface，不与 strict transfer 混合。

strict-transfer surface 切断了历史和当前候选之间的 exact-item recurrence。若真实历史
仍能可靠提高目标排序，收益才可以归因于某种可迁移信息。除逐 surface 的平均增量外，
我们同时报告它们对 all-request 总体增益的人口加权贡献，避免一个高增益但极小的 slice
主导解释。

## 3. 初步证据与统一 LLM 验证

### 3.1 已获得的初步证据

冻结的 KuaiSearch V1/V1.1 已经测试两个 history-conditioned LLM baseline。下表数值均为
同 checkpoint 的 `full-history − null-history` NDCG@10；“可靠”表示 normalized-query
cluster bootstrap 区间不跨 0。

| 数据集 | LLM baseline | overall | recurrence | strict transfer |
|---|---|---:|---:|---:|
| KuaiSearch | Qwen3-Reranker-0.6B | `+0.0131`，可靠 | `+0.2315`，可靠 | `+0.0032`，不可靠 |
| KuaiSearch | InstructRec / Flan-T5-XL | `+0.0006`，不可靠 | `+0.0338`，可靠 | `−0.0002`，不可靠 |

这两个 LLM 都会响应历史，并在 recurrence 请求上获得可靠收益；但二者都没有在 strict
transfer 上建立可靠增益。对 Qwen 的人口加权分解进一步显示，recurrence 对 all-request
NDCG 的贡献为 `+0.01401`，strict transfer 仅为 `+0.00094`；其他 overlap 请求产生负向
抵消后，最终 overall 为 `+0.01305`。因此，Qwen 已建立的总体历史收益主要来自
recurrence，而不是 transfer。

非 LLM 与跨数据集实验为这一观察提供边界，而不是扩大 LLM 家族结论。KuaiSearch 上的
TEM 同样表现为 recurrence 可靠、strict transfer 不可靠，但后续多 seed 没有稳定复现；
JDsearch full-token ranker 的 strict-transfer 增量为可靠的 `+0.0286`，说明 transfer 并非
不可能，不过 recurrence 增量达到 `+0.4249`，约为前者的 14.9 倍。现有证据因此支持一个
有界观察：历史中的可迁移信号存在，但当前测试模型更容易从 recurrence 获益。

### 3.2 当前证据仍缺少什么

现有两个 LLM baseline 使用不同 backbone、训练目标和实现边界；TEM 与 JDsearch 模型又
不是 LLM。它们足以提出研究问题，却不足以说明该现象能否在多种 LLM reranking 方法中
稳定复现。尤其是，在背景中引用一种 LLM ranking 方法，却用另一套不匹配的架构证明
“这类方法存在问题”，会留下明显的证据断层。

因此，下一阶段不立即提出新架构，而是在 KuaiSearch 上建立 controlled-Qwen 验证。四个
主方法从实验开始前固定，不再根据 transfer 结果从候选池筛选。现有
Qwen3-Reranker-0.6B 作为专用 reranker 强锚点；InstructRec、RecRanker 和 TALLRec 的核心
机制则独立最小重写到同一个约 0.5–0.6B 的通用 Qwen 上。四者共享训练人口、候选集合、
可见信息边界和 evaluator，后三者进一步共享完全相同的通用 Qwen 初始化。训练预算允许
按方法需要做有界调整并完整记录。最终主表固定为：

| KuaiSearch method | backbone role | overall history gain | recurrence | strict transfer | recurrence contribution | transfer contribution |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-Reranker-0.6B | 专用 reranker anchor | 既有结果兼容后复用/复验 | 待统一登记 | 待统一登记 | 待统一登记 | 待统一登记 |
| InstructRec-GeneralQwen | 共享轻量通用 Qwen | 待实验 | 待实验 | 待实验 | 待实验 | 待实验 |
| RecRanker-GeneralQwen | 共享轻量通用 Qwen | 待实验 | 待实验 | 待实验 | 待实验 | 待实验 |
| TALLRec-GeneralQwen | 共享轻量通用 Qwen | 待实验 | 待实验 | 待实验 | 待实验 | 待实验 |

此外，单独实现一个不进入四方法主表的 CoPPS-style transfer witness。它只回答同一
KuaiSearch strict-transfer surface 是否存在可被专门机制恢复的信号：若 witness 提升而
四个 LLM ranker 较弱，将直接加强 `under-exploit` 假设；若 witness 也失败，则应收窄为
当前数据上尚未建立 transfer headroom。UR4Rec 与 HMPPS 不进入本轮。完整执行规则见
[`motivation_v1_2/plan.md`](../experiments/motivation_v1_2/plan.md)。

## 4. 研究假设与目标

已有研究解释了为什么这一区分值得单独研究。
[Repetition and Exploration in Sequential Recommendation](https://staff.fnwi.uva.nl/m.derijke/wp-content/papercite-data/pdf/li-2023-repetition.pdf)
发现序列推荐模型通常在 repeat-next 上远强于 explore-next，容易的重复样本可能成为
accuracy shortcut，并使总体指标掩盖未见目标上的能力不足。这与 recurrence 容易、
transfer 困难的初步观察一致。

另一方面，非 LLM 个性化排序工作说明历史中并非没有可迁移信号。
[ZAM](https://assets.amazon.science/37/52/3425ce394654af4687c7feba6b0f/a-zero-attention-model-for-personalized-product-search.pdf)
根据当前 query 判断哪些历史应该生效；[RTM](https://arxiv.org/abs/2004.09424) 在历史评论
与候选评论之间建立细粒度语义匹配；[CoPPS](https://doi.org/10.1145/3580305.3599287)
使用同类别或知识图谱相似、但 ID 不同的商品构造对比视图；
[BATA](https://doi.org/10.1145/3726864) 则利用 query–query 和 item–item 关系增强
query-aware 个性化。它们没有报告本文的 recurrence–transfer 分解，因而不能证明已经
解决 strict transfer，但证明了类别、属性、评论语义和 query–history 关系可以支持跨商品
偏好建模。

上述证据形成本文的核心研究假设：

> **Current history-conditioned LLM-based rerankers may under-exploit
> transferable preference signals.**

这里的 `may` 是必要的。当前结果尚不能区分 transfer 的天然难度、训练监督不足和 LLM
排序机制不足，也不能推出所有 LLM ranker 都存在这一问题。本文下一步首先检验：在统一
规范下，多种有代表性的 LLM reranking 方法是否稳定复现“recurrence 可靠而 strict
transfer 弱”的失衡。只有该现象跨方法成立，并通过任务能力、历史来源和多 seed 控制，
才值得进一步定位为什么可迁移信号没有被充分利用。

如果后续方法能可靠提高 strict transfer，同时不损害 recurrence 和总体排序质量，它将
构成该假设的边界或可恢复性证据；如果多个匹配 LLM ranker 均复现失衡，则本文获得一个
更坚实的研究起点：问题不再只是“历史是否有用”，而是“为什么 LLM 的历史收益主要停留
在容易的 recurrence，以及如何使其迁移到新的候选商品”。

## 证据索引

- 冻结三模型结果：
  [`../reports/pps_three_transformer_history_surface_audit.json`](../reports/pps_three_transformer_history_surface_audit.json)
- V1.1 多 seed 与 JDsearch 结果：
  [`../reports/pps_motivation_v11_current_summary.json`](../reports/pps_motivation_v11_current_summary.json)
- 统一 LLM ranker 工作计划：
  [`../experiments/motivation_v1_2/plan.md`](../experiments/motivation_v1_2/plan.md)
- 完整相关工作：
  [`42_recurrence_transfer_related_work_zh.md`](42_recurrence_transfer_related_work_zh.md)
