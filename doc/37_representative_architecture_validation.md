# 代表性架构 motivation 验证方案

日期：2026-07-15

状态：**baseline/动机探索协议；不授权 proposed architecture、confirmation 或 test。**

## 一、直接决定

后续不再只用“一个 BGE、一个 Qwen”来支撑 Transformer/LLM4Rec 的动机。
核心验证矩阵固定为三个计算范式：

| 核心角色 | 固定代表 | 为什么有代表性 | 在本项目中回答什么 |
|---|---|---|---|
| 通用语言 Transformer | Qwen3-Reranker-0.6B | ranking-pretrained decoder LM；历史以文本 token 直接进入普通 Transformer | 直接序列化 history 的 ordinary LLM4Rec 是否稳定完成 base + history composition |
| 推荐原生 Transformer | HSTU，ICML 2024 | 为高基数、非平稳行为序列设计的生成式 sequential transducer | 问题是否超越语言模型 prompt/serialization，并在 rec-native 序列核中仍然存在 |
| 学术优化模型 | LLM-SRec，KDD 2025 | 明确研究 LLM 是否理解推荐序列，并以冻结 CF-SRec 表征蒸馏增强冻结 LLM | 一篇针对该问题的已发表修复机制能否消除现象，以及消除的是哪一部分 |

BGE-reranker-v2-m3 保留为**第四个额外 encoder anchor**。它已经积累了跨数据的
QC/FULL 证据，不能丢弃；但它不再被误写成三个核心代表之一。

这个选择不是期待三个模型一定给出相同结果。结果一致会支持跨范式问题；结果分叉则能更
准确地定位 failure locus。预先假设“大家应该都差不多”会产生结论导向偏差，因此本协议在
打开新结果前同时冻结一致和分叉两类解释。

## 二、为什么第三个选 LLM-SRec

LLM-SRec 的论文题目就是 *Lost in Sequence: Do Large Language Models Understand
Sequential Recommendation?*。它先诊断现有 LLM4Rec 对序列信息利用不足，再用预训练的
CF sequential recommender 提供用户序列表征，通过轻量投影/蒸馏模块将其对齐到冻结 LLM
产生的用户和候选表征。这个机制与当前的核心问题——历史是否被转换成有效的候选相对排序
增量——直接相交。

相较其他候选，它更适合当前实验：

- TALLRec 主要是参数高效 instruction tuning，和“普通 Qwen 适配”过近，不能提供足够新的
  计算范式；
- Recformer 是很好的文本序列模型，但以 Longformer 进行 next-item recommendation，加入
  query-conditioned fixed-slate ranking 需要更大任务改造；
- A-LLMRec 也是可行候选，但更早、两阶段对齐色彩更强，且 LLM-SRec 对“序列是否真的被
  LLM 捕获”的论证和本轮问题更直接；
- RPG 等 semantic-ID generative retrieval 方法改变了候选生成和评价接口，不适合用同一
  fixed candidate slate 做 true/null/wrong 反事实；
- 更新的 LoRA/graph 方法如果没有稳定的官方实现或需要引入额外图信息，会把“历史融合”与
  “新信息对象”混为一谈。

因此，LLM-SRec 不是因为它最容易得到我们想要的结果，而是因为它是对工作假设最强、最
直接的已发表反例候选。若它完全解决问题，原来的 broad motivation 必须缩窄。

## 三、四个模型的边界不能混写

### 3.1 Qwen：ordinary raw-history language baseline

Qwen 使用相同的 ranking-pretrained checkpoint、候选集和训练样本构造，形成：

- `QWEN-QC`：query + candidate；
- `QWEN-FULL`：query + 严格先验 history + candidate；
- 对同一 `QWEN-FULL` checkpoint 运行 true、null、wrong history。

当前 Lite 结果只是工程和先导证据。paper-level 代表性要求在 KuaiSearch Full 上重新完成
QC adequacy、token coverage、FULL true/null/wrong 和 base/history accounting。0.6B 模型可在
现有 A40 上做受限全参数适配；若改为 LoRA，QC 与 FULL 必须使用相同 rank、target modules、
更新次数与模型选择规则，并把它标记为 PEFT recipe，而不能和既有全参数结果混算。

### 3.2 HSTU：rec-native boundary control

HSTU 不是 LLM，也不是 full-token 文本 cross-encoder。它的价值恰恰在于：使用推荐领域专门
设计的 sequential transduction unit 处理行为序列，检查当前现象是不是“把历史写进 LM
prompt”才会出现。

HSTU 原论文任务是序列推荐，不是 query-conditioned product reranking。因此本项目会明确
标记为 **official HSTU core + PPS task adapter**，不能声称复现论文主表。适配边界为：

- HSTU block、位置/时间建模和序列 mask 保持官方实现；
- query 与 candidate 文本使用同一个冻结文本编码器产生内容向量，再经可训练投影进入固定
  维度；item ID、action、category 和 time 只使用 standardized record 中共同可得字段；
- 同一 HSTU 用户状态对一个 request 的所有候选共享，query 与 candidate 通过统一 scorer
  计算候选分数；
- `HSTU-QC` 删除历史序列输入，但保留相同 query/candidate 内容边界和 scorer capacity；
- `HSTU-FULL` 在同一 checkpoint 上运行 true/null/wrong history；
- SASRec 使用官方仓库内实现和相同输入/scorer，作为 architecture-specific control，避免把
  HSTU 的效果误归因于通用序列训练。

HSTU 若不出现 composition failure，只能说明 rec-native sequence primitive 或其接口可能
避免了 ordinary LM fusion 的问题；它不会直接证伪 Qwen/BGE 上的实测。

### 3.3 LLM-SRec：published repair boundary

官方 LLM-SRec 使用预训练并冻结的 SASRec、冻结的 LLaMA 和轻量对齐/预测模块。为了控制本地
资源并隔离机制，本项目第一版采用：

- 冻结的 `Qwen3-Reranker-0.6B` 作为语言 backbone；
- 在 training split 上预训练并冻结的 SASRec 作为 CF-SRec teacher；
- 按论文机制实现的 history/item embedding injection、user/item projection、retrieval 与
  representation-alignment losses；
- 把真实 query 加入 user-side prompt，并对 standardized candidate slate 逐项打分；
- 只训练论文允许的轻量模块，不更新 Qwen 或 SASRec teacher。

这是 **paper-mechanism-faithful、task-interface-adapted** 的实现，不是原论文 LLaMA-3.2-3B
数字复现。query 的加入和固定候选集评分都是 PPS 任务所需的显式适配，必须在论文中披露。

官方 GitHub 在本次审计的 commit 下没有 LICENSE 文件，因此不把其源码复制进仓库，也不从
官方实现直接拷贝函数。仓库只保存来源/commit、论文机制说明和独立实现边界；若作者后续补充
许可证，再单独审计是否允许 vendor。

### 3.4 BGE：现有 encoder anchor

BGE 保留既有 `E-QC/E-FULL` 结果和后续 frozen confirmation 价值。它回答 encoder
cross-encoder 是否共享现象，而 Qwen 回答 decoder LM。由于这两个模型都把全部文本直接放进
joint token sequence，它们共同构成 ordinary full-token 证据，但不能替代 HSTU/LLM-SRec。

## 四、数据集矩阵

不是每个数据集都必须通过，也不允许一个数据集一票否决。核心结论需要至少两个独立来源的
复制，同时保留有解释价值的正边界。

| 数据 | Qwen | HSTU | LLM-SRec | BGE | 角色 |
|---|---:|---:|---:|---:|---|
| KuaiSearch Full | 核心 | 核心 | 核心 | 已有/确认 | 真实自然语言 query、真实搜索候选；主定位面 |
| Amazon-C4 + Reviews-2023 history | 核心 | 核心 | 核心优先 | 已有 | 英文、丰富商品文本；接近 LLM-SRec 原始 Amazon 条件的正边界 |
| JDsearch | 不做语言主结论 | 功能性 anchor | 不作为核心 | 已有 | 匿名 term/item 下的行为规律；只验证 functional composition |

执行时先在 Amazon-C4 打通 LLM-SRec，因为论文原本就在 Amazon Reviews 2023 上验证，字段和
训练方式最接近；再迁移到 KuaiSearch Full，以真实 query 检查结论是否仍成立。HSTU 先在
KuaiSearch Full 验证 adapter 和 repeat positive control，再做 Amazon-C4。Qwen 等 Full 数据
准备完成后，在 KuaiSearch Full 与 Amazon-C4 采用同一冻结 recipe 执行。

JDsearch 没有自然 plaintext，不要求 Qwen/LLM-SRec 在它上面证明语言机制。强行把匿名 token
塞给语言模型只会产生一个已知不充分的 baseline，而不是增加论文可信度。

## 五、统一评价：比较的不是 leaderboard 名次

每个模型家族都必须在相同 standardized records、候选集合和共享 evaluator 下输出 request ×
candidate score。训练/打分代码不得读取 development/test qrels。评价前断言 candidate hash。

每个家族至少提供以下五个量：

1. `QC`：独立训练的 query--candidate base；
2. `FULL-null`：历史模型在空历史下的能力；
3. `FULL-true`：真实历史下的最终能力；
4. `FULL-wrong`：匹配的其他用户历史下的 provenance control；
5. candidate-relative response：同一 slate 内 `score_true - score_null` 去除 request 共模后的
   activity、方向、utility 和 conversion efficiency。

核心恒等式保持不变：

```text
FULL-true − QC
= (FULL-null − QC) + (FULL-true − FULL-null)
= base retention      + history utility
```

只报告 `true-null > 0` 不算成功，因为它可能只是补回联合历史训练造成的 base loss。只报告
overall NDCG 也不算，因为 repeat traffic 可能掩盖 strict-nonrepeat failure。

### 5.1 模型 adequacy 先于 motivation 判定

一个家族只有满足以下条件才有资格支持负面结论：

- QC 至少明显优于 lexical/sanity control，且没有训练崩溃；
- FULL 在 repeat surface 上能响应历史，作为 observability 正控制；
- query、candidate 和冻结的 history budget 均通过 token/field coverage；
- true/null/wrong 使用同一 checkpoint、同一候选 hash、同一 scoring signature；
- 预先声明的小型 tuning budget 已用完或出现清楚的正常训练区间；
- 至少两个 seed 方向一致后，才把现象提升为 replicated exploratory evidence。

弱 QC、截断历史、错误 adapter 或完全不响应历史的模型都只能记为 implementation/recipe
failure，不能用来证明 Transformer failure。

## 六、结果分叉在实验前就写清楚

| 观察 | 允许的结论 | 不允许的结论 |
|---|---|---|
| Qwen、HSTU、LLM-SRec 在两个独立数据都出现相似 tradeoff | controlled composition 是跨语言/推荐架构的 shared problem | 所有 Transformer 都不行 |
| Qwen/BGE 失败，HSTU 与 LLM-SRec 明显改善 | failure 主要属于 ordinary full-token history fusion；rec-native/teacher-guided interface 是有效线索 | HSTU/LLM-SRec 证明了我们的新架构 |
| HSTU 失败，LLM-SRec 改善 | 单纯换 sequence primitive 不够；显式序列表征监督或对齐可能关键 | 一定需要蒸馏 |
| HSTU 改善，LLM-SRec 失败 | rec-native transduction 比 frozen-LM alignment 更适合该信息对象 | LLM4Rec 方向应被放弃 |
| LLM-SRec 只在 Amazon 改善 | 方法依赖丰富文本/构造 query 或 source 条件，不能外推自然搜索 | motivation 被 Amazon 一票否决 |
| 普通 dropout/anchoring/objective 已关闭 tradeoff | 问题属于训练或 objective；架构必要性不成立 | 继续包装为新 attention 必要性 |
| 三者都能稳定保 base 并产生高效增量 | 当前 motivation 被强证伪，应从新观察重新定义问题 | 继续筛切片直到失败出现 |

对本轮最有价值的并不一定是“三个都失败”。若一篇已经发表的优化方法成功，而 ordinary Qwen
和 BGE 失败，故事反而更完整：现象真实、普通方法受影响、已知结构原则可以缓解，但其在
query-conditioned ranking 上仍有未解决边界。只有在结果出来后才能判断这个边界是否足以
支撑我们自己的贡献。

## 七、执行顺序与停止点

### Stage A：工程与边界复核

1. 登记并 vendor Apache-2.0 的 HSTU 官方源码；只先启用官方 PyTorch kernel，不把 Triton/CUDA
   优化编译当成科学结果的前置条件。
2. 对 HSTU 写 standardized adapter、SASRec matched control 和手工小 fixture 测试。
3. 对 LLM-SRec 只登记无许可证的外部源码边界；根据论文公式独立实现最小机制。
4. 为两个 adapter 测试 true/null/wrong 不改变 request/candidate identity 和 candidate hash。

### Stage B：最便宜的完整链路

1. HSTU/SASRec 在 KuaiSearch Full 的小型 train/dev scout 上通过机械、learnability、QC 和 repeat
   正控制；
2. LLM-SRec 在 Amazon-C4 小型 scout 上通过 frozen-backbone、teacher 与 candidate scoring
   一致性检查；
3. 只有通过 adequacy 的实现才进入完整 exploratory train/dev。

### Stage C：跨数据复制

1. HSTU：KuaiSearch Full → Amazon-C4 → JDsearch functional anchor；
2. LLM-SRec：Amazon-C4 → KuaiSearch Full；
3. Qwen：Full 准备完成后在 KuaiSearch Full 与 Amazon-C4 跑 matched QC/FULL；
4. 用 BGE 既有结果完成 encoder anchor，不为追求数量无限加模型。

### Stage D：Failure Card 判断

只有当至少两个核心架构、两个独立数据源形成可解释的共同失败，并且普通 objective/dropout/
anchoring 等 standard repair 不能同时恢复 base retention 与 history utility，才允许把现象写成
architecture Failure Card。否则按第六节收缩到 full-token、objective、interface 或特定数据
边界。

## 八、来源与本次审计结果

| 项目 | 来源 | 审计 commit | 许可证/处理 |
|---|---|---|---|
| HSTU / Generative Recommenders | https://github.com/meta-recsys/generative-recommenders | `6135bc30398f97e5786674192558d91f2ef2fa90` | Apache-2.0；允许在 `baselines/hstu/` 固定官方源码 |
| HSTU paper | https://arxiv.org/abs/2402.17152 | ICML 2024 | official architecture reference |
| LLM-SRec | https://github.com/Sein-Kim/LLM-SRec | `b81019ca655fb759cee895924b8b6c7cc0f0cce9` | 本次审计未发现 LICENSE；不 vendor、不复制实现 |
| LLM-SRec paper | https://arxiv.org/abs/2502.13909 | KDD 2025；DOI `10.1145/3711896.3737035` | 根据论文机制独立实现，并披露 task adaptation |

当前机器有四张约 48GB 的 A40；Qwen-0.6B、research-scale HSTU 和冻结 Qwen-0.6B 的
LLM-SRec 机制验证都在资源范围内。真正的风险不是显存，而是 HSTU/LLM-SRec 到 PPS fixed-slate
任务的适配是否忠实，所以先完成 adapter contract 和小型可证伪测试，再投入完整训练。

## 九、KuaiSearch Lite 首轮落实结果

截至 2026-07-15，三个 sequence-oriented 边界已经越过 mechanics：HSTU 与同接口 SASRec
完成 QC/FULL 训练和 true/null/wrong 评分；LLM-SRec 完成冻结 SASRec teacher 物化、全 Lite
轻量训练和同 checkpoint 三条件评分。统一结果见
`reports/history_response_gap_lite_representative_architecture_decision.json`。

首轮观察与 ordinary Qwen/BGE 的形状一致：history-present 请求上 response 广泛存在，而 HSTU、
SASRec、LLM-SRec 的 strict-nonrepeat candidate-relative direction 均与随机相容，排序 utility
不稳定。这个结果说明原始 Lite 现象没有因为换成 rec-native transducer 或显式序列蒸馏而自动
消失。

但本轮没有通过 representative-family adequacy gate：HSTU/SASRec 的 QC 低于 BM25，HSTU
repeat utility 正控制也不够强；LLM-SRec 只有一轮完整训练、没有独立 QC，且尚未在更接近原
论文的 Amazon 条件上正常验证。因此当前状态是 **supportive pattern**，不是 binding shared
blind spot。下一步先修 baseline/task-interface adequacy，再做第二 seed 和 Full，而不是据此
进入 proposed architecture。

本协议冻结后，共享 sequence adapter 已在 KuaiSearch Lite、现有 Full scout、Amazon-C4 和
JDsearch 的 train/dev records 上完成无 qrels 审计，时间因果、候选唯一性、query 末位 token
和统一字段序列化均通过。审计同时发现 dev candidate/item ID 的 train-vocabulary OOV 很高，
因此 HSTU 不得退化为纯 ID lookup；冻结的 query/item 文本内容向量是跨 split 评分的必要输入，
不是为了改善结果而后加的可选组件。对应审计在
`reports/representative_sequence_adapter_*.json`。

当前实现状态也已越过纯文档阶段：锁定的官方 HSTU 与同仓 SASRec 已在独立官方依赖环境中
通过 GPU forward/backward 和 architecture-parameter finite-difference 检查；official core +
PPS adapter 又在真实 KuaiSearch Lite records 上通过 true/null/wrong 候选一致性与梯度 smoke。
LLM-SRec 则依据论文 Eq. 1--4 独立实现 retrieval、MSE distillation 和 uniformity，完成 CF item
embedding 注入、冻结 Qwen 的 `UserOut/ItemOut` 抽取，并在真实 Lite records 上确认 Qwen
backbone 无梯度、轻量模块可训练。两项 smoke 都明确使用 mechanics-only synthetic features，
不是模型 adequacy 或 motivation 结果；对应报告为 `reports/hstu_official_core_smoke.json`、
`reports/hstu_pps_adapter_lite_smoke.json` 和
`reports/llm_srec_pps_adapter_lite_smoke.json`。
