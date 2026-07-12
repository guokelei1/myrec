# 10 - 方向评判与最终决策

状态：PPS 场景保留；C01--C80 architecture search 已终止，无 validated primitive，
没有 C81。当前进入 `doc/31` 的 R0 problem-discovery reset；Failure Card 通过前不
授权新 architecture implementation/training。

当前证据改变了数据轨道的使用方式：KuaiSearch 仍是 nominal main，Amazon 是当前
唯一严格确认 full-token semantic history source 的 text-rich track；JDsearch 只有在
目标 claim 所需信息对象存在时才作为 conditional robustness anchor。不能再要求
一个依赖 plaintext WordPiece 的 semantic primitive 无条件在 JDsearch 复现。

历史 M3/M4、C5-R2/R3 和 C01--C80 的构念修订与数值保留在 `doc/16`--`doc/30`、
`reports/` 和 `doc/dev_log/20260712_c01_c80_terminal_retrospective.md`，不再提供
当前设计授权。

输入：`gemini.md`、`55high.md`、`55pro..md`、`glm.md` 四份调研 + `07_paper_design_constraints.md` 约束 + 本次独立核实（2026-07-08）。

## 0. 结论

只推荐一个场景、两个 text-capable evaluation track 和一个 conditional anchor：

```text
场景：真实 query 条件下的个性化商品排序
（Query-conditioned Personalized Product Ranking / Personalized Product Search, PPS）

主轨：KuaiSearch（明文中文，全信号，2026 新发布，benchmark 空白期）
副轨：Amazon-C4 + Amazon-Reviews-2023（英文明文，query 半合成，已有 MemRerank 可打）
条件式鲁棒性锚点：JDsearch（仅验证其信息对象能够支持的 claim）
```

不推荐第二个独立场景。CRS、NBR+recipe、Bundle、Review-based 全部排除（第 4 节给出一票否决理由）。

## 1. 评判依据

### 1.1 四份调研的共识（最强证据）

四份调研由不同来源独立完成，**全部把 PPS 排在第一位**：

| 调研 | 第一推荐 | 首选数据 | 第二推荐 |
|---|---|---|---|
| gemini.md | 个性化商品搜索 | Amazon-C4 + Amazon-Review-2023 | CRS |
| 55high.md | query-conditioned personalized product ranking | KuaiSearch | JDsearch / Coveo |
| 55pro..md | Personalized Product Search under Explicit Intent | JDsearch | PSCon/U-NEED（CRS 类） |
| glm.md | S1 PPS（3.5/5 但排第 1） | JDsearch 锚点 + Amazon-2023 构造 | CRS（4/5 备选） |

四份调研在第二名上全部分歧，在第一名上全部一致。这种模式本身就是信号：PPS 是唯一同时满足"单一自然场景、论文谱系深（HEM→ZAM→TEM→RTM→JDsearch→APeB）、任务张力已被实证测量、fixed-candidate ranking 评测干净"的场景。

### 1.2 任务张力是已测量的，不是猜想

这是 PPS 相对其他场景最大的优势——核心缺口有文献实证，不需要我们自己先证明缺口存在：

- ZAM (CIKM 2019) 与 JDsearch (SIGIR 2023) 实证：**个性化并不总是改善搜索**，收益依赖 query 特性。固定权重融合历史的模型会过度个性化，query-only 模型会欠个性化。
- APeB (2026) 实证：LLM agent 在 PPS 上的主要失败模式是**历史使用低效**——能处理显式 query，但无法从长噪声历史中发现偏好。
- 现有方法（HEM/ZAM/TEM/MAI）都是浅层或固定权重融合，"当前 query 应该决定用哪段历史、是否个性化"这个三方关系（query × history × candidate）没有被充分建模。

这正好构成 07 号文档要求的 insight 模板（见第 2.2 节）。

### 1.3 本次独立核实改变了 glm.md 的核心判断

glm.md 给 PPS 只打 3.5/5，唯一扣分项是："没有任何单一公开数据集同时具备 {真实 NL query, 长期用户历史, 丰富 item 文本, 候选池 ranking 标签}"。**这个判断已过时**。本次核实（2026-07-08）：

1. **KuaiSearch 已完整公开可下载**（GitHub README："All the codes and datasets are now public"，HuggingFace `benchen4395/KuaiSearch`，MIT license）。五张表齐全：
   - user 表（33.2 万用户：性别/年龄/地域）；
   - item 表（1860 万商品：title/brand/seller/三级类目，**明文**）;
   - recall 表（257 万条：query + impressed/clicked/purchased item ids，即真实曝光候选池）;
   - ranking 表（8140 万条：query、search_entrance、recent clicked/purchased items、target item 特征、is_clicked/is_purchased 标签）;
   - relevance 表（4.6 万条 query-item 0-3 分人工相关性）。
   - 且明确**不做流行度过滤**，保留冷启动用户和长尾商品。
2. **Amazon-C4 确认存在且已有历史配套**：HuggingFace 上已有 `McAuley-Lab/Amazon-C4` 和 `zhiyuanpeng/amazon-c4-user-purchase-history`（为每条 C4 query 回溯该用户在目标交互之前的购买历史）。同时 **MemRerank (arXiv:2603.29247, 2026) 已在 C4+历史上做个性化重排**——协议现成、有 2026 年 SOTA 可打，但也意味着该轨道不再是空白。

结论：四要素齐全的数据集**现在存在了**（KuaiSearch），而且因为它 2026 年 2 月才发布，其上"语义个性化 / query-aware history selection"方向还没有 benchmark 论文占位。这是一个时间窗口。

## 2. 推荐方向（唯一主线）

### 2.1 场景定义

```text
用户发出真实商品搜索 query 后，结合用户历史行为（点击/购买序列，含商品明文文本）、
候选商品文本/属性和曝光候选集，对固定候选池做个性化排序。
```

### 2.2 Insight 模板（按 07 号文档 Section 1 填写）

```text
Observation: 在测试的 B0b/D2s history bundle 内，稳定增益集中于当前 candidate
             是否精确出现在 strictly-prior history；coarse category affinity
             没有独立显著增益，加入后反而削弱 item-only 排序。

Benchmark consequence: item-only D2s（mean NDCG@10 0.3453755）取代 full D2s
             成为静态 baseline-to-beat。任何未来 semantic/identity claim 必须
             证明增益超出 exact-item memorization。

Architecture consequence: 设计一个 candidate-conditioned evidence-fidelity
             calibration **LLM4Rec/Transformer primitive**：保存可靠的 exact recurrence，只在
             query/history/candidate 联合证据支持时接纳可迁移 residual。它是
             待证的设计假设，不是把 exact recurrence 升格为论文 primitive。

Falsification: 正式实现/训练前先冻结廉价 gate。repeat-present 不得弱于
             item-only；4,677 个 non-repeat history-present 请求上必须稳定
             优于 D2p；coarse-only、wrong-user、shuffled-event、query-masked
             evidence 不得复现该收益；no-history 必须逐请求退化为 D2p。
```

该结论来自先锁定后执行的三 seed C5-R3 removal ablation。575,609 个候选行的
item+category 均精确复现 executable B0b（最大误差 7.1e-15）；item-only vs
D2p 3/3 显著，category-only vs D2p 0/3 显著，full vs item-only 3/3 显著为负。
结论只限定于测试的 history heuristic，不外推成“所有个性化都是重复商品”。

### 2.3 数据轨道

| 轨道 | 数据 | 作用 | 已核实状态 |
|---|---|---|---|
| 主轨 | KuaiSearch（优先 Lite/抽样） | 主评测：明文 query + 历史 + 候选 + 双标签，全部主张在此产生 | 公开可下载，MIT |
| 副轨 | Amazon-C4 + Amazon-Reviews-2023（+ purchase-history 配套集） | 英文 NL 验证 + 与 MemRerank/BLAIR 直接对比；query 为 review 改写（半合成），按 07 号文档要求诚实标注 | 公开，HF 可下载 |
| 锚点 | JDsearch | 匿名文本下复现机制增益 → 证明门控/历史选择机制不依赖明文语义，回应 "gains from text access" 攻击（07 §9/§15） | 公开（CC BY-NC-SA） |

三条轨道共用同一个模型契约（query + 历史 + 固定候选，缺失证据打 mask），不写 `if dataset == X` 分支——直接满足 07 号文档的 unified interface 约束。

### 2.4 为什么这个方向最可能出有创新性的 insight

1. **当前最清楚的机制缺口是历史证据可靠性没有被正确建模。**
   强 history 数字可由 exact recurrence 主导；category 语义通道不但无独立
   增益，还显著拖累 item-only。代表性的 sequence、CTR、PPS-neighbor 与
   train-fitted residual 方法也未超过 item-only 水线。这说明已测试系统确实
   不够好，但只允许表述为“这些代表性方法没有可靠利用超出重复商品的历史
   证据”，不能外推成所有个性化方法都失败。
2. **KuaiSearch 空白期**：官方 baseline 是工业 CTR 模型（DNN/DCN/DIN）和非个性化相关性模型，没有任何"语义用户记忆 / query-aware history selection"工作占位。第一篇做系统性语义个性化分析的论文有天然引用位。
3. **实验矩阵在三份调研中已收敛**（55high §8.6、55pro §5、glm §7 几乎相同）：popularity/BM25/dense encoder/官方 CTR ranker/SASRec 类/query-aware attention/静态混合/LLM rerank 上界。控制组齐全，直接满足 07 §6 的 control families。
4. **原 recovery ladder 与 C01--C80 已完整结束。** 它们否定了各自有限 claim，
   但没有识别 ordinary full-token Transformer 的具体 defect。下一步是先调好强
   baseline 并形成 Failure Card，而不是把宽泛 evidence-fidelity 直接写成 primitive。
5. **若未来形成 proposed architecture，系统族仍为 LLM4Rec/Transformer。**
   LM/Transformer 必须直接联合建模 query/history/candidate 并产生排序 logit；
   该 eligibility 是项目约束，不构成当前 architecture authorization。

### 2.5 主要风险与对策

| 风险 | 对策 |
|---|---|
| KuaiSearch 太新，split/label 细节未经社区检验 | 第一步做数据审计（55high §11 清单）；发现问题即公开审计结果，本身是贡献 |
| 33.6% 请求 history 为空，非空 history 也可能偏离当前候选域 | 先分开报告 no-history/repeat/non-repeat；no-history exactness、wrong-user、event permutation 只在对应 claim 中 binding |
| 中文数据 + 英文审稿 | 副轨 Amazon-C4 提供英文结果；主张写成 query 属性函数而非语言相关结论 |
| MemRerank 等 2026 工作抢跑 C4 轨道 | 主轨在 KuaiSearch；C4 上直接把 MemRerank 当 baseline 打，差异化点是轻量端到端 + 何时个性化分析（MemRerank 是 inference-only 重系统） |
| 8100 万行 ranking 表算力压力 | 用 Lite/时间抽样；07 号文档允许早期单数据集单种子 |

## 3. 副方向（仅作为主线证伪后的退路，不并行开工）

若未来另立协议的 system 无法超过 item-only 水线，退到：

```text
Amazon-C4 复杂 query 个性化重排（gemini.md 主推）
```

理由：它是同一场景的另一实例，前期数据审计和 baseline 代码全部可复用；C4
的长描述性 query 可检验 residual 原语是否跨 query 分布成立。若仍失败，论文
保持当前 KuaiSearch benchmark + component analysis 结论，而不是修改已冻结
insight。当前尚未启动正式 system 训练，但 architecture/protocol formulation
已经开始获授权；只有新的 pre-implementation gate 失败时，才回到纯
benchmark + analysis 路径。
**不选 CRS 作退路**（理由见下）。

## 4. 明确排除的方向（一句话理由）

| 方向 | 排除理由 |
|---|---|
| CRS（glm/gemini 的第二名） | ranking 评测系统性不干净：候选池每篇论文定义不同、~50% 准确率来自重复捷径、增益随 LLM backbone 而非架构（Kostric & Balog 2026）。与 07 号文档的 evidence hygiene / claim separation 约束正面冲突，需要先自建并捍卫评测协议才能开始做方法——成本高、被攻击面大 |
| NBR + recipe intent（glm S6） | intent 标签必须拼接构造（Instacart×Recipe1M+），违反"自然场景、非人为拼接"底线，评审第一问就是"intent 从哪来" |
| Bundle | 两大 benchmark 无 item 文本，无自然 query，LLM-for-Bundle 2024-25 已快速涌入 |
| Review-based 序列推荐 | 只有两种证据（历史+评论），无 query/intent，不构成多证据张力 |
| 搜推一体化（KuaiSAR/UniSAR） | 论文问题必然扩成"统一搜索+推荐系统"，违反 07 号文档单一原语/复杂度预算约束 |
| MIND 新闻 / ESCI / WANDS 单独成篇 | 分别缺 query、缺历史、缺历史——只能当辅助评测 |

## 5. 下一步（按 07 §7 实验顺序）

1. 完成三轨 information-object、holdout 和 power audit。
2. 在 KuaiSearch 与 Amazon 复现 ordinary full-token true/null/wrong observability；
   shuffle 默认只报告，不再作为通用 corruption gate。
3. 按与 trainable baseline 对称的预算正常调好 ordinary full-token joint
   Transformer。
4. 在 strong baseline 上发现并独立复现 ranking-relevant failure，排除简单修复，
   形成 `Fxx` Failure Card。
5. Failure Card 通过后才从 failure 推出一个 `Hxx`，按 mechanics、learnability、
   utility、specificity、attribution 顺序做 bounded dev iteration。
6. 对唯一 survivor 做 frozen independent confirmation；失败不 rescue，test 最后
   只运行一次。

## 6. 核实来源

- [KuaiSearch paper (arXiv:2602.11518)](https://arxiv.org/abs/2602.11518)
- [KuaiSearch GitHub（已确认数据公开）](https://github.com/benchen4395/KuaiSearch)
- [Amazon-C4 (HuggingFace, McAuley-Lab)](https://huggingface.co/datasets/McAuley-Lab/Amazon-C4)
- [Amazon-C4 用户购买历史配套集](https://huggingface.co/datasets/zhiyuanpeng/amazon-c4-user-purchase-history)
- [MemRerank (arXiv:2603.29247)](https://arxiv.org/pdf/2603.29247)
- [BLaIR / Amazon-Reviews-2023 (arXiv:2403.03952)](https://arxiv.org/html/2403.03952v1)
- [JDsearch GitHub](https://github.com/rucliujn/JDsearch)
