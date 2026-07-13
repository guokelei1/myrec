# 31 - Problem Discovery and Architecture Iteration Protocol

状态：**当前权威 proposed-system 研发协议，2026-07-13 生效；已纳入 R0 Round 1
流程复盘修订。**

持续自动执行、恢复、预算总控与 whole-pipeline end states 由
`32_autonomous_pipeline_controller.md` 规定。

本文取代 `doc/15` 和 `doc/24` 中所有仍带执行含义的旧候选搜索、四路并行、
单次联合 gate 与 GPU 分配规则。`doc/24`、`systems/01_*`--`systems/80_*`、对应
reports 和 dev log 仅作为 C01--C80 历史证据保留，不授权 C81、C80 rescue 或
沿旧候选链继续。

本文不放松以下证据卫生：label isolation、candidate hash、共享 evaluator、dev
预算、test lock、matched control 和 outcome 前冻结 final claim。它修正的是这些
规则的**使用阶段**：开发期用于发现和调好模型，确认期才执行论文级联合审计。

---

## 1. 当前研究边界

当前已建立：

1. KuaiSearch 上被测静态历史增益主要来自 exact recurrence；coarse category
   transfer 未建立，混合会稀释 item-only。
2. Amazon pooled-text history 只产生很弱的 strict-non-repeat signal；普通
   full-token joint Transformer 明确产生 `true-null +0.025298` 和
   `true-wrong +0.035944`。
3. Amazon 的 Q--H、C--H 和 history-read-context token edges 承重；event order
   当前不是已建立的承重变量。
4. C80 只在 pre-label event-permutation mechanical contract 失败；365 个 fresh
   labels 未打开，C80 utility 未知。
5. R0 Round 1 在 KuaiSearch 上复现了 ordinary full-token history observability：
   三 seed `true-null` 均值为 `+0.020936`，`true-wrong` 均值为 `+0.031812`，
   strict-non-repeat `true-null` 均值为 `+0.005388`。
6. 同一 Round 1 full-token family 还不是合格 strong base：true NDCG@10 三 seed
   均值 `0.331028`，低于 item-only `0.345376`；no-history 相对同一静态 base 为
   `-0.013066`，且 replication fold 95% CI 为 `[-0.025734,-0.006001]`。
7. Round 1 的 repeat-query-conflict 与 query-aligned strict-non-repeat 两个 failure
   idea 均被证伪；前者连 item-only 都能处理，后者 full-token 点估计已高于
   D2p/D2s。因此二者不能形成 shared baseline blind spot 或 Failure Card。

当前**未建立**：

- 普通 full-token Transformer 在 PPS 上存在必须由新架构修复的稳定缺陷；
- 一个跨 KuaiSearch、Amazon 和无 plaintext JDsearch 的共同 token-level evidence
  object；
- 任一 C01--C80 primitive 的强-base utility、unique rent 和跨域复现；
- CCF-A 级 proposed architecture。

因此下一阶段的第一个晋级产物不是 architecture proposal，而是一个经过正常调优后
仍然成立的 **Failure Card**。R0 日常迭代只写五字段轻量记录；没有通过的 Failure
Card，不得启动正式架构实现。

### 1.1 Round 1 的论文级判断

Round 1 是有效的 **L1 observability/measurement progress**，不是 CCF-A 级方法贡献。
它回答了“full-token Transformer 能否读到历史信号”，也排除了两个错误问题；但它
没有回答“强模型共同解决不了什么”“普通 Transformer 为什么原生解决不好”以及
“修复后能回收多少整体排序价值”。当前证据最多支持一段扎实的 motivation/negative
analysis，不能单独支撑 proposed-architecture 论文；即使走 measurement paper，仍需
更完整的强 baseline、跨 split/dataset 复现和一个可概括的行为规律。

本协议用下列 contribution ladder 判断进展，而不再用运行数、文档数或候选数代替：

| Level | 必须形成的证据 | 当前状态 |
|---|---|---|
| L0 | 数据、控制和 evaluator 可信 | 已通过 |
| L1 | 有规模、有 ranking 后果的现象可复现 | full-token history observability 已通过 |
| L2 | 至少两个强模型家族共享一个有实践影响的 blind spot | 未通过 |
| L3 | ordinary Transformer 的 native shortfall 被定位，简单修复无效 | 未通过 |
| L4 | 一个 primitive 有 utility、specificity 和 unique rent | 未通过 |
| L5 | independent confirmation 通过，test 仍封存 | 未通过 |

一个 R0 scientific round 必须使 ladder 前进一级，或以有力反证关闭一个原本达到
L1/L2 价值门槛的 thesis。修 donor、YAML、dtype、锁或运行脚本是必要工程工作，但不
计为 scientific round progress。

---

## 2. 两个循环，两个制度

### 2.1 Discovery Loop

目标是发现问题、修实现、调好强 baseline、定位 failure locus。开发者可以查看
development metrics 并在冻结预算内反馈调整。

允许：

- 单 seed provisional 训练；
- 在预注册 search space 内调 learning rate、epoch、capacity、history length、
  dropout、objective 和 initialization；
- 在同一固定 development surface 上重复使用标签反馈；
- 修 mechanical bug、补 property test、增加只用于诊断的 instrumentation；
- 根据结果关闭、收缩或重写尚未进入 confirmation 的 hypothesis。
- 在 `tmp/r0_prototypes/<id>/` 做 CPU/tiny-data disposable prototype，帮助判断
  failure 是否可构造；它不得读取 fresh/dev/test labels、调用 evaluator、进入
  `systems/` 或产生 utility/novelty claim。

必须：

- 每次 evaluator call 写入 `reports/dev_eval_log.jsonl`；
- 每个 trial 只有一个 primary change class；
- score-affecting 修改消耗 trial，不能伪装成 mechanical retry；
- 所有 provisional 结果明确标记，不能进入论文主表。

### 2.2 Confirmation Loop

目标是验证已经发现并调好的一个 survivor。进入后禁止反馈调整。

必须在 outcome 前冻结：

- code/config/checkpoint-selection rule；
- primary metric、MDE、统计方法和 claim-specific controls；
- seeds、candidate hash、数据版本和 run IDs；
- numerical contract、stop rule 和唯一 test-opening 条件。

确认失败即关闭该 claim。失败形态可进入下一轮问题定义，但不得对同一 confirmation
cohort 做局部 rescue。test 只在完整系统冻结后运行一次。

### 2.3 AI 执行默认值

本协议面向 AI 开发系统。系统保持自主判断和创新空间，tracked work product 聚焦
可审计的事实、决定和下一步。

- 优先执行当前最便宜、可逆、能区分结论的实验。
- R0 每轮最多登记 3 个 active failure idea，只允许前 2 个进入 probe；其余只记一行
  parking-lot，不展开设计。
- R0 记录只包含 hypothesis、single change、result、next action、budget；额外文档
  只在产生新 evidence、lock 或 decision 时创建。
- 非常规想法不因机制形式被排除；只要能映射到 observed failure、给出 cheap
  falsifier 并遵守预算，就可以进入 active idea 排序。
- 遇到不影响证据有效性的歧义，采用最简单的合理假设并运行可逆 probe；只有数据边界、
  label safety、破坏性操作或高成本资源不明确时才停下来询问。
- 一个 cheap falsifier 已能关闭问题时按 stop rule 结束该分支；没有新证据时登记
  `stop`、`park` 或明确 blocker。

---

## 3. 版本与状态分离

禁止再用一个连续候选编号同时表示科学假设、实现修复、调参和确认。

| 对象 | ID 例子 | 含义 |
|---|---|---|
| Research reset | `R0-OBS` | source observability / scope reset |
| Failure hypothesis | `F01` | 标准模型的一个可复现 failure |
| Architecture hypothesis | `H01` | 由 `F01` 推出的一个 primitive |
| Implementation | `H01-I03` | 不改变 hypothesis 的实现版本 |
| Dev trial | `H01-T007` | 消耗 development feedback budget 的运行 |
| Confirmation | `H01-CONF01` | 完全冻结的独立确认 |

状态机：

```text
R0 source audit
  -> source observability
  -> Motivation Brief
  -> model-family adequacy
  -> normally tuned strong full-token baseline
  -> Fxx failure reproduced
  -> Hxx architecture consequence
  -> Ixx mechanics pass
  -> Txxx learnability/utility tuning
  -> specificity and attribution
  -> CONF lock
  -> independent confirmation
  -> one-shot test
```

不允许从 `R0` 直接跳到 `Hxx`，也不允许从 mechanical activity 直接跳到
confirmation。

---

## 4. R0：先建立值得解决的问题和真正的强模型

### R0-A. Scope and data-object audit

对 KuaiSearch、Amazon 和 JDsearch 分别登记：

- query/history/candidate 中实际可用的信息对象；
- plaintext、item ID、timestamp、historical slate、repeat rate、text coverage；
- 可以共同支持的 claim 和只能单数据集支持的 claim；
- development、confirmation、test 的请求数、用户隔离和功效。

若 JDsearch 没有支持 token-semantic mechanism 所需的信息对象，它只能支持另一个
明确分离的 robustness claim，不能作为同一 semantic primitive 的强制 gate。

C80 的 365-request fresh role 继续封存，不用于 C80、C80 rescue 或本 pipeline 的
primary confirmation。新的 architecture project 必须准备独立且经过 power analysis
的 confirmation holdout。

### R0-B. Full-token observability parity

在 KuaiSearch 主轨和至少一个可比副轨运行普通 full-token joint Transformer，固定
报告：

- query/candidate `null-history`；
- true history；
- freshness/length-matched wrong-user history；
- shuffled history，作为 report-only，除非新 claim 明确依赖 order；
- repeat-present、strict non-repeat、no-history 三个 surface；
- pooled interface 和 ordinary target-attention control。

该阶段只回答信号在哪里，不提出 architecture novelty。

历史结果只有在信息对象、输入接口、模型角色、训练目标和报告 surface 足够可比时，
才可替代本轮副轨运行；否则必须写 equivalence card 或明确标成 background evidence，
不能用“以前有过一个正数”宣布 parity 完成。

### R0-M. Motivation Brief and problem-value gate

在继续大规模 baseline tuning 或 Failure Atlas 前，先写一页以内、允许被证伪的
Motivation Brief。它描述问题，不命名新 primitive，至少包含：

```text
User/ranking problem: <现实请求中发生了什么错误>
Prevalence and severity: <覆盖率、paired loss、CI 和对整体指标的贡献>
Strong methods expected to fail: <至少两个不同模型家族及原因>
Transformer asset to preserve: <已经由 matched control 证明的能力>
Suspected native shortfall: <ordinary Transformer 可能失败的位置；尚属假设>
Recoverable paper-level payoff: <若修复，整体或预先声明主 surface 能回收多少>
Nearest prior/simple answer: <已有机制、更多容量或更好训练是否已能解释>
Cheapest kill test: <最便宜的反证>
```

Motivation 不是修辞 gate。必须有定量的 prevalence × severity、至少一个真实请求级
例子族以及相对最强相关 baseline 的差距。默认要求潜在整体收益达到 `doc/11 §1.4`
的最小可主张效应；若只主张一个重要 surface，必须在 outcome 前说明该 surface 的
实际意义、规模和为什么它可成为 primary claim，不能靠事后切片放大。

若还不能填写，下一步应是 measurement/data analysis，而不是更多 Transformer
微结构。若 cheapest kill test 失败，关闭这份 brief，换一个高层问题；不能把阈值、
切片或模块改名后继续。

### R0-C. Strong-baseline development

普通 full-token Transformer 必须获得与 trainable baseline 相称的正常调参权。预算按
搜索维度、单次成本和对照预算在 outcome 前声明：小型固定 recipe 通常 4--8 个
configuration trials，多轴 trainable search 默认上限 16 个。16 是 ceiling，不是必须
用完的目标；另行登记每个 configuration 所需的 evaluator invocations。更高预算需要
pre-outcome amendment，并给关键 control 对称预算。

mechanical retry 不消耗 evaluator call；任何改变分数语义、训练数据、优化、容量、
tokenization 或 checkpoint selection 的修改都消耗 trial。

R0-C 分成两个顺序固定的粒度，禁止先在弱 model family 上反复微调：

1. **R0-C0 model-family adequacy**：先比较 ranking-pretrained cross-encoder、当前
   sentence-encoder-plus-random-head recipe、同 backbone query-candidate base，以及
   最强 eligible static/upstream baseline。先回答 pretraining role、scoring interface、
   supervision/objective 和 base preservation 是否正确。
2. **R0-C1 within-family tuning**：只有一个 model family 通过 C0 后，才搜索 learning
   rate、epoch、token budget、dropout、history length 等局部轴。

“normally tuned”不自动等于“strong”。进入 Failure Atlas 还必须同时满足：

- null/no-history 与同输入、同 backbone 的 query-candidate base 在预先冻结的
  non-inferiority margin 内；
- overall true-history 结果至少对最强 eligible baseline 有竞争力；若显著落后，必须
  证明差距不是 backbone、ranking pretraining、objective、capacity 或 checkpoint
  selection 造成；
- history observability 在选定 family 上仍成立；
- 至少包含一个为 ranking/cross-encoding 训练的强模型，不能只用通用 sentence
  encoder 加随机线性头代表 ordinary Transformer 上界；
- 若多个配置之差低于 MDE 或 CI 跨零，按 base preservation、简单性和成本选择，
  不以最高点估计制造 winner。

没有通过这些条件时登记 `BASELINE_INADEQUATE` 并停留在 R0-C；此时看到的
no-history degradation、低分 slice 或弱 attention 只能指导 baseline repair，不能写成
“Transformer 架构缺陷”。

预算同时记录 **configuration trials** 与 **evaluator invocations**。一个配置的
true/null/wrong/shuffle 四个 surface 不是“四次模型搜索”，也不能反过来让 16-call
ceiling 实际只剩四个配置。调参期只打开做当前决策必需的 primary/base-preservation
surface；wrong/shuffle 等 claim-specific 或 report-only control 在配置冻结后运行一次，
除非它正是当前 Motivation Brief 的 binding outcome。多 seed 稳定性在 family 与
motivation 都存活后执行，避免在错误 model family 上提前消耗 Tier-2 预算。

### R0-D. Failure atlas

只在调好的 strong baseline 上调查 failure。允许调查但不得预设成立的例子：

- irrelevant/wrong history 是否稳定伤害 null base；
- relevant event 是否因 token budget/truncation 丢失；
- exact recurrence 是否压制 strict-non-repeat transfer；
- 某类 Q--H--C edge 是否在特定层或特定请求面失效；
- candidate-independent scoring 是否缺失必要的 list context；
- objective 是否稳定学习到 popularity、position 或 identity shortcut。

每轮最多保留 3 个 active failure idea，并按 ranking impact、可构造性和 falsifier
成本排序；只实现前 2 个 cheap probe。每个 idea 必须从同一份 R0-M Motivation Brief
推出，并在 probe 前补齐以下 problem-value fields：

- prevalence、paired severity 和可恢复的 overall contribution；
- ordinary Transformer 与至少一个最强相关非 Transformer/static/upstream family 的
  shared blind spot；
- 已建立且必须保留的 Transformer asset；
- 最强简单修复、nearest existing method 和 cheapest falsifier；
- 若成功，论文 intro 中能写出的 one-sentence claim。

优先探索高层行为失败（例如 evidence conflict、long-range dependency、
candidate-conditioned preference、shortcut learning），再用 layer、edge、token 或
objective probe 定位。char-overlap 阈值、很小 cohort、attention mass 或某个细小结构
只能作为 localization，不得单独充当 paper motivation。没有 shared blind spot 或
recoverable payoff 的 idea 不进入 top 3。

probe 前使用五字段 R0 iteration record，不写完整 Failure Card。一个 idea 失败后先
关闭或 park，再补位，禁止并行扩展 failure tree。若 atlas 暴露的是 base degradation，
立即返回 R0-C0，不得将其升级为 Fxx。

slice mining 只能用于提出 `Fxx`，不能直接成为论文 claim。`Fxx` 必须在另一个时间
split、用户 split 或可比 dataset 上复现后才能进入 architecture formulation。

---

## 5. Failure Card：架构准入门

只有准备申请 architecture entry 的 survivor `Fxx` 才提交完整 Failure Card。内容应
以数字、artifact link 和短句为主，不为填满模板补写推测：

```text
Failure ID:
Motivation Brief and one-sentence paper problem:
Strong baseline and tuning budget:
Affected request surface:
Prevalence, severity, and recoverable overall contribution:
Cross-family failure matrix:
Transformer asset that must be preserved:
Counterfactual/intervention:
Observed paired effect and confidence interval:
Replication split or dataset:
Simpler repairs already tested:
Localized representation/attention/objective locus:
Nearest prior method and reviewer alternative:
Minimum claimable effect and power:
Cheapest falsifier:
Claim boundary if repaired:
```

进入 `Hxx` 的必要条件：

1. failure 出现在强且正常调优的 ordinary full-token baseline 上；
2. failure 有 ranking utility 后果，不只是 norm、attention mass 或 rank activity；
3. 至少在两个独立 split，或主轨与可比副轨上方向一致；
4. effect 达到基于 development variance 预先计算的 MDE；
5. capacity、context length、普通 optimization、pooling removal、ordinary attention
   等简单修复不能解释；
6. intervention 将问题定位到一个具体 locus；
7. 能写出一个低成本、结果前冻结的 falsifier。
8. 同一 failure 至少覆盖 ordinary Transformer 与一个强相关替代 family；否则必须把
   贡献明确降级为某个模型 family 的修复，而不能声称现有方法共同遗漏；
9. 已量化 prevalence × severity 与可恢复 overall payoff，并达到 Motivation Brief
   预先声明的论文价值门槛；
10. 已完成轻量 nearest-work audit，说明贡献不是最近方法、更多容量或普通
    ranking pretraining 的直接实例；
11. repair 必须保留 R0-B 已建立的 Transformer asset，而不是用 repeat shortcut、
    fixed-score fallback 或 slice router 换取局部增益。

Failure Card 不通过时，允许继续 baseline/measurement study，不允许发明 primitive。
在 Failure Card 前允许 §2.1 的 disposable prototype；通过后必须在正式源码树中重新
实现，不能直接把 `tmp/` 原型提升为系统。

Failure Card 评审前先做一次 reviewer-shaped pre-mortem，只回答四个问题：为什么该
问题重要、为什么最强已有方法没解决、为什么 ordinary Transformer 已有资产仍不够、
为什么预期修改能同时改善 targeted surface 与整体指标。任一答案只能靠“模型更复杂”
或“Transformer 表达能力强”时，退回 R0-M/R0-C，不进入 Hxx。

---

## 6. Hxx：从 failure 推出一个架构假设

架构提案必须使用：

```text
Observed failure: <Fxx 的可复现事实>
Architecture consequence: <一个具体 primitive>
Cheapest falsifier: <失败时关闭 Hxx 的实验>
Nearest simpler repair: <最强普通修复/已有机制>
Expected failure boundary: <在哪些 surface 不应有增益>
```

约束：

- Transformer/LM 必须是 end-to-end ranking core；
- 一个 hypothesis 只有一个 load-bearing primitive，组件必须通过 ablation 交租；
- 不得使用 dataset-name branch；只允许按 evidence availability mask 条件化；
- nearest-neighbor 和 matched-capacity control 在实现前登记；
- novelty 不得只由名称、数学记号或不可归约性声明；
- 不能因为项目要求“必须新架构”而在没有 `Fxx` 时开工。

架构可以联合修改 tokenization、attention、objective 和 base preservation，但必须在
proposal 中说明为什么这些是同一个 failure 的共同修复，而不是模块堆叠。

---

## 7. 开发期分层验证

### M0. Mechanics

在不读取 evaluation labels 的情况下验证：

- finite forward/backward、tiny-batch overfit、gradient reachability；
- candidate hash、candidate permutation 和 deterministic rescore；
- no-history fallback，仅当 hypothesis 声明该性质；
- padding、truncation、empty history、all-no-history batch；
- fp32/bf16 的 reference error 和 rank/top-k stability。

`MECHANICAL_FAIL` 只说明实现不满足规格，不说明 hypothesis utility。允许在同一个
`Hxx` 下创建新的 `Ixx` 修复。修复一旦改变数学语义或 score behavior，就必须创建
新的 dev trial 并计入预算。

### M1. Learnability

检查 tiny-set overfit、train/internal loss、gradient、训练方差和 checkpoint stability。
该阶段区分 `NO_LEARN` 与 `BAD_IMPLEMENTATION`，不把任意 loss-window 阈值当作论文
utility 结论。

### M2. Development utility

在固定 development surface 上比较 strong baseline。早期允许 single seed；选择
冻结后至少 3 seeds。primary inference 使用请求/用户级 paired 统计，不使用 best seed。

fold 是异质性诊断，不默认作为“每 fold 全过”的联合否决条件。只有 claim 明确要求
每个 cohort 都成立时，fold 才是 binding gate。

### M3. Specificity

control 必须与 claim 对齐：

- wrong-user 验证 user provenance，不等于纯 null；
- query mask/wrong query 只约束 query-conditioned claim；
- shuffle 只约束 order-sensitive claim；
- repeat/no-repeat 分开验证 recurrence 与 transfer；
- no-history exactness 只约束 base-preserving claim。

`UTILITY_NO_SPECIFICITY` 可以保留“更强 encoder/representation”结果，但不能包装成
personalized architecture。

### M4. Attribution and rent

幸存模型必须比较：

- ordinary full-token Transformer；
- nearest existing mechanism；
- matched-capacity/compute backbone；
- primitive degeneration；
- 必需组件 ablation。

`UTILITY_NO_RENT` 降级为 interface、objective 或 baseline improvement，不进入新
architecture claim。

### M5. Confirmation readiness

只有 utility、specificity、attribution 都通过，并完成 confirmation power analysis，
才可创建 `CONF` lock。

---

## 8. Numerical and invariance contracts

数值安全与科学 utility 分开。

新的 numerical contract 必须同时报告：

- fp32 或 fp64 reference；
- 与实际 dtype 对应的预注册 `atol/rtol`；
- per-request max/quantile score error；
- candidate rank、top-k membership 和 near-tie margin stability；
- deterministic repeat 的环境和 kernel 条件。

不再用与 dtype、序列长度、reduction order 和 ranking margin 无关的统一绝对常数
裁决所有模型。若数学 claim 是 exact invariance，则实现必须给出 reference/property
test；若论文只需要 ranking invariance，则以预注册的 rank/top-k contract 为 binding。

event permutation 不是通用 gate。只有 `Hxx` 明确声称 event-set invariance 时才是
binding；否则只报告。candidate permutation 在 candidate-wise/set ranker 中仍是通用
mechanical contract。

---

## 9. Feedback classification

R0 probe 先使用以下状态；这些状态不能伪装成 Hxx utility verdict：

| R0 状态 | 含义 | 后续动作 |
|---|---|---|
| `INSTRUMENTATION_FAIL` | 数据/control/运行机械不可信 | 修工程，不累计 scientific progress |
| `MOTIVATION_WEAK` | prevalence、severity、shared blind spot 或 payoff 不足 | 关闭/重写 Motivation Brief，不做微结构 |
| `BASELINE_INADEQUATE` | model family/base preservation/竞争力未过 | 返回 R0-C0/C1，不建 Failure Card |
| `FAILURE_IDEA_FALSIFIED` | cheapest probe 否证一个合格 idea | 关闭 idea；只在同一 brief 尚有高价值预测时补位 |
| `FAILURE_REPLICATED` | ranking failure 独立复现且简单修复未解释 | 提交 Failure Card review |
| `ROUND_ADVANCE` | contribution ladder 前进一级 | 关闭本 round，登记下一 scientific question |

Hxx 获准后，每次 trial 结束必须且只能登记一个主状态：

| 状态 | 含义 | 后续动作 |
|---|---|---|
| `MECHANICAL_FAIL` | 实现/数值不满足规格 | 修 `Ixx`，不产生 utility 结论 |
| `NO_LEARN` | 固定实现未学到目标 | 在冻结 search space 内优化，预算耗尽后关闭 |
| `NO_UTILITY` | strong baseline 上无 dev value | 关闭 `Hxx` |
| `UTILITY_NO_SPECIFICITY` | 有 utility，但 provenance claim 不成立 | 收缩为 generic improvement |
| `UTILITY_NO_RENT` | 简单 control 同样有效 | 降级为 interface/baseline result |
| `DEV_SURVIVOR` | dev utility/specificity/rent 均成立 | 冻结 `CONF` |
| `CONFIRM_FAIL` | 独立确认失败 | 关闭 claim，不 rescue cohort |
| `CONFIRMED` | 独立确认通过 | 完成 Tier-2 和 one-shot test |

不允许把 `MECHANICAL_FAIL` 写成负 utility，也不允许把 `NO_LEARN` 写成 primitive 被
数据否证。反过来，工程修复也不能自动恢复已经失败的 confirmation claim。

---

## 10. Trial manifest and required artifacts

R0 日常 probe 只使用
`experiments/problem_discovery/_r0_iteration_template.yaml` 的五个字段。Hxx 获准后，
每次 score-affecting development trial 的 config/metadata 至少包含：

```yaml
research_phase: R0-BASE
failure_id: F01
hypothesis_id: H01
implementation_id: H01-I03
trial_id: H01-T007
change_class: optimization
expected_observation: null
allowed_diagnostics: []
dev_call_index: 7
candidate_manifest_sha256: null
stop_condition: null
```

`expected_observation`、`candidate_manifest_sha256` 和 `stop_condition` 运行前必须为
具体值；上例的 `null` 只是 schema 占位。

Tracked artifacts：

```text
experiments/problem_discovery/<R0Mxx>_motivation_brief.md
experiments/problem_discovery/<Fxx>_failure_card.md
experiments/problem_discovery/<Hxx>_proposal.md
experiments/problem_discovery/<Hxx>_trial_budget.yaml
reports/<Hxx>_dev_summary.json
reports/<Hxx>_confirmation.json
doc/dev_log/<date>_<id>_<decision>.md
```

Raw checkpoints、scores、logs 和 sweeps 仍只放在 `models/`、`runs/`、`artifacts/`。

---

## 11. Developer run loop

每个开发者按以下顺序工作：

1. 读取当前 R0 iteration 或 `Fxx`、strong baseline、剩余 budget 和允许的 change class；
2. 先完成 M0 property tests，不能用 evaluation labels 调 mechanics；
3. 如果下一步已经明确，直接创建 run ID/manifest 并执行，不再生成额外 plan；
4. 训练并只导出统一 `scores.jsonl`；
5. 持公共锁调用共享 evaluator；
6. 将结果分类为 §9 的一个状态；
7. 只根据该状态执行允许的下一步；
8. 更新 budget ledger 和 concise dev log；
9. budget 用尽、`NO_UTILITY` 或 `UTILITY_NO_RENT` 后停止，不创建机械 successor；
10. `DEV_SURVIVOR` 交给 confirmation owner 冻结，原开发者不能根据 confirmation
    outcome 修改同一 candidate。

可以并行运行独立 trial，但不能并行调用 evaluator，不能共享 run directory，也不能
通过 sibling outcome 临时改变自己已经开始的 trial。并行是资源调度，不再承担制造
机制多样性的功能。

---

## 12. Round boundary and next execution order

R0 Round 1 已结束并由用户暂停。它完成了 scope audit、KuaiSearch observability、一个
有限 ordinary full-token family 的 tuning/stability 和 Failure Atlas；两个 idea 均被
证伪，且暴露 `BASELINE_INADEQUATE`。这不授权自动进入 Round 2。

用户显式恢复后，下一 scientific round 只能按修订后的顺序执行：

1. 复核 `R0-A/B` 的跨轨 equivalence，不能把不可比历史结果当 parity；
2. 建立 `R0-M` Motivation Brief，先冻结高层问题、实践影响、Transformer asset 和
   kill test；
3. 进入 `R0-C0`，优先修 model family 与 no-history base degradation；
4. 通过 adequacy 后才进入 `R0-C1` 局部调参和必要稳定性；
5. `R0-D` 只 probe 来自 Motivation Brief 且具有 shared blind spot/overall payoff 的
   前两个 idea；
6. 形成第一个合格 `Fxx`，或以 contribution ladder 判定转为 measurement/negative-
   design paper、换题或停止。

在上述步骤完成、Failure Card 通过并登记预算前，**不授权新的 architecture source
tree、GPU architecture training 或 confirmation label opening**。仅允许 §2.1 定义的
CPU/tiny-data disposable prototype。
