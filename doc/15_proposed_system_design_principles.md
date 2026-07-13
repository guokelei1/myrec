# 15 - Proposed-System Design Principles

状态：**当前架构准入与设计约束。** 研发状态机、trial 反馈和 confirmation 规则以
`31_problem_discovery_and_architecture_iteration_protocol.md` 为准；continuous
orchestration 和 whole-pipeline end states 以
`32_autonomous_pipeline_controller.md` 为准。

C01--C80 已终止；没有 C81。旧候选的 formulation、gate amendment 和逐候选结论
保留在 `systems/README.md`、candidate-local notes、reports 和
`dev_log/20260712_c01_c80_terminal_retrospective.md`，不再堆叠在本文，也不构成当前
执行授权。

---

## 1. 当前证据允许说什么

已建立：

- KuaiSearch 被测历史增益主要集中在 exact recurrence；coarse category transfer
  未建立，undifferentiated mixing 会稀释 item-only。
- pooled representation 会隐藏重要的 token-level history signal。
- Amazon ordinary full-token joint Transformer 达到
  `true-null +0.025298`、`true-wrong +0.035944`，三个 seed 均为正。
- KuaiSearch R0 Round 1 ordinary full-token family 达到三 seed
  `true-null +0.020936`、`true-wrong +0.031812`，证明 joint token contextualization
  在主轨也能读到 history；strict-non-repeat direction 均值为 `+0.005388`。
- Amazon 的双向 Q--H、C--H 和 history-read-context edge 承重。
- event order 当前不是已建立的承重因素。

未建立：

- ordinary full-token Transformer 有一个必须由新架构修复的缺陷；
- candidate-conditioned evidence fidelity 已指定某个 concrete primitive；
- 一个 token-semantic primitive 能自然覆盖无 plaintext 的 JDsearch；
- 任一 C01--C80 方法有未知标签上的正或负 utility；
- proposed architecture 已经 ready。

Round 1 还明确显示：当前 KuaiSearch full-token recipe 不能称为 strong base。其 true
NDCG@10 三 seed 均值 `0.331028`，低于 item-only `0.345376`；no-history 相对静态 base
为 `-0.013066`，replication CI 仍全负。两个预注册 Failure Atlas idea 也均被证伪。
因此下一步若恢复，应先修 model-family/base adequacy 和 motivation，不是开始微结构。

因此“history evidence fidelity”只能作为宽问题背景，不能直接授权 gate、memory、
transport、routing、fast weight 或其他算子。

---

## 2. 架构设计前必须有 Failure Card

架构只能从 `doc/31 §5` 已通过的 `Fxx` 推出。必要链条是：

```text
normally tuned ordinary full-token baseline
  -> replicated ranking-relevant failure
  -> simple repairs ruled out
  -> model locus localized
  -> one architecture consequence
  -> one cheap falsifier
```

强制模板：

```text
Observed failure: <Fxx 中已经复现的事实>
Architecture consequence: therefore use <one primitive> at <one locus>.
Falsification: if <cheap matched control>, close Hxx.
Claim boundary: this can support <claim>, but not <stronger claim>.
```

如果无法填写，不得创建新的 `systems/<hypothesis>/` 目录或运行 architecture GPU
training。允许在 `tmp/r0_prototypes/` 做一次性 CPU/tiny-data localization probe，
但不得读取 evaluation labels、调用 evaluator 或产生 architecture claim。

---

## 3. 合格的 proposed architecture

未来 proposed system 必须：

1. 是 LLM4Rec-style Transformer/LM ranker；
2. 由 Transformer/LM 端到端产生 ranking logits；
3. 在内部联合建模 query、strictly-prior history 和 candidate；
4. 用一个 load-bearing primitive 修复一个已确认 failure；
5. 使用统一 record/mask interface，不含 dataset-name branch；
6. 在 no-history、text-missing 等 evidence absence 情况按声明退化；
7. 与 ordinary full-token、nearest existing mechanism 和 matched-capacity backbone
   比较；
8. 通过 primitive degeneration 和必要组件 ablation 支付 unique rent；
9. 记录训练、推理、显存和延迟成本。

以下不构成合格 architecture contribution：

- prompt-only LLM scorer；
- frozen/offline LLM embedding 后接 MLP；
- 对固定 score channel 做 router 或 mixture；
- DIN/ZAM/TEM/ordinary target attention 的重命名；
- 只有 training trick、loss weight、feature engineering 或超参变化；
- 为通过某个 gate 添加 per-dataset/per-slice fallback；
- 形式不可归约但没有 ranking utility 的数学 operator。

interface、objective 或 optimization 改进可以成为诚实贡献，但应按实际贡献命名，
不能为了“架构创新”强行包装。

### 3.1 Transformer 必要性与架构闭环

未来架构的论文叙事必须由同一组实验事实闭合，不能只以“Transformer 表达能力更强”
作为动机：

1. **Baseline blind spot**：在有代表性的强非 Transformer baseline、强静态方法和
   normally tuned ordinary full-token Transformer 上，复现同一个 ranking-relevant
   failure；不要求枚举所有已有模型，但必须覆盖最接近、最有竞争力的替代解释。
2. **Transformer asset**：证明 query--history--candidate 的 token/context/sequence
   联合建模能力对排名承重，并相对 pooled、fixed-score 或非上下文化替代项产生可归因
   的收益。具体是顺序、集合交互、长程依赖还是 candidate-conditioned evidence，必须由
   matched control 决定，不能预设 event order 必然重要。
3. **Native Transformer shortfall**：failure 在普通 Transformer 正常调优后仍存在，
   并被定位到明确的 representation、attention、tokenization、objective 或
   optimization locus；否则优先修 baseline，不发明 primitive。
4. **Architecture repair**：一个 load-bearing modification 直接修复该 locus，同时保留
   普通 Transformer 已建立的 query relevance、history utility、strict-nonrepeat utility
   和 evidence-absence 行为；不允许用固定通道、slice fallback 或重复商品捷径替代
   Transformer 的联合建模能力。
5. **End-to-end payoff**：survivor 必须同时在 failure surface 上通过机制检验，并在统一
   primary ranking metric 上超过 ordinary full-token Transformer 和最强 eligible
   baseline；只修 slice、不改善整体指标，或只改善整体而不能归因，均不足以支撑完整
   architecture claim。

对应的最小论证链为：

```text
strong baselines share a replicated blind spot
  -> ordinary Transformer exposes a localized native shortfall
  -> one modification repairs that shortfall
  -> established Transformer advantages are preserved
  -> targeted mechanism gain and overall ranking gain both survive confirmation
```

这是一条 architecture-entry 原则，不是后验选故事的许可。blind spot、asset 和
shortfall 都必须在 outcome 前冻结的比较与 Failure Card 中可证伪。

### 3.2 研究问题与搜索粒度

“更注重 motivation”不等于写更宏大的背景，而是先量化一个高层、真实且可证伪的
ranking problem：发生频率、单请求损失、整体可恢复贡献、哪些强方法共同失败，以及
为什么该问题需要保留 Transformer 已验证的 joint-context asset。

搜索顺序固定为：

```text
high-level user/ranking failure
  -> cross-family shared blind spot
  -> adequate ordinary-Transformer family
  -> native shortfall localization
  -> one small structural repair
```

因此不在起点枚举 attention、gate、memory、token threshold 或细小 slice。小结构是
最后一级的因果修复，不是寻找论文问题的工具。相反，也不能停留在“序列理解更强”这类
high-level 口号；每个 high-level statement 必须落到 matched intervention、effect size
和 kill test。

模型搜索也遵循由粗到细：先判断 ranking pretraining、cross-encoder interface、
objective、base preservation 和 list context 等 model-family 选择；family 合格后才调
token length、learning rate、dropout 等局部参数；Failure Card 通过后才动新结构。若
point estimate 差异低于 MDE，选择更简单且 base-preserving 的版本，不追逐噪声 winner。

---

## 4. 一个 hypothesis，不等于一个固定 implementation

科学 hypothesis、实现、调参和 confirmation 使用不同 ID：

- `Hxx`：一个 primitive；
- `Hxx-Iyy`：mechanical implementation revision；
- `Hxx-Tzzz`：消耗 dev feedback budget 的 trial；
- `Hxx-CONFyy`：冻结 confirmation。

允许在不改变数学语义且未读取新 outcome 的前提下修复 `Iyy`。一旦修改训练数据、
tokenization、capacity、objective、optimization、score behavior 或 checkpoint selection，
就必须创建并登记新的 `Tzzz`。

`MECHANICAL_FAIL`、`NO_LEARN`、`NO_UTILITY`、`UTILITY_NO_SPECIFICITY` 和
`UTILITY_NO_RENT` 是不同结论，不得统一写成“candidate closed，因此 primitive
无效”。

---

## 5. Gate 分阶段，不再一次合取

### 5.1 Day-one evidence hygiene

始终 binding：

- label-free dev/test records；
- scoring/training 永不读取 qrels；
- candidate manifest hash；
- 共享 evaluator 和 dev-eval log；
- fixed development split；
- test lock；
- run/config/checkpoint/data provenance。

### 5.2 Mechanics

在 evaluation labels 前验证 finite、shape、padding、empty history、determinism、
candidate permutation、reference precision 和 hypothesis 声明的 fallback/invariance。

mechanics 失败只返回 implementation repair，不产生 utility 结论。

### 5.3 Learnability and normal tuning

允许在 `doc/13 §2.5` 的对称预算内调 learning rate、epoch、capacity、dropout、
history length、objective 和 initialization。tiny-set overfit、loss、gradient 和
checkpoint stability 是诊断，不是 paper utility。

### 5.4 Development utility

先在固定 dev 上相对 normally tuned ordinary full-token base 检查 primary metric。
早期 single seed 可标记 provisional；冻结配置后运行至少 3 seeds，不报告 best seed。

### 5.5 Specificity

control 必须匹配 claim：

| Claim | Binding control |
|---|---|
| user provenance | true vs matched wrong-user history |
| query conditioning | true vs wrong/masked query |
| order-sensitive history | true vs shuffled event order |
| recurrence preservation | repeat-present surface |
| transferable personalization | strict non-repeat surface |
| base preservation | no-history fallback |
| set/list interaction | candidate permutation and edge ablation |

wrong-user 不是纯 null；shuffle 不是通用 corruption；no-history exactness 也不是每个
模型的通用要求。

### 5.6 Attribution and novelty

utility survivor 才支付 mechanism rent：nearest neighbor、matched capacity/compute、
primitive degeneration 和 component ablation。简单 control 追平时，结果降级为
interface/baseline improvement，不继续添加模块追 rent。

### 5.7 Confirmation

independent confirmation 使用预先冻结的 primary comparison、MDE、统计方法、seeds、
controls 和 numerical contract。失败不 rescue cohort；通过后才进行 Tier-2 完整审计
和 one-shot test。

---

## 6. Numerical contract

数值阈值必须与 dtype、sequence length、reduction order 和 ranking risk 对齐，报告：

- fp32/fp64 reference；
- dtype-aware `atol/rtol`；
- score error distribution；
- rank/top-k stability；
- near-tie margins；
- deterministic environment/kernel conditions。

只有论文 claim 要求 exact algebraic invariance 时，exact property 才是 binding。
event permutation 仅对 event-set/order-invariance claim binding；不能再作为所有
history model 的共同生死门。

---

## 7. Dataset and claim scope

统一接口不等于三个数据集必须支持同一个信息对象。

- KuaiSearch：主轨，full-token observability 已建立，但 ordinary model family 的
  strong-base adequacy 未通过。
- Amazon：full-token semantic history source 已确认；与 KuaiSearch 的 parity 仍需
  interface/model-role equivalence audit，不能只比较两个历史正数。
- JDsearch：没有 plaintext 时不能支持 token-semantic claim，只能在 mechanism 输入
  条件满足时作为 conditional robustness anchor。

如果一个 claim 只在 Amazon 可构造，应缩窄 claim 或更换主轨，不能用 per-dataset
branch 假装统一。跨域确认前必须先通过 `doc/31 R0-A` 信息对象审计。

---

## 8. 当前授权

只授权：

1. 三轨 scope/data-object/holdout/power audit；
2. KuaiSearch 和 Amazon full-token observability parity；
3. 一页 R0-M Motivation Brief 与 problem-value/nearest-work audit；
4. ranking-pretrained cross-encoder、same-backbone query-candidate base 和现有 recipe 的
   model-family adequacy 比较；
5. adequacy 通过后的 ordinary full-token normal tuning；
6. 基于合格 strong baseline 和同一 Motivation Brief 的 failure atlas；
7. 每轮至多 3 个 active failure idea、其中前 2 个 cheap probe；
8. `tmp/r0_prototypes/` 下的 CPU/tiny-data disposable prototype；
9. survivor Failure Card formulation 和 review。

在首个 Failure Card 通过前，不授权新的 architecture source tree、architecture GPU
training、fresh confirmation label opening 或 test。C80 的 365 fresh labels继续封存；
没有 C81。以上是科学范围，不解除当前 `PAUSED/USER_REQUESTED_PAUSE`；只有用户显式
恢复后才可执行。
