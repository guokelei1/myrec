# 32 - Autonomous Development Pipeline Controller

状态：**当前权威 autonomous orchestration 协议。**

本文规定开发 AI 如何读取状态、连续执行 `doc/31`、根据反馈自动转换、恢复中断，并在
整个流水线真正完成或需要外部输入时结束。`doc/31` 仍是科学研发阶段、Failure Card、
Hxx validation 和 confirmation 的权威；本文不重复定义模型或科学 gate。

---

## 1. Controller objective

开发 AI 必须持续运行：

```text
R0 problem discovery
  -> Failure Card
  -> Hxx architecture development
  -> independent confirmation
  -> terminal or resumable pause
```

单个实验完成、单个状态转换或单个 Hxx 失败都不是 controller 的结束条件。每次结果
产生后，controller 必须持久化状态并自动选择下一项合法动作，直到 §9 的三个顶层状态
之一成立。

Controller 优化的是 `doc/31 §1.1` contribution ladder 的前进，不是 GPU 利用率、run
数量或候选数量。工程修复可以连续执行，但不能被计作新的 scientific round 或论文
进展。

---

## 2. Authority and hard boundaries

读取顺序：

1. `AGENTS.md`；
2. 本文；
3. `doc/31_problem_discovery_and_architecture_iteration_protocol.md`；
4. `doc/15_proposed_system_design_principles.md`；
5. `reports/pps_architecture_readiness.md`；
6. `doc/11`--`doc/13`；
7. 当前 Failure Card、trial budget、confirmation lock 和 run metadata。

始终保持：

- C01--C80 关闭；无 C81、C80 rescue 或 C80 fresh-label opening；
- scoring/training 不读取 dev/test qrels；
- candidate hash、统一 evaluator、dev log 和 test lock；
- confirmation outcome 后不修改 model/config/cohort/threshold；
- test 不由 controller 自动打开；成功状态最多到 `READY_FOR_TEST`；
- 不自动 commit 或 push。

Motivation/diagnostic work 只用于补齐当前 R0/Failure Card 缺失的 premise，不重新打开
M3/M4、C5-R3 或 C01--C80 claim。

---

## 3. Persistent state

状态文件：

```text
experiments/problem_discovery/pipeline_state.yaml
```

不存在时从 `_pipeline_state_template.yaml` 初始化；存在时恢复，禁止从头重跑。

每次 run、feedback classification、budget change 和 state transition 后更新。至少记录：

- `pipeline_status` 和 `reason_code`；
- 当前 R0 round、phase、Fxx、Hxx、Ixx、Txxx 或 CONF ID；
- active/closed failure ideas 和 hypotheses；
- 当前 contribution level、active Motivation Brief 和其 kill criterion；
- strong-baseline adequacy verdict、model family、waterline 与 non-inferiority margin；
- last feedback/evidence/run；
- configuration trials、evaluator invocations、GPU-hours 和各级预算；
- stagnant iteration count；
- next authorized action；
- terminal/pause reason。

若发现未完成 run：

1. 检查 `run.lock`、process、config/checkpoint hash 和现有 outputs；
2. 能合法恢复则恢复；
3. 已完成但未汇总则只完成汇总；
4. 无效 run 按 evidence boundary 登记，不重复消耗 evaluator call；
5. 不创建重复 run ID。

---

## 4. Default global limits

除非 repository 中存在更严格的 pre-outcome lock：

```text
MAX_R0_SCIENTIFIC_ROUNDS = 3
MAX_ACTIVE_FAILURE_IDEAS = 3
MAX_FAILURE_PROBES_PER_R0_ROUND = 2
MAX_MODEL_FAMILY_REPAIRS_PER_R0_ROUND = 2
MAX_BASELINE_CONFIG_TRIALS_PER_FAMILY = 12
MAX_ARCHITECTURE_HYPOTHESES_PER_FAILURE = 3
MAX_MECHANICAL_REVISIONS_PER_HYPOTHESIS = 2
STAGNATION_REVIEW_AFTER = 2 iterations
MAX_TOTAL_DEV_EVALUATOR_CALLS = 64
MAX_TOTAL_GPU_HOURS = 48 A40-hours
```

16 configuration trials 是多轴 trainable method 的通用 ceiling；本 controller 对单个
baseline family 默认使用更紧的 12-trial 上限。evaluator invocations 独立记账，不能因
每个配置有四个 scenario 就把 16-trial search 缩成四个配置；反过来也不能漏记调用。
调参期只打开当前决策必需的 surface，report-only control 默认在冻结配置上运行一次。
关键 control 使用对称或可辩护预算。达到全局资源上限但仍有合法注册实验时只能
`PAUSED`，不能据此产生 scientific no-go。

Scientific round 与 iteration 不同。一个 round 从冻结 Motivation Brief 开始，以
contribution ladder 前进、一个有论文价值的 thesis 被反证，或 `BASELINE_INADEQUATE`
被路由回 family repair 后的正式 round review 结束。donor/control materialization、
YAML、numerical、lock 和 crash recovery 都只记 engineering iteration，不增加 round。

---

## 5. Continuous controller loop

重复执行：

1. 读取权威文档、state、budget、reports、dev log 和未完成 run；
2. 判断当前 phase：R0-A/B/M/C0/C1/D、Failure Card、Hxx-M0--M4 或 CONF；
3. 选择该 phase 下信息价值最高、成本最低、可逆且已授权的下一动作；
4. 创建或恢复 run，完成实现、测试、运行、共享评测和审计；
5. 将 outcome 分类为 `doc/31 §9` 的 feedback state；
6. 更新 run artifacts、iteration record、budget ledger 和 pipeline state；
7. 按 §6 转换到下一状态；
8. 若 §9 未成立，立即继续 loop，不等待用户确认。

R0 使用五字段 `_r0_iteration_template.yaml`。完整 Failure Card 只用于 survivor；Hxx
使用 trial budget；CONF 使用 confirmation lock。额外 tracked 文档只对应新 evidence、
lock 或 decision。

在 R0 每次高成本动作前，controller 还必须问：该动作会验证 Motivation Brief 的哪一
句、可能让 contribution ladder 前进哪一级、失败后关闭什么。三个答案任一为空时，
该动作不是合法 scientific next action。小结构/小切片 probe 只有在 model-family
adequacy 通过且其结果能定位一个已达到价值门槛的 failure 时才有优先级。

---

## 6. Automatic feedback transitions

R0 transitions：

| Feedback | Controller action |
|---|---|
| `INSTRUMENTATION_FAIL` | 修复同一 control/implementation；不增加 round 或 contribution level |
| `MOTIVATION_WEAK` | 关闭 brief；从剩余高层现象中选择下一 brief，不得下钻微结构 |
| `BASELINE_INADEQUATE` | 返回 R0-C0；最多比较两个更合适 model family，再做 C1 局部 tuning |
| `FAILURE_IDEA_FALSIFIED` | 关闭 idea；只有同一 brief 仍有预先登记的高价值预测时才补位 |
| `FAILURE_REPLICATED` | 检查 problem value、shared blind spot、native shortfall 与 nearest work 后提交 Failure Card |
| `ROUND_ADVANCE` | 固化 contribution-level evidence，完成 round review，再选择下一 scientific question |

R0-D 若发现 no-history/base degradation、ranking pretraining 不足或弱 family 造成的
整体落后，必须分类为 `BASELINE_INADEQUATE`，不能把它重新命名为 Fxx。若一个 slice
只击败弱对照、强对照已经解决它，则分类为 `FAILURE_IDEA_FALSIFIED`。

Hxx transitions：

| Feedback | Controller action |
|---|---|
| `MECHANICAL_FAIL` | 同一 Hxx 创建下一 Ixx；不产生 utility 结论；最多 2 次，之后关闭为 implementation unresolved |
| `NO_LEARN` | 在冻结 optimization space 和剩余预算内调整；预算耗尽后关闭 Hxx |
| `NO_UTILITY` | 关闭 Hxx；有剩余配额时提出同一 Fxx 下机制不同的下一个 Hxx |
| `UTILITY_NO_SPECIFICITY` | 收缩 claim；若只剩 generic improvement，则登记并关闭 architecture claim |
| `UTILITY_NO_RENT` | 降级为 interface/objective/baseline result；关闭 Hxx，不追加模块 |
| `DEV_SURVIVOR` | 冻结 CONF lock，进入 independent confirmation |
| `CONFIRM_FAIL` | 进入 `COMPLETED_NO_GO/CONFIRMATION_FAILED`，不 rescue 或返回同一 claim |
| `CONFIRMED` | 完成 Tier-2 audit，进入 `COMPLETED_SUCCESS/CONFIRMED_READY_FOR_TEST` |

一个 Hxx 关闭后，只在 Hxx 配额和全局预算允许时继续下一个 Hxx。不得把多个失败 Hxx
拼装，也不得用局部 gate failure 连续增加模块。

同一 Fxx 的 Hxx 全部关闭时：

1. 检查结果是否形成新的、可复现且改变 failure locus 的 scoped evidence；
2. 有新 evidence 且仍有 R0 scientific round 时，返回新一轮 R0-M；
3. 新名称、新记号、重新解释或新增文档不算新 evidence；
4. 无新 evidence 或 R0 rounds 已用尽时，进入
   `COMPLETED_NO_GO/NO_DEV_SURVIVOR`。

---

## 7. R0 exhaustion and scientific no-go

只有同时满足以下条件，才允许 `NO_ARCHITECTURE_PREMISE`：

- 所有允许的 R0 rounds 已完成；
- 每个 round 都有冻结 Motivation Brief、paper-value verdict 和 contribution-level
  transition；工程 iteration 数不能替代；
- 每轮已注册的 top probes 实际执行或被有效 integrity gate 否决；
- strong baseline 已通过 model-family adequacy、base preservation 和声明的合理 tuning；
- 没有通过的 Failure Card；
- 不是因为数据、GPU、凭证、预算或环境缺失而停止。

仅测试两个由可用 slice 临时产生的 idea、仅证明弱 baseline 失败、或因 ordinary
Transformer 自身 base degradation 而停止，都不足以产生 `NO_ARCHITECTURE_PREMISE`。
这类结果必须先触发 Motivation Brief/model-family reset，或诚实转为 scoped
measurement finding。

只有所有获准 Hxx 实际执行并关闭，才允许 `NO_DEV_SURVIVOR`。短期负结果、两个连续
失败或一个 implementation blocker 均不能直接产生 scientific no-go。

---

## 8. Stagnation recovery

连续 2 个 iteration 没有新 evidence 时触发 recovery，而不是终止：

1. 关闭重复、等价或只改名称的分支；
2. 回到最近有证据支持的状态；
3. 检查剩余已注册 probe/Hxx；
4. 执行尚未测试的最高优先级合法动作；
5. 有合法动作但预算/资源不足时进入 `PAUSED`；
6. 只有 §7 的阶段完成条件满足时才能 `COMPLETED_NO_GO`。

新 evidence 指独立数据结果、新 intervention、可复现 ranking failure 或 failure locus
的实质改变。单纯 proposal、数学改名、阈值/精度/canonicalization 调整不算。

同样，loss 下降、seed 重复、control materialization 通过或更多 slice 表格只在它们
改变 adequacy/problem-value 判断时才算 ladder progress。连续两次局部超参变化都未
超过 noise/MDE 时，必须回到 model-family 或 Motivation Brief 层级，不得继续同粒度
搜索。

---

## 9. Three top-level end states

### 9.1 `COMPLETED_SUCCESS`

唯一 reason：`CONFIRMED_READY_FOR_TEST`。

要求：independent confirmation 通过、Tier-2 audit 完成、method/config 冻结、test 未
打开。下一步只能由用户显式授权 test。

### 9.2 `COMPLETED_NO_GO`

允许 reason：

- `NO_ARCHITECTURE_PREMISE`：§7 的 R0 完成条件满足，无 Failure Card；
- `NO_DEV_SURVIVOR`：Failure Card 通过，但所有获准 Hxx 完成并关闭；
- `CONFIRMATION_FAILED`：唯一 dev survivor 未通过 independent confirmation。

这是科学结论，不能由预算耗尽、资源缺失或简单停滞触发。

### 9.3 `PAUSED`

允许 reason：

- `BUDGET_REAUTHORIZATION_REQUIRED`；
- `DATA_OR_RESOURCE_BLOCKED`；
- `INTEGRITY_REVIEW_REQUIRED`；
- `USER_REQUESTED_PAUSE`。

`PAUSED` 不产生科学结论。保留完整 state；用户解除唯一 blocker 后从原 phase 恢复。
用户要求在 round boundary 暂停时，不得预先创建下一 round 的 lock、source、trial 或
GPU run；可以只做只读复盘和经用户请求的协议修订。

---

## 10. Conditions that do not end the pipeline

- 一个普通实验或 state transition 完成；
- 一个 R0 probe 或一个 Hxx 失败；
- 连续两个负结果；
- 当前结果不理想但仍有注册动作和预算；
- 只完成分析、计划或 proposal；
- stagnation recovery 尚有合法动作。

---

## 11. Final or pause report

进入三个顶层状态之一时，持久化并汇报：

- `pipeline_status`、`reason_code` 和具体证据；
- 初始/最终状态与完整 transition sequence；
- run IDs、Failure Card 和所有 Hxx verdict；
- 最强合法 baseline、survivor 或最终负结论；
- contribution ladder 的初始/最终 level、Motivation Brief verdict 和仍缺的论文链条；
- dev calls、GPU-hours 和剩余预算；
- confirmation/test lock 状态；
- tests、integrity audits 和修改文件；
- supported/unsupported paper claims；
- 若 `PAUSED`，解除 blocker 所需的唯一具体动作。
