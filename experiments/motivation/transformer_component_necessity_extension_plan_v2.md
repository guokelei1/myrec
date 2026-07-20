# Transformer component necessity extension V2

状态：2026-07-19，在任何 component-necessity score bundle 启动、任何扩展效应或 qrels 被读取前冻结。
V2 因机械审计发现 V1 的 `null_to_full_removal` 同时改变历史内容和自然位置而取代 V1；V1 文件保留
为先验记录但不得执行或进入报告。V2 不修改 `transformer_deep_dive_plan.md`、其 manifest、family
或停止规则。

## 1. 问题与边界

D2 selected-branch 的 `same_full_to_null` 只建立 full-context 节点状态在 null recipient 中的充分性。
本扩展反向测试：在保留 full query、history、candidate、mask、自然 position IDs 与其他参数时，
把 selected block 的节点状态换成同请求 counterfactual donor，是否消除有害 full response。

V1 只使用短序列 null donor；已有位置审计表明删除历史会同步移动 history-summary/readout 的自然
position，因而它不能单独区分内容与位置。V2 固定加入冻结 D5 等长 content-neutral donor：只把完整
history-content token span 换成 `<|endoftext|>=151643`，其余 token、mask、padding、tensor length、
semantic readout index和自然 position IDs 逐元素不变。

最高允许结论仍只是“该节点承载的 history-context state 是 full-context harm 的必要中介”。本扩展
不证明算子必要性、唯一来源、直接 history-token flow、精确层架构参数或跨数据集/模型规模规律。

## 2. 固定人口、模型与顺序

- 数据、candidate、Q2/Q3 checkpoint、字段白名单、content-neutral eligibility与 qrels 边界绑定原
  deep-dive manifest；source test保持关闭。
- 只使用原 D2 fold-0 选择、fold-1 确认后生成的 immutable selected-branch contract，不另选层。
- scorer只处理 normalized-query fold 1，不读取 qrels、surface或target identity。
- 某模型 transition 未确认时，该模型全部计划 cells 保留 `p=1`，不能换层。
- 必须在对应原 D2 selected-branch bundle 完整后、当前 D3--D7/MLP-formation lane释放后才运行。

## 3. 固定节点与条件

四个节点固定为：

1. `block_input_residual`：上游 incoming-state control；
2. `attention_o_projection`：attention branch increment；
3. `mlp_down_projection`：MLP branch increment；
4. `block_output_residual`：完整 selected-block state ceiling。

共同条件：`baseline_full`、`baseline_null`。每节点固定：

- `full_to_full_identity`；
- `null_to_full_removal`：位置混杂的敏感度条件；
- `neutral_to_full_removal`：等长、position-preserving主要条件。

Q2改native candidate readout position；Q3同时改shared prompt、teacher-forced Yes与No三个native
scoring states。neutral donor由冻结full prompt原地替换history span后捕获，且必须逐元素断言full与
neutral path除注册span token IDs外完全相同。找不到完整span、无可见历史或截断不合格的request在
model/qrels前已经冻结；其neutral条件写回full identity以保持完整coverage，统计只用冻结eligible
requests。null条件仍使用全部fold-1 requests。

RMSNorm输出不作为operator bypass；post-attention residual也不加入，因为反向绝对state patch仍不能
证明residual addition本身必要。

## 4. 机械门

- 四个full-to-full identities最大native score绝对误差`<=1e-5`；
- recomputed full/null baseline满足现有path-local BF16 bound；
- eligible neutral path与full path的shape、mask、positions完全一致，span外token逐元素一致，span内
  恰为冻结neutral token；
- Q3 full/null/neutral的shared prompt donor在Yes/No路径逐元素一致；
- 8,000-request control row SHA、eligible request SHA、selected contract、parent bundle、config、
  checkpoint、dataset/request/candidate manifest、V2 plan/manifest与implementation digest全部绑定；
- 完整有限condition coverage；任一失败只记mechanical non-result。

## 5. 注册 contrasts 与判定

donor mode `d in {neutral,null}`、模型`m`、节点`n`、endpoint`y`：

`R_dmny = y(d_to_full_removal) - y(baseline_full)`。

endpoint固定strict-transfer target margin与NDCG@10。neutral使用strict-transfer与冻结content-neutral
eligible交集；null使用全部strict-transfer。bootstrap固定normalized-query cluster、5,000 draws、
seed `20260715`。

每个endpoint包含2 donor modes × 2 models × 4 nodes = 16 units，分别做16-unit BH；两endpoint共32
units。gate-stop/mechanical missing保持`p=1`，计划family不缩小。

有害中介的预期方向固定`R>0`。主要组件支持必须满足：

1. `neutral_to_full_removal` point estimate正、95% CI下界>0、双侧BH `q<0.05`；
2. 原D2同节点same-request sufficiency与history-specific negative control通过；
3. incoming block state若通过，attention/MLP只能称必要mediator，不能称selected block起源；
4. null donor只报告方向一致/不一致的position-confounded sensitivity，不能替代neutral门。

NDCG完整CI落入`[-0.005,+0.005]`才可称practical equivalence；`p>0.05`不能称无必要性。target
margin没有SESOI。若attention/MLP均不通过而block-output neutral removal通过，保持residual/nonlinear
interaction unresolved。跨Q2/Q3模式才可改变全局架构排序，单模型只作model-scoped约束。

## 6. 计算与停止点

Q2/Q3各一个可续跑bundle，单次连续job不超过13,500秒，独立输出目录。两条lane只在parent与既有
末班车释放后触发，不抢占注册主任务。完成两个bundle（或绑定gate-stop/mechanical record）、共享
evaluator、32-unit table、source/data boundary审计后停止；不实现transfer架构、不换数据集、不加
seed/head/neuron，也不覆盖冻结first-round/deep-dive结果。
