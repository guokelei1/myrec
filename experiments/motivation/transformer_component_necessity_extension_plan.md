# Transformer component necessity extension

状态：2026-07-19，在未读取未闭合 D2 family 科学效应、未打开 source test 的条件下冻结。
本扩展不修改 `transformer_deep_dive_plan.md`、其 manifest、family 或停止规则。它补充一个由
“层扫描只能定位、null-context patch 只能证明充分性”暴露出的独立缺口：反向移除某个 full-context
组件状态，是否能消除有害的 history response。

## 1. 问题与边界

D2 selected-branch 的 `same_full_to_null` 回答：full-context 节点状态在 null recipient 中是否足以
重现最终响应。即使通过，它也不证明该节点状态对于 full-context 响应是必要的，更不证明该算子是
唯一来源。这个扩展固定回答：

> 在保留 full query、history、candidate、mask、自然 position IDs 与其余模型参数时，把 selected
> block 的一个 full-context 节点状态替换为同请求 null-context 状态，是否把 strict-transfer 响应
> 从有害 full baseline 拉回正确方向？

精确 block 编号仍只来自原 D2 的 split-sample 手术坐标，不是架构参数。本扩展不直接观察全部
history-token flow，不声明 operator necessity、唯一原因或跨数据集/模型规模规律；它最多建立
“该节点承载的状态是 full-context 有害响应的必要中介”。

## 2. 固定人口、模型与选择顺序

- 数据、candidate、checkpoint、字段白名单与 qrels 边界原样绑定 deep-dive manifest；source test
  保持关闭。
- 模型固定为 Q2 `q2_recranker_generalqwen` 与 Q3 `q3_tallrec_generalqwen`。
- 仅使用原 D2 fold-0 选择、fold-1 确认后生成的 immutable selected-branch contract；不得另选层。
- scorer 只处理 normalized-query fold 1，不读取 qrels、surface 或 target identity。
- 若某模型的 D2 transition 未确认或 selected-branch contract gate-stop，该模型的全部八个
  endpoint cells 固定缺失并以 `p=1` 留在计划 family 中；不能换层。
- 必须等相应原 D2 selected-branch bundle 完成后才运行，以免更改或拖慢原注册因果核心。

## 3. 固定节点与干预

只测试四个事先固定、功能不同的节点：

1. `block_input_residual`：上游进入 selected block 的 incoming-state control；
2. `attention_o_projection`：attention branch increment；
3. `mlp_down_projection`：MLP branch increment；
4. `block_output_residual`：完整 selected-block state ceiling。

每个节点固定生成：

- `full_to_full_identity`：在 full recipient 写回 full donor；
- `null_to_full_removal`：在 full recipient 写入同请求 null donor；
- `baseline_full` 与 `baseline_null` 共同保存。

Q2 只改 native candidate Yes/No readout position；Q3 同时改 shared prompt、teacher-forced Yes 与
teacher-forced No 三个 native scoring states。`post_attention_residual` 不进入本扩展，因为它需要
组合安全的特殊重组，且其反向状态替换仍不能单独证明 residual addition 算子必要性。RMSNorm 输出也
不进入本扩展，因为替换 normalized state 不是对 RMSNorm operator 的 bypass。

## 4. 机械门

- 四个 full-to-full identities 的 native score 最大绝对误差均不超过 `1e-5`；
- recomputed full/null baseline 分别满足现有 path-local BF16 bound；
- Q3 shared prompt donor 在 Yes/No 路径逐元素一致；
- 同一 request/candidate 顺序、完整有限 score coverage、selected contract SHA、config、checkpoint、
  dataset/request/candidate manifest 与 deep-dive manifest 全部绑定；
- 任何 hook、shape、identity、coverage 或 resume 失败只记 mechanical non-result，不能进统计。

## 5. 注册 contrasts 与判定

对模型 `m`、节点 `n`、endpoint `y` 定义请求级 removal effect：

`R_mny = y(null_to_full_removal) - y(baseline_full)`。

endpoint 固定为 strict-transfer target margin 与 NDCG@10；使用 fold-1 normalized-query cluster
bootstrap 5,000 draws、seed `20260715`。两个模型、四个节点、两个 endpoint 共 16 个 unit，按 endpoint
分成两个 8-unit BH families，缺失 cell 保留 `p=1`。

有害中介的预期方向固定为 `R > 0`。支持必须同时满足 point estimate 为正、95% CI 下界大于 0、
双侧 BH `q<0.05`，并且原 D2 对应节点的 same-request sufficiency 与 history-specific negative
control 已通过。NDCG 若完整区间落入 `[-0.005,+0.005]` 只能称 practical equivalence，不能用
`p>0.05` 声称没有必要性。target margin 没有 SESOI。

解释顺序固定：

- `block_input_residual` 通过：有害状态已从上游进入，selected block 不能称来源；
- attention/MLP removal 与原 sufficiency 同时通过：相应 branch state 是必要中介，但不是唯一来源；
- 两个 branch removal 均不通过而 `block_output_residual` 通过：保留 residual/nonlinear interaction
  unresolved，不把它强行归给某一分支；
- sufficiency 通过而 removal 不通过：该节点可复现响应但没有建立必要性，不进入架构优先级；
- 只有跨 Q2/Q3 的组件级模式可改变架构机会排序；单模型结果只作 model-scoped 约束。

## 6. 计算与停止点

两模型各一个可续跑 bundle，使用独立输出目录，单次连续 GPU job 不超过 13,500 秒。它们只在原
D2 selected-branch 与当前 D3--D7 主队列之后进入四卡末班车，不能抢占正在运行的注册任务。

完成两个 bundle（或产生绑定的 gate-stop/mechanical record）、共享 evaluator、16-unit evidence
table、源码/数据边界审计后停止。本扩展不实现 transfer 架构、不增加数据集/seed/head/neuron，
也不修改冻结 first-round 或 deep-dive 结果。
