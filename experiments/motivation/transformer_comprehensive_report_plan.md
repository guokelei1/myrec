# Transformer comprehensive exploration report plan

状态：2026-07-19，在 supplemental registry 的四个 pending outputs 产生前固定。本文件是最终报告
合同，不新增实验 family、不改变 parent deep-dive 或 component-necessity 统计。

## 1. 生成前置条件

最终全面报告只能在以下条件全部满足后生成：

1. `transformer_deep_dive_plan.md` 的 19 个 deliverable 全部通过 closeout；
2. D2 固定60包与最多2个条件selected-branch均终态，所有有效机械失败保留；
3. `transformer_supplemental_evidence_registry.yaml` 的21项全部通过字节、schema与边界审计；
4. component-necessity V2 的32-unit evaluator终态，设计综合绑定同一parent selected-branch bytes；
5. source test未打开，未切数据集、未实现transfer架构、未覆盖首轮冻结结果；
6. 最终human interpretation逐项引用机器输出，不手抄或重算paper metric。

任一项pending时只允许输出进度报告，不能输出“最终原因”或“最终架构”。

## 2. 证据层级

每项发现必须标记一个且仅一个等级：

- `M mechanical`：hook、identity、coverage、recomposition或失败记录；没有科学方向。
- `D descriptive`：表示、几何、head/group浓度、logit-lens或update分布；只能约束假设。
- `S sufficiency`：full state写入null recipient能复现注册行为，并通过parent结构负控。
- `N necessity`：等长、position-preserving neutral state写入full recipient能移除注册行为。
- `G design-qualified`：同一功能节点同时通过N、S、wrong-user specificity、cross stress、方向/尺度/随机
  负控，并在Q2/Q3复现；只授权设计方向排序，不授权把诊断patch当方法。
- `U unresolved`：不完整、异质、等价区间不收敛或只能定位到混合residual/nonlinear interaction。

描述性supplement不能把component、H0--H5或architecture opportunity升级为supported；单模型N/S结果
只能给model-scoped约束。绝对block index始终是lineage metadata，不进入优化设计。

## 3. 必备报告结构

最终JSON与Markdown必须同时包含：

1. **执行与证据总表**：run数量、机械失败、19个formal deliverable、21个supplement、模型/endpoint/fold；
2. **冻结观察与问题边界**：transfer失败定义、strict-transfer、recurrence、overlap和null/wrong-user边界；
3. **逐层轨迹但非层号设计**：完整曲线、相邻转折、局部/分布式形态、跨模型相对深度与禁止外推；
4. **18组件×4模型矩阵**：serialization、embedding、RoPE、Q/K routing、V transport、O output、SwiGLU
   formation、MLP output、residual、RMSNorm、representation、history routing、candidate interaction、
   native readout、score nullspace、loss、optimizer、LoRA；
5. **功能因果链**：incoming state→attention→MLP→block output→final norm→native score，分别列S/N/G；
6. **输入/表示/路由/readout/训练五层解释**：不能用一个层面的证据替代另一个层面；
7. **Q0--Q3横向边界**：同一现象、异质现象、未覆盖组件和不可推广项；
8. **H0--H5矩阵**：支持、削弱、反证、矛盾证据、negative-evidence basis与剩余不确定性；
9. **全部负结果与冲突**：不删除seed、模型、surface、endpoint或机械non-result；
10. **优化机会排序**：每个方向写目标机制、证据等级、模型scope、预期收益、风险、最小证伪实验；
11. **明确不建议的方向**：说明是被反证、证据不足、位置混杂还是只在描述层成立；
12. **论文claim边界**：CCF-A级设计尚需什么、当前能说什么、不能说什么；
13. **可复现附录**：计划/manifest/registry/input/output SHA、命令、evaluator与source-test审计。

## 4. 优化方向排序合同

每个候选方向必须包含以下字段：

- `opportunity_id`与功能组件，不允许填写绝对layer/head/neuron编号；
- `mechanism_target`和失败链中被干预的位置；
- `minimum_evidence_level`与实际达到等级；
- `supporting_formal_deliverables`、`supporting_supplements`和SHA；
- `contradictory_evidence`，不得为空字符串掩盖冲突；
- `model_scope`、`dataset_scope=kuaisearch_dev`与`source_test_opened=false`；
- `design_priority`、`reason`、`falsification_gate`和`do_not_infer`；
- `diagnostic_patch_promoted_as_method=false`。

只有`component_functional_design_gate_synthesis`可以把基于selected branch的组件方向提升为G；其他架构
机会最多由formal main report排序为待验证候选。NDCG practical equivalence不等于正向utility收益，
target-margin修复也不自动等于ranking改善。

## 5. 停止点

完成上述双格式报告、机器schema审计、21项supplement admission、19项formal closeout和边界检查后停止，
等待用户选择是否进入架构实现。此阶段不实现候选方法、不打开source test、不切换数据集或追加
outcome-selected层/head/neuron/seed。
