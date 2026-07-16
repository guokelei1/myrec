# Motivation V1.1 当前结论（收束版）

状态：按用户要求停止后续训练与评分，只汇总截至 2026-07-16 已完成且通过边界审计的证据。Motivation V1 的原始报告、指标、surface 和结论保持不变。

## 已完成证据

KuaiSearch Axis A 在固定 V1 人口上完成了 TEM 和 InstructRec 的两个预冻结 seed。InstructRec 两个 seed 均得到可靠的 target-repeat 增益（`+0.0326`、`+0.0421`），且 repeat-minus-no-overlap contrast 均可靠为正（`+0.0348`、`+0.0473`）；no-overlap 点估计为负但区间跨零，overall gain 也未可靠偏离零。TEM 没有复现该分解，其 repeat 和 contrast 区间均跨零。

KuaiSearch Axis B 只完成了 TEM 两个 seed。overall gain 分别为 `-0.00089` 和 `+0.00323`，方向分叉；repeat、no-overlap 和二者 contrast 的区间全部跨零。因此，已完成的 TEM 证据不能支持更大 KuaiSearch 训练人口上的稳定扩展。Axis B InstructRec 在 checkpoint 和确认评分产生前按用户要求终止，不属于实验结果，也不作任何正负解释。

JDsearch 既有 full-token pairwise bundle 通过同一 evaluator、相同 target-aware surface 定义和 manifest 复核。overall history gain 为 `+0.0790 [0.0660, 0.0917]`；target-repeat 为 `+0.4249 [0.3724, 0.4802]`；no-overlap 为 `+0.0286 [0.0165, 0.0406]`；repeat-minus-no-overlap 为 `+0.3963 [0.3402, 0.4529]`。这说明 repeat 是很强的放大 surface，但 no-overlap 历史也能产生可靠收益。由于 JDsearch 查询边界匿名化且模型家族不同，它只能支持 functional replication，不能支持自然语言语义普遍性。

## 最新结论

当前最强且诚实的结论是：**target recurrence 可以显著放大模型对历史的响应，但它不是已证实的 Transformer 家族普遍规律，也不是历史收益的唯一来源。**

具体而言：

- V1 在冻结的原始 KuaiSearch 人口上仍成立；
- 在冻结的较长 epoch 预算下，InstructRec 的 repeat-versus-no-overlap 分解跨 seed 稳定，但 TEM 不复现；这本身不把 epoch 数量认定为因果来源；
- 已完成的更大人口 TEM 结果不稳定，不能升级人口扩展结论；
- JDsearch 显示 repeat 效应远强于 no-overlap，同时明确否定“no-overlap 没有可靠收益”的跨人口扩展；
- 因此不升级模型家族 prevalence claim，也不支持“历史提升只来源于 repeat”的排他性表述。

完整数值见 `reports/pps_motivation_v11_axis_a_epoch_robustness.json`、`reports/pps_motivation_v11_completed_evidence.json` 和 `reports/pps_motivation_v11_current_summary.json`。当前没有训练、评分或自动监控进程，等待下一步指示。

与该结论最相关的 repeat/explore、个性化商品搜索和 transfer-oriented rerank 文献整理见
[`42_recurrence_transfer_related_work_zh.md`](42_recurrence_transfer_related_work_zh.md)。
将当前实验、相关工作和下一步可检验假设串联后的完整研究逻辑见
[`43_llm_rerank_recurrence_transfer_research_logic_zh.md`](43_llm_rerank_recurrence_transfer_research_logic_zh.md)。
