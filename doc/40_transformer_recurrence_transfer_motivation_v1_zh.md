# Transformer 历史收益的 recurrence–transfer 分离：Motivation V1

状态：2026-07-16 V1，当前 motivation 的唯一简明入口。本文只整理已完成证据，
不提出架构、不打开 test，也不把经验现象归因到某个 Transformer 模块。

机器可读结果见
[`../reports/pps_three_transformer_history_surface_audit.json`](../reports/pps_three_transformer_history_surface_audit.json)。
冻结 Qwen 确认见
[`../reports/pps_motivation_confirmation_decision.json`](../reports/pps_motivation_confirmation_decision.json)。

## 1. V1 结论

> 在同一冻结的自然商品搜索人口上，Qwen3、TEM 和 InstructRec 三个不同的
> query-conditioned Transformer 排序模型都对历史产生广泛响应，并在
> `target-repeat` 请求上获得显著排序增益；但在规模更大的
> `target-nonrepeat/no-candidate-overlap` 请求上，都没有建立可靠的
> same-checkpoint 历史增量。

因此，当前观测到的共同结构不是“模型完全不使用历史”，而是：

```text
历史进入打分函数
    → direct target recurrence 可以被可靠利用
    → 对新候选集的 nonrepeat transfer 没有被可靠建立
```

这个结论可以称为三个已选 Transformer ranker 上的共同观察，不能扩写成所有
Transformer 或所有 LLM 的普遍定理。TEM 是商品搜索 item Transformer，而不是 LLM；
因此 V1 的家族措辞统一使用 `Transformer ranker`。

## 2. 统一结果

三者使用同一 2,000-request KuaiSearch confirmation population、同一候选 manifest、
同一 target-aware surface 定义。数值为每请求 graded NDCG@10 的
`true history − null history`，区间为 5,000 次 normalized-query cluster bootstrap。

| 模型 | 历史响应率 | overall recovery | target-repeat recovery | nonrepeat/no-overlap recovery | repeat − no-overlap |
|---|---:|---:|---:|---:|---:|
| Qwen3-Reranker-0.6B | 88.85% | +0.01305 `[+0.00575,+0.02023]` | +0.23150 `[+0.17834,+0.28447]` | +0.00324 `[-0.01437,+0.02147]` | +0.22826 `[+0.17164,+0.28293]` |
| TEM | 89.15% | +0.00032 `[-0.00578,+0.00637]` | +0.05716 `[+0.01764,+0.09760]` | −0.01633 `[-0.03317,+0.00039]` | +0.07349 `[+0.03023,+0.11924]` |
| InstructRec / Flan-T5-XL | 85.75% | +0.00061 `[-0.00279,+0.00410]` | +0.03375 `[+0.00905,+0.06064]` | −0.00017 `[-0.00947,+0.00938]` | +0.03392 `[+0.00673,+0.06253]` |

三个模型的 repeat positive control 和 repeat−no-overlap contrast 均通过；三个模型的
no-overlap nonrepeat recovery 均未建立。只有 Qwen 的 overall recovery 显著为正，
所以 V1 不要求、也不声称每个模型都从历史获得正的 aggregate gain。

## 3. 为什么 TEM 和 InstructRec 的 overall 接近零

两者不是 score-invariant。它们的 aggregate 由互斥 target-aware surface 加权后发生
抵消：

| 模型 | repeat 对 all-request delta 的贡献 | other-candidate overlap | nonrepeat/no-overlap | 合计 |
|---|---:|---:|---:|---:|
| TEM | +0.00346 | +0.00161 | −0.00475 | +0.00032 |
| InstructRec | +0.00204 | −0.00138 | −0.00005 | +0.00061 |

因此“overall true−null 约为零”不能解释为历史输入无效。它只说明该 checkpoint 在
整个请求混合上的净增量接近零；分层后可以同时存在显著 recurrence 收益和无效或有害
的 nonrepeat 增量。

## 4. 代码与反事实审计

- 三个模型的 true/null 使用同一 checkpoint、请求集合、候选集合和评分接口；
- TEM confirmation true materialization 含 22,706 个历史事件，null 为 0；其
  1,783 个 history-present 请求全部产生有效响应，217 个原生 no-history 请求没有
  排序效用变化；
- InstructRec true/null/wrong 使用同一 Flan-T5-XL checkpoint；原生 no-history 请求
  的 true/null 分数逐项完全相等；
- InstructRec 在 train/confirmation 上约 0.9% prompt 超过 2,048 token；删除任何条件
  下会 overflow 的请求后，repeat 仍为 `+0.03645 [0.01113,0.06422]`，no-overlap
  nonrepeat 仍为 `−0.00031 [-0.01010,0.00931]`；
- adapter、surface、counterfactual evaluator 和 paired comparison 的聚焦测试
  在 V1 整理复核中 `21 passed`。

当前没有发现可以解释三模型共同分层结果的候选集、历史构造、checkpoint、截断或
evaluator 错误。

## 5. V1 支持与不支持什么

### 支持

- 三个选定的 query-conditioned Transformer ranker 都能读取历史；
- direct target recurrence 是三者共同的、显著的历史学习正控制；
- 更大的 target-nonrepeat/no-candidate-overlap 流量没有建立可靠历史增量；
- overall、history response 和 target-aware incremental utility 是不同证据义务；
- recurrence success 可以掩盖或抵消 nonrepeat transfer 的不足。

### 不支持

- 所有 Transformer、LLM 或 LLM4Rec 都有该现象；
- TEM 或 InstructRec 已建立正的 aggregate history gain；
- no-overlap 上最终排序模型本身一定差于 query-only 模型；
- 现象已经被定位到 architecture，而不是数据、监督、目标、优化或接口；
- V1 单独授权了 proposed-system 工作或证明必须设计新架构。

## 6. V1 与后续证据边界

V1 已足够支撑论文 motivation 中的有界表述：

> Across three selected query-conditioned Transformer rankers, history utility
> is consistently recurrence-concentrated: target-repeat gains are reliable,
> whereas incremental value is not established on the larger
> target-nonrepeat/no-candidate-overlap surface.

如果后续要把“三模型观察”升级为“模型家族常见现象”，最低还需要：

1. TEM 与 InstructRec 的冻结多 seed 结果，且不得选择最好 seed；
2. TEM 的 wrong-user provenance control；
3. 最好在第二个自然搜索人口上由 Qwen 加至少一个其他模型复现相同 surface contrast。

这些属于 V1.1 robustness，不改变 V1 已经观察到的事实，也不授权架构搜索。

## 7. 证据索引

- 三模型统一审计：
  [`pps_three_transformer_history_surface_audit.json`](../reports/pps_three_transformer_history_surface_audit.json)
- Qwen 冻结五门判定：
  [`pps_motivation_confirmation_decision.json`](../reports/pps_motivation_confirmation_decision.json)
- Qwen target-aware surface：
  [`pps_history_response_confirmation_target_aware_surfaces.json`](../reports/pps_history_response_confirmation_target_aware_surfaces.json)
- 三模型任务端点与边界：
  [`pps_query_conditioned_baseline_comparison.json`](../reports/pps_query_conditioned_baseline_comparison.json)
- 当前状态与剩余边界：
  [`history_response_gap_motivation_status.json`](../reports/history_response_gap_motivation_status.json)
