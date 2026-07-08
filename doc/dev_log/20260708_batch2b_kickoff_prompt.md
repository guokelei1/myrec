# Batch 2b 开发 kickoff prompt

用途：把下面整段作为开发任务的启动 prompt（给开发者或 coding agent）。
它自包含地指向所有规范文档，开发过程中不需要口头补充规则。

---

## PROMPT 开始

你要在 /home/gkl/myrec 仓库中执行 Batch 2b：把三个占位 adapter baseline
（B4/B5/B6）升级为官方实现或经外部基准验证的复现（B4o/B5o/B6o）。

**唯一的执行依据是 `doc/14_official_baseline_plan.md`，按其 §8 checklist
从 step 0 顺序执行。** 开工前必须完整读过：

- `doc/14_official_baseline_plan.md`（本批次计划，含结论层级、止损规则、红线）
- `doc/13_baseline_implementation_plan.md` §1/§2.4/§2.5/§2.6/§6/§7
  （统一输入输出、公平矩阵、预算、交付物、完成定义、红线）
- `doc/12_experiment_execution_protocol.md`（环境组、run_id、metadata、
  评测边界、dev_eval_log、复跑确定性）
- `doc/11_experiment_and_dataset_plan.md` §1.4（显著性定义）
- `experiments/pps_baseline_cards.md` 的 Batch 2 卡片（了解旧 adapter 边界）

背景（为什么做）：Batch 2 的 B4/B5/B6 是 hashed-logistic 简化 adapter，
显著低于 B0b/B7，只能证明该实现弱，不能支撑"现有方法在此数据上不行"
的论文 motivation（strawman 问题）。Batch 2b 的目的是让这个 motivation
可辩护。论证逻辑见 `doc/baseline_notes/batch2b_motivation_logic.md`。

硬性要求（违反任一条 = run 作废重跑，见 doc 14 §9）：

1. step 0 先行：提交 `reports/pps_batch2b_budget_amendment.md` 之前，
   禁止产生任何 Batch 2b dev 评测。
2. 训练数据只准来自 `records_train.jsonl`（统一交互导出脚本，带
   "无 dev/test 字段" assert）；qrels 任何时候不准读。
3. 推理历史 = 当前 record 冻结的 ≤50 条 history，不准用用户全量序列。
4. candidate manifest（94eb667...）、共享 evaluator、compare 脚本不准改。
5. 每个方法 16 次 dev 评测预算，第 1 次 = 官方默认超参；每次评测自动
   记入 `reports/dev_eval_log.jsonl`；冻结后 3 seeds。
6. 外部对齐（RecBole ml-100k sanity、B5o 官方数据 ±10%、B6o Amazon 基准
   ±10%）不计 dev 预算，但必须落盘报告；对齐失败走 doc 14 各节降级
   规则，不准硬标 official/faithful。
7. 执行顺序 B4o → B6o → B5o，一个完成验收（doc 13 §6 十一条 +
   doc 14 各节验收）再开下一个。
8. **结论中立**：无论结果偏向哪个方向（官方 baseline 赢或输 B7-best），
   合格 run 一律登记，措辞按 doc 14 §0 结论层级填写，不准按期望结果
   筛选或重跑。

交付终点：doc 14 §7 交付物表全部齐备 +
`reports/pps_batch2b_decision_summary.md` + 重跑 M3 三通道 oracle +
旧 B4/B5/B6 卡片标记 retired placeholder。

遇到计划未覆盖的决策点（例如官方 repo 结构与预期不符），不要自行发挥：
在 `doc/baseline_notes/` 记录问题与候选方案，停下来等确认。

## PROMPT 结束
