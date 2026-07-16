# Motivation V1.2 后续执行 Prompt

你正在 `/data/gkl/myrec` 中继续 Query-conditioned Personalized Product Ranking 论文的
Motivation V1.2。不要只写计划；在遵守证据边界的前提下，自主完成数据准备、代码开发、
真实训练与评估，并产出第一轮可信结论。

## 开始前必须阅读

先完整阅读仓库根目录 `AGENTS.md`，然后把以下三个 V1.2 文档作为核心：

1. `doc/43_llm_rerank_recurrence_transfer_research_logic_zh.md`：研究逻辑和待检验假设；
2. `experiments/motivation_v1_2/plan.md`：本轮方法、数据、预算、seed 和确认协议；
3. 本执行 prompt：实际交付和停止位置。

实现数据与 evaluator 时再查阅：

- `doc/11_experiment_and_dataset_plan.md`：统一记录、split、label isolation 和 metric；
- `doc/12_experiment_execution_protocol.md`：运行、日志、确定性和确认边界。

需要核对旧结果时再查阅：

- `doc/40_transformer_recurrence_transfer_motivation_v1_zh.md` 和
   `doc/41_motivation_v11_current_conclusion_zh.md`：已有结果和不能越过的 claim boundary。



## 要完成的工作

### 1. 数据准备

审计并复用现有 KuaiSearch 标准化人口、候选 manifest、train-only internal-dev 和冻结
2,000-request cohort。准备 V1.2 所需的统一输入、history assignments 和共享评分接口；在
最终方法冻结后，再从未参与决策的 KuaiSearch source-train 记录中建立同数据集的新
holdout。保持 request/session/time 隔离、候选一致和 qrels 分离，绝不打开 source test。

### 2. 代码开发

搭建一套项目内共享、可 resume 的 Qwen ranking harness。优先复用现有项目自己的 Qwen
配置、checkpoint、评分和 evaluator；论文上游代码只用于阅读核对，最终独立最小重写：

- `Q0 Qwen3-Reranker-0.6B`；
- `Q1 InstructRec-GeneralQwen`；
- `Q2 RecRanker-GeneralQwen`；
- `Q3 TALLRec-GeneralQwen`；
- 主表外的 `W0 CoPPS-style transfer witness`。

`Q1–Q3` 共享一个约 0.5–0.6B、未经额外批准不超过 1B 的通用 Qwen。为共享数据、label
isolation、方法核心 loss、pairwise/listwise 转换、candidate identity、checkpoint resume
和 evaluator contract 添加必要测试。训练标签只能作为监督目标，不能进入模型输入。

### 3. 真实训练与评估

先运行一个预冻结 pilot seed，尽快得到四方法和 witness 的第一轮结果。存在 ready job 时
尽可能并行利用四张 GPU。每个连续训练 job 不超过 4 小时；接近上限时保存完整 checkpoint
并安全退出，未收敛任务记为 pending，不能阻塞其他工作，之后可按同一 run lineage 续跑。

使用同一共享 evaluator，至少生成 full/null 的逐请求结果，并报告 overall、recurrence、
strict transfer、other-overlap、cluster-bootstrap 区间和人口加权贡献；最终确认再补
wrong-user。pilot 结果无论方向都保留。不能因为不符合预期而替换方法、seed、slice 或
evaluator；需要第二 seed 时同时保留并报告第一 seed。所有 dev 调用写入规定日志。

### 4. 结果收束

第一轮结束后，更新 concise report、baseline boundary cards、`experiments/pps_results.md`
以及 motivation 文档中的待填主表。清楚区分：有效结果、机械失败、预算下未收敛、单-seed
preliminary observation 和最终可支持的 claim。完成第一轮单-seed 表与审计后停止扩展，
总结结果并等待下一步指示；不要自行扩大研究范围。

## 待检验的预期结论

目标假设是：多个 history-conditioned LLM ranker 能从历史提升排序，但收益更集中于
recurrence，strict transfer 相对更弱；与此同时，CoPPS-style witness 可能在同一
KuaiSearch strict-transfer surface 上恢复一部分信号，从而说明当前 LLM 方法存在可优化的
transfer headroom。

这只是预期而不是必须得到的结果。若数据表现为部分方法成功 transfer、witness 也失败、
模型未收敛或不同方法结论分裂，应原样报告并相应收窄或否定假设。高质量、可复核的反例
优先于符合预期但经选择的结果。

## 完成标准

- 数据、候选、标签和 holdout 边界通过审计；
- 四个固定 LLM ranker 与 witness 的最小实现和必要测试完成；
- 第一轮 pilot seed 的所有有效结果由共享 evaluator 生成并登记；
- 未完成任务有 resumable checkpoint、明确状态和后续成本；
- 形成一份简洁的 V1.2 当前结论，能够明确说明假设被支持、部分支持、否定或仍不确定。
