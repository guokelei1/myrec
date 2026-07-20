# 2026-07-20：从机制探索转入 transfer 架构开发

## 决定

用户在审阅机制解释、架构方向、相关工作风险和训练需求后，明确授权正式实现并验证新的
Transformer-native transfer 架构。当前工作从“继续穷尽 Transformer 组件根因”转为“用已有
证据约束一个可证伪方法”。

## 证据边界

- M0--M4 首轮机制诊断完整结束，作为冻结机制依据；
- D0--D7 deep-dive 只保留当前部分完成的证据快照，不补写成完整组件根因；
- N8--N34 与未完成的 component/readout/optimizer 扩展全部转为非活动归档，不是方法开发前置；
- 最窄解释是 task-aligned、candidate-relative 历史成分在深层相对衰减或被
  candidate-common/off-task 更新覆盖，最终表现为候选特异的符号/校准失配；该解释不是跨模型
  单组件定论。

完整实验盘点见
[`../../reports/motivation_post_stage_experiment_inventory_zh.md`](../../reports/motivation_post_stage_experiment_inventory_zh.md)。

## 新阶段

新阶段按
[`../../experiments/motivation/candidate_contrast_architecture_plan.md`](../../experiments/motivation/candidate_contrast_architecture_plan.md)
执行。主方向是 Candidate-Contrast Personalization：在 Qwen Transformer 内部将历史条件更新分成
候选公共与候选相对分量，通过可关闭的 candidate-contrast 路径写回候选状态。

开发仍只使用 KuaiSearch train/internal-dev；冻结 Q0--Q3/W0 不覆盖；legacy 2k/new 4k 只在方法
冻结后描述性复核；source test 保持关闭。下一动作是设计冻结、代码骨架和数值单元合同，不自动
恢复任何旧 deep-dive queue。
