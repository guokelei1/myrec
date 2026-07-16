# Motivation V1.2 controlled-Qwen LLM ranker plan

状态：2026-07-16 执行前工作计划。当前目标是先搭建统一框架并取得可信实验数据；最终论文
表述在结果完成后再决定。完成第一轮结果汇总后停止并等待用户下一步指示。

## 1. 核心问题

本轮在 KuaiSearch 上测试四个预先固定的 LLM ranker，观察 history gain 在 recurrence 与
strict transfer 之间如何分布。实验不预设四个方法必须得到相同结果，也不根据 transfer
结果替换主表方法。

核心输出是同一 evaluator 下的逐请求排序结果、`full − null` history delta、recurrence、
strict transfer、other-overlap 及其人口加权贡献。先取得数据，再决定最终结论应写成共同
失衡、方法边界还是其他更准确的表述。

## 2. 固定方法与角色

KuaiSearch 主实验从一开始固定为四个方法：

1. `Q0 Qwen3-Reranker-0.6B`：现有专用 Qwen reranker 强锚点；
2. `Q1 InstructRec-GeneralQwen`：推荐/搜索指令化机制的独立最小重写；
3. `Q2 RecRanker-GeneralQwen`：pointwise、pairwise/listwise ranking 机制的独立最小重写；
4. `Q3 TALLRec-GeneralQwen`：recommendation alignment 与参数高效训练机制的独立最小重写。

`Q1–Q3` 使用同一个通用 Qwen checkpoint 和相同初始化。目标规模为 0.5–0.6B；若本机没有
合适模型，可在其他本地目录查找、复制或下载。未经新的明确决策，参数规模不得超过 1B。
`Q0` 保留专用 Qwen reranker，因为它本身就是有意义的强 baseline；报告中必须明确它与
`Q1–Q3` 不是同一个预训练任务边界，不能声称四行完全 matched-backbone。

另设一个不进入四方法主表的诊断角色：

- `W0 CoPPS-style transfer witness`：在相同 KuaiSearch 记录和候选集合上，最小迁移
  不同-ID 语义相关历史增强/对比学习机制，用来检验 strict-transfer surface 上是否存在
  可被专门方法恢复的信号。原 CoPPS 不是 LLM，本项目实现也只能称为 structural witness，
  不能伪装成 LLM baseline 或官方复现。

UR4Rec 与 HMPPS 不进入本轮。

## 3. 实现约束

论文和官方代码可以下载、阅读并用于核对机制，但最终采用
**source-audited independent minimal reimplementation**：

- 不直接接入上游训练器、数据管线、checkpoint 管理或 evaluator；
- 共享的数据读取、字段白名单、候选打分、训练循环、checkpoint 和评估导出由本项目实现；
- 每个方法只增加区分论文机制所必需的 prompt、采样、模块或 loss；
- 上游来源、commit、许可证、迁移内容和未迁移内容写入 boundary card；
- 统一使用 `-style reimplementation`，不称为官方复现；
- 必须测试 candidate identity、pairwise/listwise 转换、loss、checkpoint resume 和
  label isolation。

训练记录中的 `clicked`、`purchased`、`relevance` 只可作为监督目标，绝不能序列化为模型
输入。评分端只允许显式白名单字段；任何方法不得读取确认 qrels。所有论文数字只能由共享
evaluator 产生。

现有项目内 Qwen reranker 配置、checkpoint、score bundle 和 evaluator 代码优先复用。
若数据人口、候选 hash、history condition、输入字段和 metric 完全兼容，`Q0` 的既有结果
可直接登记；只有协议发生实质变化时才重训或重评分。该复用不延伸到论文上游的私有框架。

## 4. 共同实验边界

以下部分统一：

- 同一 KuaiSearch 标准化接口、训练人口、train-only internal-dev 规则和候选集合；
- 相同可见信息类型、因果历史、候选身份与 evaluator；
- `Q1–Q3` 使用同一通用 Qwen、同一最大信息边界和同一分阶段 seed 规则；
- 每个 checkpoint 在相同请求上分别评分 `full` 和 `null`，最终确认再加入 `wrong-user`；
- 同一组 recurrence、strict transfer 和 other-overlap surface；
- 同一 graded NDCG@10、逐请求 delta、人口加权贡献和 cluster bootstrap；
- 所有完成的 seed 和 recipe 如实登记，不选择 best seed。

Seed 采用“单 seed 快速发现、多 seed 后置确认”的两阶段方式：

- 第一轮为所有方法预先固定同一个 pilot seed，只用于尽快验证框架并取得首轮结果；
- pilot 结果无论是否符合预期都保留和报告，不能删除后换一个更有利的 seed；
- 若某个 run 机械失败、数值异常或四小时内未收敛，优先修复后重跑同一 seed 或从 checkpoint
  续跑；这些状态不构成 transfer 结果；
- 若 pilot 出现方向异常或方法间明显不稳定，可以提前运行第二个预先登记的 seed 做诊断，
  但必须同时报告两个 seed，不能用第二个覆盖第一个；
- 若 pilot 结果清晰，可以先推进框架、witness 和后续实验，把多 seed 放到最终稳健性阶段；
  在多 seed 完成前，结果只能称为 preliminary，不能支持最终的多方法稳健性结论。

`null` 不是独立训练任务，也不是论文主角，只是一次廉价的 same-checkpoint counterfactual
评分。没有它，recurrence 与 transfer 的绝对 NDCG 会混入两个 surface 本身的难度差异，
无法估计“历史分别贡献了多少”，因此保留。若输入分布变化成为实际问题，统一使用明确的
`[NO_HISTORY]` 表示或少量 train-only history dropout，而不是删除 counterfactual。

训练超参数可以方法化：learning rate、epoch、batch 组织、负样本和 loss 权重不要求逐项
相同，但应处于合理量级并完整记录。history budget 和 context 设置优先参考已验证的 Qwen
配置，再结合 train-only token coverage 选择；`Q1–Q3` 的主设置保持一致。listwise 方法需
额外登记 candidate chunking、顺序处理和 truncation，但仍在相同候选集合上输出完整排序。

## 5. 数据划分与确认

所有工作仍在 KuaiSearch 内完成，不引入新的数据集来替代主实验：

1. 使用现有训练记录建立固定的 train-only internal-dev，完成实现调试、普通调参和
   checkpoint selection；
2. 现有 2,000-request frozen cohort 用于与 V1/V1.1 对齐和兼容性测量，不用于改变方法或
   recipe；
3. 四个方法、witness、配置和选择规则冻结后，从尚未参与本轮决策的 KuaiSearch
   source-train 记录中建立一个新的同数据集 holdout；它与训练/internal-dev 在 request、
   session 和时间边界上隔离，记录保持 label-free，候选与 qrels 单独落盘；
4. 新 holdout 的具体数量和比例在数据审计后按可用正例与 surface power 合理确定并冻结，
   不使用 source test，也不根据模型结果反向挑人口。

这里的新 holdout 不是换数据集，而是避免用已经观察过多次的 2,000 个请求承担最终确认。
现有 cohort 和新 holdout 都会报告，前者负责可比性，后者负责最终未参与决策的确认。

## 6. 训练时限与四卡调度

4 小时是每个 job 的单次连续运行上限，不是方法的永久累计预算：

- 定期保存 optimizer、scheduler、RNG 和数据游标；接近 4 小时时安全保存 resumable
  checkpoint 并退出；
- 未收敛方法记为 `under-converged/pending`，不能作为 transfer failure，但也不能阻塞其他
  方法、评分或框架开发；
- 后续可以从同一 checkpoint 续跑一个新的不超过 4 小时的 job；run lineage、累计 wall
  time/GPU-hours 和任何配置变化必须记录；
- 不因确认结果不理想而改变 recipe 后续跑，也不能把多个续跑片段伪装成一次短训练。

有四张可用 GPU 且存在 ready job 时，优先并行不同方法、seed、已登记 recipe 或 scoring
任务，目标是尽可能利用四卡。每个 job 使用独立 run ID、输出目录和可写状态。只有一个
ready job 且多卡确有吞吐收益时才使用 DDP；不得为了占卡临时增加无意义实验。

## 7. 执行节点

### P0：模型与数据审计

冻结 `Q0` 既有结果的可复用边界；在全机查找合适的轻量通用 Qwen，必要时下载，并冻结
`Q1–Q3` 的 checkpoint checksum。完成字段白名单、candidate/listwise token coverage、
internal-dev 和新 holdout 构建规则审计。

### P1：共享 ranking harness

以现有 Qwen reranker 管线和共享 evaluator 为基础，抽取统一的数据、训练、resume、评分和
结果导出接口。先让 `Q0` 兼容性复验通过，再实现通用 Qwen 的 plain scoring 能力，确认
backbone 能承载后三个方法。

### P2：四方法与 witness 实现

并行实现 InstructRec、RecRanker、TALLRec 的最小核心迁移和 CoPPS-style witness。每个方法
先通过 tiny fixture 和 smoke run；机械失败、标签泄漏、listwise 输出失败或主任务退化不能
被解释为 transfer failure。

### P3：internal-dev 训练与收敛检查

使用合理且有界的方法化 recipe 训练。每个方法至少需要：数值与候选检查通过、具备非退化
query-candidate 排序能力、learning curve 足以判断 checkpoint 状态。四小时内未收敛则保存
并标记 pending，其他方法继续推进。第一轮只运行预冻结 pilot seed；第二 seed 可以因明确的
稳定性诊断提前启动，也可以统一留到最终确认前补充，但不能按结果好坏选择性替换。

### P4：冻结与确认

先在既有 2,000-request cohort 完成兼容性表；随后冻结所有方法和规则，再一次性运行新的
KuaiSearch holdout。首轮单-seed 表用于快速获得方向，论文级确认再补齐预冻结的多 seed。
四个主方法和所有已运行 seed 全部报告；不按 transfer 方向替换方法或 seed。W0 单独回答
同一 strict-transfer surface 是否存在可恢复信号。

### P5：结果收束

根据数据决定最终表述：可能是多个 LLM ranker 共同失衡、某些方法的边界、预算下未收敛，
或 witness 也未能建立 KuaiSearch transfer headroom。任何一种结果都先如实记录，再决定
当前假设受到何种程度的支持。完成本轮总结后等待用户决定是否开展 transfer 优化。

## 8. 交付物

- 一套可 resume、四卡可调度的共享 ranking harness；
- 四个固定 LLM ranker 与一个 CoPPS-style witness 的实现、测试、配置和 boundary card；
- internal-dev 收敛记录、既有 cohort 兼容性表和新 KuaiSearch holdout 确认表；
- 更新后的 motivation 结论与下一步决策建议。
