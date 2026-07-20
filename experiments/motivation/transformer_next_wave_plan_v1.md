# Transformer mechanism next-wave plan (v1)

状态：2026-07-20，在现有 D0--D7 formal wave 和 21 项 supplemental registry
尚未读取本波效应值前冻结。本文档是后续机制诊断计划，不修改既有
`transformer_deep_dive_plan.md`、parent manifest、component-necessity V2 或
supplemental registry，也不把诊断干预升级为 transfer 方法。

## 1. 为什么需要下一波

现有波次已经覆盖：全层 post-block 定位、固定节点的单组件 sufficiency、attention
edge/head/group、MLP group/formation、RoPE、native readout、Q0/Q1 breadth、objective
conflict 和 LoRA/optimizer geometry。它仍然留下两个不能由单组件结果回答的问题：

1. attention 与 MLP 的影响是相加、互相抵消，还是在 residual composition 中产生非线性
   interaction？单个 `null_to_full_removal` 不能回答这个问题。
2. 目前的 attention causal formal branch 主要集中在 history→readout transport；formation
   （query 如何形成 history summary）与 transport、candidate readout 的串联是否在同一
   请求上形成瓶颈，仍缺少全人口、position-preserving 的联合干预。

下一波只在这两个缺口上增加实验，不按结果增加层、head、group、seed、slice 或模型。

## 2. 不可变边界

- 数据继续只用 `full_confirm_preceding40k_v11` 的 train 与 label-free internal-dev；source
  test、legacy 2k、新数据集均关闭。
- Q2/Q3 使用现有 immutable selected-branch contract 的固定 block；若 contract gate stop，
  对应模型登记 gate-stopped，而不是重选 block。Q0/Q1 不复用 Q2/Q3 的 selected block。
- 所有 score bundle 继续通过显式输入字段白名单、candidate/request hash、完整 finite coverage
  和 shared evaluator；scorer 不读取 qrels。
- 每个 condition 使用独立 run ID、独立 resume 目录，连续 GPU job 不超过 13,500 秒。
- identity/no-op、position-preserving shape、shared prompt（Q3）和 low-precision path 先过
  mechanical gate；任何失败只记 mechanical non-result。
- 主要 endpoint 仍为 strict-transfer target-versus-best-lower-gain margin 和 NDCG@10，
  request-level paired difference，normalized-query cluster bootstrap 5,000 次，seed
  20260715；无效应值不用于本波的层/条件选择。

## 3. N8：joint attention × MLP composition

### 科学问题

单节点 reverse necessity 若都显著，仍不能说明两条分支各自承担独立功能；联合替换可以
区分 additive removal、相互补偿和 residual/nonlinear interaction。

### 固定输入与节点

- 模型：Q2、Q3；请求：normalized-query fold 1；block：各自现有 component contract 的
  `selected_block`，不重新选择。
- 节点：`attention_o_projection` 与 `mlp_down_projection`。`block_output_residual` 的
  单节点结果沿用已注册 reverse-necessity，作为 composition boundary 对照，不在 N8 中
  把最终 residual 覆盖误称为 attention/MLP 作用。
- donor：full recipient 中写入同请求 position-preserving content-neutral donor；null donor
  只作 sensitivity control，不单独支持机制结论。

### 固定 condition

`baseline_full`、`baseline_null`、两个单节点 neutral removal、joint neutral
attention+MLP removal、四个 full/neutral no-op identity。joint removal 的主要统计量是
`joint - (attention_single + mlp_single - full)`，同时保留逐请求 raw contrasts。

### 判定

只有 joint bundle 完整且 identity 通过后才进入 shared evaluator。interaction 只有在两模型
方向一致、position-preserving eligibility、预注册 family 的 BH q<0.05 且相应 same/cross
negative controls 不矛盾时，才能标为 `composition_interaction_candidate`；否则保持
`unresolved`。它不授权“attention 或 MLP 是唯一原因”，也不授权直接实现 residual 架构。

## 4. N9：history formation → transport → candidate readout

### 科学问题

history 可能在 query-to-history formation 阶段没有形成可用 summary，也可能已经形成但在
history-to-readout transport 或 candidate comparison 中丢失。只屏蔽 readout edge 无法区分三者。

### 固定干预

在 blocks `[13,20,27]`、Q2/Q3 全部固定运行：

- formation logits mask：只屏蔽注册 history-summary query 对 history span 的 logits；
- formation value-zero：只去除同一 query row 的 history value edge；
- transport logits mask：沿用既有 history→readout edge condition；
- transport value-zero：沿用既有 value-edge condition；
- sequential formation→transport joint condition；
- no-op mask-then-restore 和 zero-delta identity。

所有 token IDs、span 长度、attention mask、position IDs 和 candidate slate 保持不变；Q3 的
`shared_prompt_last`、`teacher_forced_yes_last`、`teacher_forced_no_last` 三条 native path
一并运行。formation 的 8 个 GQA group 仍全部记录，但不按结果挑 group。

### 判定

N9 主要比较 `formation-only`、`transport-only` 和 joint 的 candidate gap effect，并报告
recurrence、strict-transfer、other-overlap 三个解释面。只有 joint effect 与至少一个独立
表示/中介结果同向，且 cross/wrong-user/position controls 通过，才可把瓶颈定位到路径级；
单个 edge 的显著性只能称 edge sensitivity。

## 5. N10：训练参数化与最终相对分数的补充因果检查

N10 在 N8/N9 终态后启动，固定为两个互不替代的子波：

1. **Q3 LoRA rank-path ablation**：对全部 28 blocks 的 q/v adapter，预注册 A-only、B-only、
   outer-product rank group 和 no-adapter identity；不按层或 rank 的结果选单元，保留所有
   rank group 的 strict-transfer margin/NDCG。
2. **native readout candidate-gap perturbation**：在 final RMSNorm input/output 的完整
   candidate rows 上，固定 full-null direction、candidate-common direction 与 orthogonal
   direction 的小幅 norm-matched perturbation，报告 target margin、all-candidate common
   shift 和 pairwise order flip；它是 readout geometry 诊断，不是训练方法。

N10 只能回答“训练参数化或 candidate-gap readout 是否具有必要的方向性”，不能把某个 LoRA
   rank、hidden coordinate 或绝对层号直接写成架构设计。

## 6. 四卡排程

当前 D2/D3 worker 不被打断。待现有 D3--D7、MLP-formation、necessity 队列完成并通过
ownership audit 后，下一批固定为：

| 物理卡 | 首先运行 | 随后运行 |
|---|---|---|
| GPU0 | N8 Q2 joint composition | N10 Q3 rank-path shard 0 |
| GPU1 | N8 Q3 joint composition | N10 Q3 rank-path shard 1 |
| GPU2 | N9 Q2 formation/transport | N10 candidate-gap shard 0 |
| GPU3 | N9 Q3 formation/transport | N10 candidate-gap shard 1 |

每个 lane 都由独立 `watch_then_run` + `run_deep_dive_resume_loop` 接管；前置 contract、
manifest SHA 和 identity gate 未满足时只等待或 gate-stop，不占用 GPU。

N8 的接续队列已登记为
`scripts/run_deep_dive_next_wave_n8_queue.sh`，当前以 CPU watcher 运行，等待既有
component-design sentinel、Q2/Q3 selected-branch contract 和 parent bundle 全部终态；它
不会抢占当前四卡 worker。N8 两个模型完成后由同一 shared evaluator 读取 qrels，其 scorer
仍保持 qrels-blind。

N9 的独立协议冻结在
`experiments/motivation/transformer_n9_history_path_manifest_v1.yaml`，实现为
`src/myrec/mechanism/history_path_{scoring,runtime,evaluator}.py`，由
`scripts/run_deep_dive_next_wave_n9_queue.sh` 作为第二个 CPU watcher 接续。N9 使用
GPU2/GPU3，按 Q2/Q3 各自顺序扫描 block 13/20/27，与 N8 的 GPU0/GPU1 并行；formation
使用 `history_summary` query 对 `query` span 的 mask/value-zero，transport 使用
`native_readout` query 对 `history` span 的 mask/value-zero，joint 条件在同一次 forward
中同时安装两类 hook。N9 不读取 qrels，且不复用任何 N8 输出或按效应值选择路径。

N10 的第一项 rank-path 子波也已具体化为
`experiments/motivation/transformer_n10_q3_lora_rank_manifest_v1.yaml`，使用
`src/myrec/mechanism/q3_lora_rank_{scoring,runtime,evaluator}.py`。它逐项保留全部
28 层 q/v LoRA 的 `B[:,r:r+1] @ A[r:r+1,:]`（`r=0..7`），并同时运行 A-only、B-only、
no-adapter controls。A-only/B-only 只作为 factor-composition 的机械对照；主要统计只对
固定的 outer-product rank groups 和 no-adapter 进行 strict-transfer 对比。N10 rank queue
只有在 N8/N9 shared evaluator 完成后才会接管 GPU，避免新波与当前四卡波次重叠。

N10 的第二项 candidate-gap geometry 已独立冻结在
`experiments/motivation/transformer_n10_candidate_gap_manifest_v1.yaml`，实现为
`src/myrec/mechanism/candidate_gap_{scoring,runtime,evaluator}.py`。它覆盖 Q0/Q1/Q2/Q3
四个 native scorer，在 `final_rmsnorm_input/output` 分别施加 full-minus-null、
candidate-common 和 deterministic orthogonal 的 0.10 norm-matched perturbation，保留
all-candidate common shift、candidate-relative L2、pairwise order flip 与 absolute shift。
scorer/evaluator 均不读取 qrels；`scripts/run_deep_dive_next_wave_n10_candidate_gap_queue.sh`
等待 N8/N9 后以 GPU1 运行 Q0→Q1、GPU2 运行 Q2、GPU3 运行 Q3，和 GPU0 上的 rank-path
queue 并行。该方向只诊断 readout geometry/normalization，不把任何坐标或扰动升级为方法。

## 7. 交付与停止点

下一波交付 N8/N9/N10 的机器可读 bundle、逐请求 contrasts、机械失败记录和一页增量解释，
再与现有 H0--H5 矩阵合并。若 N8/N9 仍不能把问题从 residual composition 与路径级瓶颈
进一步拆开，结论明确保持 unresolved；不通过诊断结果继续追加 head/neuron/layer，也不在
本阶段实现 transfer architecture。
