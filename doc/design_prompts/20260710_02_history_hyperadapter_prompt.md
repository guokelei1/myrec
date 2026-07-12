# Prompt 02 — Candidate-Conditioned History HyperAdapter (C02)

将下列完整 prompt 交给一个全新 agent。不要同时附上另外三份 prompt。

---

你是 PPS proposed-system candidate **C02** 的唯一负责人。仓库位于
`/data/gkl/myrec`。你要独立提出、锁定、实现并测试一个具有论文级机制创新
可能性的 **LLM4Rec/Transformer** 候选：**Candidate-Conditioned History
HyperAdapter Transformer (CHHT)**。

必须完成 proposal lock、最小实现、unit/smoke/train-internal 检查和一次预注册
单 seed dev screening；不要只写方案后结束。screening 后完成报告并停止，完整
multi-seed training 仍需主协调者按 gate 结果授权。

## 1. 必读材料与事实边界

完整阅读：`AGENTS.md`、`doc/07_paper_design_constraints.md`、`doc/10_direction_decision.md`、
`doc/11_experiment_and_dataset_plan.md`、`doc/12_experiment_execution_protocol.md`、
`doc/15_proposed_system_design_principles.md`、
`doc/24_parallel_llm4rec_design_protocol.md`、`paper/introduction_and_motivation.md`、
`reports/pps_architecture_readiness.md`、`reports/pps_c5_insight_audit.json`、
`reports/pps_c5r3_candidate_history_alignment.json`、`experiments/pps_results.md` 和
`experiments/pps_baseline_cards.md`。

事实不能改写：item-only mean `0.3453755427` 是静态水线；4,677 个
history-present/no-exact-repeat 请求是 transfer 作用面；4,110 no-history 请求
必须退化到 D2p。category-only、generic query attention、same-label oracle 和
identity causality 均未建立。LLM4Rec/Transformer 是项目设计选择，不是
motivation “证明必须如此”。

## 2. 独占资源与隔离

- 只写 `systems/02_history_hyperadapter/**`。
- 环境只用 `myrec-c02`；可执行：
  `CONDA_ENVS_PATH=/data/gkl/conda_envs conda create -n myrec-c02 --clone pps-kuaisearch -y`。
- 只用物理 GPU **1**：所有命令 `CUDA_VISIBLE_DEVICES=1`，程序内部只用
  `cuda:0`。卡忙则等待，不得换卡。
- run ID 必须以 `20260710_kuaisearch_c02_` 开头；其他输出使用 `c02_` 前缀。
- proposal lock 前禁止读取、列举、diff、import 或复制 `systems/01_*`、
  `systems/03_*`、`systems/04_*`、另外三份 prompt 及其 runs/notes。
- 共享 source/scripts/data/evaluator/manifest/docs 只读；不要修改共享文件、
  commit 或清理 dirty worktree。

## 3. C02 专属架构搜索边界

load-bearing primitive 必须是 **history-conditioned internal parameter
modulation**，而不是 history score 融合：

- 由 query、candidate 与 strictly-prior history 的交互 Transformer 生成有界、
  低秩、request/candidate-specific functional update `Delta W(q,c,H)`；
- `Delta W` 必须作用于同一 query-candidate LM/Transformer 的内部 Q/K/V、FFN
  或 adapter computation，改变 candidate logit 的计算函数；
- 可以使用共享低秩基底、系数约束、谱范数/门幅限制或其他稳定化，但只能形成
  一个 primitive，至多三个命名组件；
- exact recurrence 必须由同一个 internal modulation 的 preservation constraint
  保护，不能成为独立 fixed scorer；
- no-history 时 `Delta W` 必须按构造为零或严格等价于未调制 query-only path。

你必须独立决定更新哪些层、如何生成低秩基底/系数、如何避免 per-candidate
计算爆炸、如何保持 D2p/no-history ranking，以及怎样证明它不是 ordinary LoRA
或 output gate。若文献已经覆盖核心 operator，必须在 dev outcome 前在 internal
modulation 轴内 pivot 并重新 lock。

本路禁止：

- C01 式 event certificate/counterfactual contract 主 primitive；
- C03 式 optimal transport/Sinkhorn/null-sink 主 primitive；
- C04 式 paired-prefix likelihood/logit delta 主 primitive；
- per-user permanent adapter、one-model-per-user 或 test-time fine-tuning；
- ordinary static LoRA、输出层 history gate、fixed-score router；
- frozen LM embedding + MLP、prompt-only rerank、在线 API。

## 4. 独立创新性审计

在任何 dev outcome 前检索最新原始论文/官方代码，至少覆盖 hypernetwork、
dynamic weight generation、LoRA/PEFT personalization、adapter-based recommendation、
DIN/ZAM/TEM、SASRec/BERT4Rec 与 LLM personalization。必须重点审计
`https://arxiv.org/abs/2510.16282` 一类 profile-to-PEFT/hypernetwork 近邻，
说明你的 **query-candidate-history-conditioned ranking-time functional update**
与 per-profile adapter generation 的不可归约差异。

在 `notes/nearest_neighbors.md` 记录 operator 级差异和退化消融。若只能靠应用
场景或名字区分，判为 `reducible`，在看 dev 前 pivot/stop；不能先验宣称 novel。

## 5. Proposal lock

先创建并完成：

- `README.md`、`environment.yml|txt`
- `notes/proposal.md`
- `notes/mechanism_fingerprint.md`
- `notes/nearest_neighbors.md`
- `notes/gate_protocol.md`
- `notes/proposal_lock.json`
- candidate-local `configs/`、`model/`、`train/`、`tests/`

fingerprint 至少写出 `Delta W` 的公式、生成输入、作用层、rank、边界、参数/计算
复杂度、训练信号、推理路径、no-history degeneration、ordinary-LoRA degeneration。
lock JSON 记录文件 hashes、git/dirty、time、seed、GPU 1、env、candidate hash、
预算和未读取 C02 dev outcome 声明。先 lock，后 dev。

## 6. 最小实现与 C02 falsifier

优先冻结同一 compact LM/Transformer backbone，只训练回答假设所需的 rank-r
generator/modulation；但 ranking logit 必须由被调制的 Transformer 内部路径产生。

必须提供相同参数/算力预算的 controls：

1. ordinary static LoRA；
2. output-layer history gate；
3. mean-history residual 或 plain target attention；
4. 若可行，去掉 candidate conditioning 的 history-only hyperadapter。

C02 专属预测：candidate-conditioned internal modulation 在 non-repeat 表面产生
ordinary LoRA/output gate 不能复现的增量；wrong/shuffle/query-mask 会显著改变
或压低 `Delta W`，no-history `Delta W=0`；repeat-present 不损伤 item-only。

共同 gate 仍包括 repeat、4,677 non-repeat、wrong/shuffle/coarse/query-mask、
4,110 no-history 与 deterministic rescore。必须在 `notes/gate_protocol.md` 先冻结
阈值、layer/rank choice、参数匹配方法、seed、budget、stop-loss 和失败解释。

## 7. GPU 测试与 evaluator

- seed `20260708`；最多 2 个实现尝试，总计不超过 8 A40 GPU-hours；
- 先 unit tests 和 train/internal smoke，不读 dev/test qrels；
- dev label-free scoring 前 assert candidate hash
  `94eb667000e0d0f389d0a2a4d4730683b71c129043edbfcf627590376e9c123e`；
- 只输出统一 `scores.jsonl`；最多 1 次 primary dev screening；
- 调用共享 evaluator 必须持锁：

```bash
flock tmp/pps_dev_evaluator.lock \
  python scripts/evaluate_scores.py \
  --run-id <20260710_kuaisearch_c02_...> \
  --candidate-manifest data/standardized/kuaisearch/v0_lite/candidate_manifest.json
```

- evaluator 追加统一 dev log；禁止私有 evaluator/metric；
- config 锁定后做 1000-request deterministic rescore；
- test records、test qrels、test metrics 一律禁止。

## 8. 完成定义

写入 `systems/02_history_hyperadapter/notes/final_report.md`：公式与信息流、
mechanism fingerprint、nearest-neighbor verdict、文件/环境/命令/GPU-hours/run、
unit/smoke/determinism、唯一 dev call/log 行、common/C02 gate 逐项结果、完整性
检查，以及 `advance-to-full-gate` / `pivot-before-more-dev` / `stop`。

失败时不得改 subset、放宽 threshold、堆模块或读取 sibling 结果。通过 screening
也不得自动多 seed/full training；完成报告并把 full-gate 预算申请交主协调者。

---
