# Prompt 01 — Counterfactual Evidence-Contract Transformer (C01)

将下列完整 prompt 交给一个全新 agent。不要同时附上另外三份 prompt。

---

你是 PPS proposed-system candidate **C01** 的唯一负责人。仓库位于
`/data/gkl/myrec`。你的任务不是写一份空泛设计，而是独立提出、锁定、实现并
测试一个具有论文级机制创新可能性的 **LLM4Rec/Transformer** 候选：
**Counterfactual Evidence-Contract Transformer (CECT)**。

你必须完成 proposal lock、最小实现、unit/smoke/train-internal 检查，以及一次
预注册单 seed dev screening。不要只给建议后结束。screening 通过后完成本路
报告并停止，不得未经主协调者登记预算就开始完整多 seed 训练。

## 1. 先读规则与事实

按顺序完整阅读：

1. `AGENTS.md`
2. `doc/07_paper_design_constraints.md`
3. `doc/10_direction_decision.md`
4. `doc/11_experiment_and_dataset_plan.md`
5. `doc/12_experiment_execution_protocol.md`
6. `doc/15_proposed_system_design_principles.md`
7. `doc/24_parallel_llm4rec_design_protocol.md`
8. `paper/introduction_and_motivation.md`
9. `reports/pps_architecture_readiness.md`
10. `reports/pps_c5_insight_audit.json`
11. `reports/pps_c5r3_candidate_history_alignment.json`
12. `experiments/pps_results.md` 与 `experiments/pps_baseline_cards.md`

必须接受的证据边界：motivation 已完成；item-only mean NDCG@10
`0.3453755427` 是当前静态水线；4,677 个 history-present/no-exact-repeat 请求
是 transferable personalization 的关键作用面；4,110 no-history 请求必须退化
到 D2p。C5-R3 只否定其 multi-granular/coarse-category 候选，不禁止本设计，
但也没有证明 query attention、semantic transfer 或 user-identity causality。

## 2. 独占资源与写边界

- 只写 `systems/01_counterfactual_evidence_contract/**`。
- 唯一环境：`myrec-c01`。建议以只读 base 环境克隆到数据盘：
  `CONDA_ENVS_PATH=/data/gkl/conda_envs conda create -n myrec-c01 --clone pps-kuaisearch -y`。
- 唯一物理 GPU：**0**。所有 GPU 命令显式
  `CUDA_VISIBLE_DEVICES=0`；代码内部只见 `cuda:0`。GPU 0 被占用时等待，不得换卡。
- run ID 只能以 `20260710_kuaisearch_c01_` 开头；模型、artifact、tmp 也使用
  `c01_` 前缀。
- proposal lock 前禁止读取、列举、diff、import 或复制
  `systems/02_*`、`systems/03_*`、`systems/04_*` 以及另外三份 prompt、run 和 notes。
- 共享 `src/myrec`、`scripts`、data、baseline、evaluator、manifest 和当前文档
  均为只读。不要修改 doc/paper/reports/experiments/AGENTS 或共享代码。
- 不 commit，不清理 dirty worktree，不覆盖任何既有文件。

## 3. C01 专属架构搜索边界

你的 load-bearing primitive 必须位于 **event-level counterfactual evidence
contract**：

- 一个本地可训练的 Transformer/LM 直接形成 `(query, candidate, history_event)`
  evidence token/state，并端到端产生 candidate ranking logit；
- 对 true event 与 wrong-user、event-shuffled、query-masked、coarse-only 等
  counterfactual twin 使用共享 encoder；只有相对 counterfactual 具有稳定、
  可校准 margin 的 event 才允许产生 personalized residual；
- exact recurrence 是同一 evidence contract 中受保护的 high-fidelity atom，
  不是独立 scorer 或事后 heuristic；
- counterfactual twin 只用于训练与诊断。正式推理只能输入真实 `(q,c,H)`，不能
  依赖在线构造 wrong-user history。

以上是搜索轴，不是现成答案。你必须独立决定 operator 的数学形式、它进入
attention/token/hidden state 的位置、contract 如何归一化、如何避免全零/全一
坍缩，以及如何用至多三个命名组件实现。若文献检索证明这个 operator 已存在，
必须在**读取 dev outcome 前**在 event-contract 轴内 pivot 并重新 lock。

本路禁止：

- C02 式 history hypernetwork / request-specific `Delta W`；
- C03 式 Sinkhorn/optimal-transport/null-mass 主 primitive；
- C04 式 paired-prefix likelihood/logit delta 主 primitive；
- fixed-score router、MoE over baselines、query-type classifier；
- 普通 DIN target attention、ZAM/TEM pooling 或 cross-attention 换名；
- frozen LM embedding + MLP、prompt-only LLM rerank、在线 API。

## 4. 创新性审计必须先于 GPU outcome

使用最新原始论文和官方代码独立检索 nearest neighbors。至少检查 DIN、
SIM/UBR、ZAM、TEM、RTM、SASRec/BERT4Rec、counterfactual recommendation、
counterfactual LLM recommendation 与 evidence calibration。可从
`https://arxiv.org/abs/2409.20052` 和 `https://arxiv.org/abs/2005.08936`
开始，但必须继续检索至当前日期。

在 `notes/nearest_neighbors.md` 逐项写：已有 operator、你的不可归约差异、把
差异退化回邻居的 matched-capacity ablation。不得写“名字不同所以 novel”。若
核心机制可归约，结论必须是 `reducible` 并在看 dev 前 pivot 或 stop。

## 5. Outcome 前必须冻结的文件

在自己的目录创建并填满：

- `README.md`
- `environment.yml` 或 `environment.txt`
- `notes/proposal.md`
- `notes/mechanism_fingerprint.md`
- `notes/nearest_neighbors.md`
- `notes/gate_protocol.md`
- `notes/proposal_lock.json`
- `configs/`、`model/`、`train/`、`tests/`

`proposal_lock.json` 必须记录上述设计文件 SHA256、git commit/dirty、时间、
candidate ID、environment、GPU 0、seed、candidate hash、预算，以及“尚未读取
C01 dev outcome”的声明。先 lock，再生成任何 dev metric。

## 6. 最小实现与 C01 falsifier

只实现回答下列问题所需的最小模型。优先冻结/复用本地 text Transformer，训练
小型 contract path；但 LM/Transformer 必须是实际排序信息流的一部分，不能只
产 embedding 后让传统头完成全部工作。

必须有参数/算力匹配的 plain target-attention Transformer control。C01 的专属
预测是：contract margin/certificate 能识别 true evidence，并且其增量不能由
plain attention 复现。

共同 gate：

1. repeat-present 不弱于 item-only；
2. 4,677 non-repeat history-present 上相对 D2p 有预注册正值；
3. wrong/shuffle/query-mask/coarse twins 使 certificate 与增益消失，而不是与
   true history 同样有效；
4. no-history personalized delta 严格为零，并满足 D2p rank-equivalence；
5. contract 不是近常数，且 event permutation 会产生方向一致的内部变化；
6. 替换为 matched plain target attention 后机制增益消失，否则 C01 不成立。

exact threshold、loss、参数预算、train/internal split、stop-loss 与失败解释必须
先写进 `notes/gate_protocol.md`，不得看 dev 后修改。

## 7. 执行预算与评测

- seed 固定 `20260708`；最多 2 个实现尝试，累计不超过 8 A40 GPU-hours；
- 先 unit tests，再 train/internal smoke；训练/打分永不读 dev/test qrels；
- dev records 只作 label-free scoring；assert candidate hash
  `94eb667000e0d0f389d0a2a4d4730683b71c129043edbfcf627590376e9c123e`；
- 输出统一 `scores.jsonl`，不得自写论文指标；
- 最多 1 次 primary dev screening，调用共享 evaluator 时必须：

```bash
flock tmp/pps_dev_evaluator.lock \
  python scripts/evaluate_scores.py \
  --run-id <20260710_kuaisearch_c01_...> \
  --candidate-manifest data/standardized/kuaisearch/v0_lite/candidate_manifest.json
```

- evaluator 必须追加 `reports/dev_eval_log.jsonl`；不得并发 evaluator；
- config 锁定后做 1000-request deterministic rescore；
- 严禁读取 test records/qrels 或计算 test metric。

## 8. 完成定义

不要仅返回设计文字。结束前必须写
`systems/01_counterfactual_evidence_contract/notes/final_report.md`，包含：

- 架构公式、信息流图的文字定义和 mechanism fingerprint；
- 文献 nearest-neighbor verdict：`distinct` / `reducible` / `uncertain`；
- 所有新文件、环境、命令、GPU-hours、run IDs；
- unit/smoke/determinism 结果；
- dev evaluator call 数及对应 log 行；
- common gate 与 C01-specific falsifier 的逐项结果；
- integrity/leakage/test-lock 检查；
- `advance-to-full-gate`、`pivot-before-more-dev` 或 `stop` 的明确结论。

若 screening 失败，诚实停止；不得追加模块、换 subset、放宽 threshold 或读取
sibling 结果救场。若 screening 通过，也不要擅自启动 full multi-seed training；
完成报告并把预算申请交给主协调者。

---
