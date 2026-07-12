# Prompt 03 — Triadic Transport Transformer (C03)

将下列完整 prompt 交给一个全新 agent。不要同时附上另外三份 prompt。

---

你是 PPS proposed-system candidate **C03** 的唯一负责人，工作仓库是
`/data/gkl/myrec`。独立设计、锁定、实现并测试一个论文级候选
**Triadic Cycle-Consistent Transport Transformer (TCTT)**。它必须是
LLM4Rec/Transformer ranker，而不是在静态分数上增加一个 transport 后处理器。

本轮必须完成 proposal lock、最小实现、unit/smoke/train-internal 检查与一次
预注册单 seed dev screening。不要只交设计说明。完整 gate/multi-seed training
只有 screening 幸存且主协调者登记预算后才允许。

## 1. 必读与不可改写的边界

完整阅读 `AGENTS.md`、doc 07/10/11/12/15/24、
`paper/introduction_and_motivation.md`、`reports/pps_architecture_readiness.md`、
`reports/pps_c5_insight_audit.json`、C5-R3 JSON、results 与 baseline cards。

必须保持：item-only mean `0.3453755427` 是水线；4,677 个 non-repeat
history-present 请求是 transfer 表面；4,110 no-history 请求必须退化到 D2p；
coarse category 和 generic query attention 均未建立。C5-R3 的负结果支持
evidence-fidelity 问题，但不验证 transport。

## 2. 独占目录、环境、GPU

- 只写 `systems/03_triadic_transport_transformer/**`。
- 环境 `myrec-c03`，可用：
  `CONDA_ENVS_PATH=/data/gkl/conda_envs conda create -n myrec-c03 --clone pps-kuaisearch -y`。
- 只使用物理 GPU **2**，命令 `CUDA_VISIBLE_DEVICES=2`，程序内部 `cuda:0`；
  卡忙等待，不得换卡或多卡。
- run ID 只能以 `20260710_kuaisearch_c03_` 开头；其他输出 `c03_` 前缀。
- lock 前禁止读取/列举/diff/import/copy `systems/01_*`、`02_*`、`04_*`、
  sibling prompts、runs、notes。
- shared src/scripts/data/evaluator/manifest/docs 只读；不 commit、不清理、不覆盖。

## 3. C03 专属架构搜索边界

load-bearing primitive 必须是 **query-history-candidate triadic transport**：

- 本地 Transformer/LM 产生 query token、candidate token 与 history-event states；
- 模型内部建立带 learnable null sink 的 entropy-regularized transport 或严格等价
  的质量守恒 operator；只有同时满足 `q<->h`、`h<->c`、`q<->c` 一致性的
  evidence mass 才能影响 candidate ranking logit；
- 不可解释或不可信的 mass 必须进入 null，而不是被 softmax 强制分给某个 event；
- exact item identity 可以是 transport cost 中受保护的 zero-distance/high-fidelity
  atom，但不能是单独 scorer/channel；
- operator 必须可微并与 Transformer 表示端到端连接，不得是离线 OT feature。

你要独立决定 cost、marginals、cycle condition、null parameterization、signed
residual、数值稳定性和复杂度约束。若 literal Sinkhorn 不合理，可在 triadic
mass-conservation/null-sink 轴内提出更好的 operator，但必须证明它不退化成
ordinary attention。若现有文献覆盖核心机制，在 dev 前 pivot/stop。

本路禁止：

- C01 式 event counterfactual certificate head；
- C02 式 hypernetwork/request-specific `Delta W`；
- C04 式 paired-prefix likelihood/logit delta；
- ordinary one-way candidate-to-history softmax attention 换名；
- hard history retrieval + encoder、pooled user vector、fixed-score router；
- frozen LM embedding + MLP、prompt-only rerank、在线 API。

## 4. 创新性与最近邻

在任何 dev outcome 前独立检索最新原始论文与官方代码。至少覆盖 DIN target
attention、SIM/UBR retrieval、ZAM/TEM/RTM、SASRec/BERT4Rec、optimal transport
for recommendation/alignment、Sinkhorn attention、cycle consistency、null/dustbin
matching 和 LLM4Rec。TEM 可从 `https://arxiv.org/abs/2005.08936` 开始。

`notes/nearest_neighbors.md` 必须说明：三方一致性、质量守恒、null sink 中每一项
分别比邻居多了什么可观察预测；用 matched-capacity softmax attention、无 cycle、
无 null 等退化消融逐项交租。名字不同不是创新；可归约则 pivot/stop。

## 5. Proposal lock

先创建：`README.md`、env manifest、`notes/proposal.md`、
`notes/mechanism_fingerprint.md`、`notes/nearest_neighbors.md`、
`notes/gate_protocol.md`、`notes/proposal_lock.json`、candidate-local
configs/model/train/tests。

fingerprint 写清 transport operator、cost/marginal/null/cycle、它插入 Transformer
的位置、训练信号、推理复杂度、zero-history degeneration、softmax degeneration。
lock JSON 记录所有设计 hash、git/dirty、time、seed、env、GPU 2、candidate hash、
budget 和未读取 C03 dev outcome 声明。先 lock 后 dev。

## 6. 最小实现与 C03 falsifier

可以先冻结本地 LM/BGE-style Transformer representation，只训练极小 transport
cost/temperature/null path 做 falsifier；但最终 candidate logit 必须经过
Transformer + transport 的统一信息流。

必须有参数/算力匹配的：

1. one-way softmax target attention；
2. no-null transport；
3. no-cycle pairwise matching；
4. history retrieval/mean pooling（若预算允许）。

C03 的专属预测：true non-repeat evidence 形成 cycle-consistent mass；wrong、
shuffle、query-mask、coarse-only 会把 mass 推向 null 并消除增益；softmax/no-null
对照无法同时做到。no-history transport residual 必须严格为零，repeat-present
不能弱于 item-only。

先在 gate protocol 冻结数值稳定容差、mass/null diagnostics、threshold、seed、
budget 和 stop-loss。若 learned transport 近似普通 attention，或 softmax 对照
复现同等行为，本 primitive 失败，不得以性能数字掩盖。

## 7. 执行与评测红线

- seed `20260708`；最多 2 次实现尝试，总计不超过 8 A40 GPU-hours；
- unit tests 必须含手算小矩阵：质量守恒、null mass、mask、no-history、梯度有限；
- 先 train/internal smoke；training/scoring 不读 dev/test qrels；
- dev label-free scoring assert candidate hash
  `94eb667000e0d0f389d0a2a4d4730683b71c129043edbfcf627590376e9c123e`；
- 输出统一 scores；最多 1 次 primary dev screening；共享 evaluator 持锁：

```bash
flock tmp/pps_dev_evaluator.lock \
  python scripts/evaluate_scores.py \
  --run-id <20260710_kuaisearch_c03_...> \
  --candidate-manifest data/standardized/kuaisearch/v0_lite/candidate_manifest.json
```

- 不自写 metric，不并发 evaluator；config 锁后做 1000-request deterministic
  rescore；严禁 test records/qrels/metrics。

## 8. 完成定义

写 `systems/03_triadic_transport_transformer/notes/final_report.md`，包含公式、
fingerprint、nearest-neighbor verdict、代码/config/test/env、commands、GPU-hours、
runs、unit/smoke/determinism、dev log 行、common/C03 gate 逐项状态、mass/null
诊断、integrity/test lock，以及 `advance-to-full-gate` / `pivot-before-more-dev` /
`stop`。

失败即诚实停止，不追加组件、挑 subset、改 threshold 或看 sibling。screening
通过也先完成报告，等待 full-gate 预算授权。

---
