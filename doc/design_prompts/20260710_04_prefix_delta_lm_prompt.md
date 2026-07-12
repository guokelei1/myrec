# Prompt 04 — Counterfactual Prefix-Delta Language Recommender (C04)

将下列完整 prompt 交给一个全新 agent。不要同时附上另外三份 prompt。

---

你是 PPS proposed-system candidate **C04** 的唯一负责人，仓库为
`/data/gkl/myrec`。独立设计、锁定、实现并测试一个候选
**Counterfactual Prefix-Delta Language Recommender (CPDLR)**。它必须是本地
LLM4Rec/Transformer ranking architecture；不是调用大模型重排，也不是把两个
现成 scorer 的差值接到 D2p 上。

必须完成 proposal lock、最小实现、unit/smoke/train-internal 检查和一次预注册
单 seed dev screening。screening 后报告并停止；完整 gate 与 multi-seed training
需要后续显式授权。

## 1. 必读材料和事实边界

完整阅读 `AGENTS.md`、doc 07/10/11/12/15/24、paper motivation、architecture
readiness、C5 insight audit、C5-R3 JSON、results 和 baseline cards。

保持以下事实：item-only mean `0.3453755427` 为静态水线；4,677 个
history-present/no-repeat 请求是 transferable surface；4,110 no-history 请求
必须 D2p rank-equivalent。C5-R3 未证明 semantic transfer、query attention 或
identity causality。LLM4Rec 是项目 scope，不是由 C5-R3 推出的必然结论。

## 2. 独占资源与边界

- 只写 `systems/04_prefix_delta_lm/**`。
- 环境 `myrec-c04`：
  `CONDA_ENVS_PATH=/data/gkl/conda_envs conda create -n myrec-c04 --clone pps-kuaisearch -y`。
- 只用物理 GPU **3**；命令 `CUDA_VISIBLE_DEVICES=3`，代码内 `cuda:0`。卡忙
  等待，禁止换卡/多卡。
- run ID 只用 `20260710_kuaisearch_c04_` 前缀，其他输出用 `c04_`。
- proposal lock 前禁止读取、列举、diff、import/copy `systems/01_*`、`02_*`、
  `03_*`、其他 prompts/runs/notes。
- shared src/scripts/data/evaluator/manifest/docs 只读；不 commit、不清理 dirty
  worktree、不修改共同文件。

## 3. C04 专属架构搜索边界

load-bearing primitive 必须是 **同一个共享参数 LM 的 paired history/null-history
candidate-logit operator**：

- 本地 compact causal 或 masked LM/Transformer 对同一 candidate 建模
  `[query, history, candidate]` 与 `[query, NULL_HISTORY, candidate]`；
- personalized evidence 定义在同一模型内部的 token/candidate logit 或 conditional
  likelihood difference，而不是两个独立模型/score channel；
- paired consistency 使 empty、wrong、shuffled、query-masked history 的 delta
  归零，只在 joint `(q,H,c)` 证据足够时允许改变 candidate ordering；
- candidate 必须在固定池内判别式打分或受约束 likelihood scoring，不生成新的
  item ID，不允许 hallucinated candidate；
- query-only path 必须用 train-only objective/anchor 学到并保持 D2p ordering；
  no-history 时两个 prefix 实际相同、delta 为零，整个模型通过 gate 验证与 D2p
  rank-equivalent。

你必须独立决定 prefix/tokenization、candidate scoring、delta placement、共享
forward 的计算复用、query-only anchor、PEFT/LoRA 范围和效率。不得简单实现为
外部 `score = D2p + lambda * LM_delta` 固定混分；同一个 LM ranking path 必须
产生最终 candidate logit。若文献中已有同构 operator，dev 前 pivot/stop。

本路禁止：

- C01 event certificate head；
- C02 hypernetwork/request-specific `Delta W`；
- C03 optimal transport/null-sink；
- 两套独立 LM、fixed-score router、query-type branch；
- prompt-only zero-shot reranking、在线 API、自由文本 item generation；
- 只识别 exact recurrence 而在 4,677 non-repeat 表面无机制预测。

## 4. 创新性审计

在 dev outcome 前检索当前原始论文和官方代码，至少覆盖 generative/sequential
LLM recommendation、Recformer/LLM4Rec、paired/masked-prefix scoring、
counterfactual inference、ZAM/TEM/RTM、SASRec/BERT4Rec 与 personalized search
memory。可从 `https://arxiv.org/abs/2402.10548`、
`https://arxiv.org/abs/2005.08936` 与 BERT4Rec 原论文开始，但必须检索到当前日期。

`notes/nearest_neighbors.md` 写 operator 级不可归约差异，以及 matched single-pass
LM、history concatenation LM、ordinary LoRA 的退化消融。若只有 prompt 模板或
应用数据不同，判 `reducible`；不得凭 CPDLR 命名宣称创新。

## 5. Proposal lock

先完成 `README.md`、env manifest、`notes/proposal.md`、
`notes/mechanism_fingerprint.md`、`notes/nearest_neighbors.md`、
`notes/gate_protocol.md`、`notes/proposal_lock.json` 和本路 configs/model/train/tests。

fingerprint 必须写清两个 prefix、共享参数、candidate logit/likelihood、delta
operator、anchor、训练信号、推理成本、no-history identity、single-pass degeneration。
lock JSON 记录所有 hash、git/dirty、time、seed、env、GPU 3、candidate hash、
budget 和未读取 C04 dev outcome 声明。先 lock 后 dev。

## 6. 最小实现与 C04 falsifier

使用完全本地的 compact LM/Transformer；允许冻结大部分 backbone 或 candidate-
local LoRA，但 LM 必须直接产生 ranking logit。先在小 train/internal slice 证明
prefix、mask、candidate order 和 delta 实现正确。

matched controls 至少包括：

1. single-pass `[q,H,c]` LM（同参数预算，无 paired delta）；
2. query/history simple concatenation + ordinary ranking head；
3. ordinary static LoRA；
4. delta 只看 exact identity 的 shortcut control。

C04 预测：paired delta 在 4,677 non-repeat 表面产生 single-pass LM 不能复现的
正值；wrong/shuffle/query-mask 使 delta 与增益归零；no-history 两 prefix/logit
一致且整体 D2p rank-equivalent；repeat-present 不弱于 item-only。如果 delta
仅识别 exact recurrence、single-pass control 同样有效或 paired forward 只是两
scorer 混分，则 primitive 失败。

在 gate protocol 先冻结 model/backbone、PEFT、candidate scoring、anchor、
threshold、seed、budget、latency、stop-loss 与失败解释。

## 7. GPU 测试和统一评测

- seed `20260708`；最多 2 个实现尝试，总计不超过 8 A40 GPU-hours；
- unit tests 覆盖 shared-parameter identity、empty-prefix delta=0、mask、fixed
  candidates、deterministic likelihood/logit；
- 先 train/internal smoke；training/scoring 永不读 dev/test qrels；
- dev label-free scoring assert hash
  `94eb667000e0d0f389d0a2a4d4730683b71c129043edbfcf627590376e9c123e`；
- 统一 `scores.jsonl`，最多 1 次 primary dev screening；evaluator 持锁：

```bash
flock tmp/pps_dev_evaluator.lock \
  python scripts/evaluate_scores.py \
  --run-id <20260710_kuaisearch_c04_...> \
  --candidate-manifest data/standardized/kuaisearch/v0_lite/candidate_manifest.json
```

- 不并发 evaluator、不自写论文 metric；config 锁后 1000-request deterministic
  rescore；绝不访问 test records/qrels/metrics。

## 8. 完成定义

写 `systems/04_prefix_delta_lm/notes/final_report.md`：公式与信息流、fingerprint、
nearest-neighbor verdict、源文件/config/test/env、commands/GPU-hours/runs、
unit/smoke/determinism、唯一 dev call/log 行、common/C04 falsifier、latency/token
cost、integrity/test lock，以及 `advance-to-full-gate` / `pivot-before-more-dev` /
`stop`。

失败不得改 subset、放宽门槛、堆模块或看 sibling；通过 screening 也先报告，
等待 full-gate 预算授权。

---
