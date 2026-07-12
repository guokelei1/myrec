# 24 - Parallel LLM4Rec Candidate Design Protocol

状态：**历史执行协议；C01--C80 已关闭。最终终局见
`doc/dev_log/20260712_c01_c80_terminal_retrospective.md`。本文只保留当时的四路
隔离、GPU 与 evidence-hygiene 记录，不授权新的 probe、training、C81 或 rescue。
当前执行权威是 `doc/31_problem_discovery_and_architecture_iteration_protocol.md`。**

本协议把 `doc/15_proposed_system_design_principles.md` 落成四个可并行执行、但在
科学上互相独立的 proposed-system candidate。它不修改 C5-R3 的冻结结果：
`TERMINAL_FAIL` 仍关闭 doc/23 的 multi-granular/coarse-category recovery
ladder；当前新问题是如何用 LLM4Rec/Transformer 内部机制区分可靠 exact
recurrence 与未经验证的 cross-item transfer。

“测试”在本文只指 unit test、smoke、train-internal validation 和受预算约束的
dev-only gate probe。**test records、`qrels_test.jsonl` 和 test metrics 全程
禁止访问。**

---

## 1. 共同硬约束

四路共享以下规则；这些规则不能为了制造方案多样性而改变：

1. 每个候选必须满足 doc/15 §3 的 LLM4Rec 定义：Transformer/LM 是
   load-bearing 排序核心，并在内部联合建模 query、strictly-prior history、
   candidate；embedding+MLP、prompt-only rerank、fixed-score router 不合格。
2. 每路只有一个 falsifiable primitive，至多三个命名组件。可以增加控制模块，
   也可以修改 token、attention、Q/K/V、FFN、memory、adapter 或训练目标，但
   必须由 matched-capacity 与退化消融证明机制贡献。
3. item-only C5-R3 mean NDCG@10 `0.3453755427` 是 binding static waterline；
   full D2s `0.3416289845` 只是 bundled-history reference。
4. 共同作用面：8,119 history-present 请求；其中 4,677 个没有 exact-repeat
   candidate，是 transferable personalization 的关键表面；4,110 no-history
   请求必须逐请求退化到 D2p。
5. coarse category、same-label oracle/router、Consensus Law、Slot-Complementarity
   与 generic query attention 均不是已验证 premise。
6. 统一 label-free JSONL、candidate manifest、score schema 和共享 evaluator；
   candidate hash 固定为
   `94eb667000e0d0f389d0a2a4d4730683b71c129043edbfcf627590376e9c123e`。
7. 训练/打分代码永不读取 `qrels_dev.jsonl` / `qrels_test.jsonl`；只有共享
   evaluator 可读 dev qrels。任何违反使该 candidate 全部 run 作废。
8. test 不参与 proposal、筛选、gate、修复或 tie-break。

## 2. 四路机制与资源冻结

2026-07-10 `nvidia-smi` 只显示四张空闲 NVIDIA A40，物理编号 0--3；没有
GPU 4。四路绑定如下：

| ID | 独立目录 | 主搜索轴 | Env | GPU | Run prefix |
|---|---|---|---|---:|---|
| C01 | `systems/01_counterfactual_evidence_contract/` | event-level counterfactual evidence contract inside Transformer | `myrec-c01` | 0 | `20260710_kuaisearch_c01_` |
| C02 | `systems/02_history_hyperadapter/` | query/candidate/history-conditioned low-rank modification of LM internals | `myrec-c02` | 1 | `20260710_kuaisearch_c02_` |
| C03 | `systems/03_triadic_transport_transformer/` | triadic transport with explicit null sink | `myrec-c03` | 2 | `20260710_kuaisearch_c03_` |
| C04 | `systems/04_prefix_delta_lm/` | shared-LM history/no-history candidate-logit delta | `myrec-c04` | 3 | `20260710_kuaisearch_c04_` |

Agent 只能使用 assigned physical GPU；命令显式设置
`CUDA_VISIBLE_DEVICES=<assigned>`，代码内部只使用可见的 `cuda:0`。卡被占用时
等待或报告，不能改用 sibling GPU，不能隐式多卡。

环境建议以当前可用 PyTorch/Transformers 环境为只读 base，在数据盘本地环境区
创建四个独立 env，例如：

```bash
CONDA_ENVS_PATH=/data/gkl/conda_envs \
  conda create -n myrec-c01 --clone pps-kuaisearch -y
```

其他三路只替换环境名。真实 prefix/cache 不提交；每路在自己的 systems 目录
记录精简 `environment.yml` 或 `environment.txt`、Python/Torch/CUDA/Transformers
版本与创建命令。任何依赖修改只发生在本路环境。

## 3. 独立性与写边界

proposal 与 gate protocol 锁定前，每个 agent：

- 不得读取、列举、复制、import、diff 或总结其他三路 systems 目录、prompt、
  notes、configs、code、checkpoint、run 或输出；
- 可读取共同权威文档、共享 `src/myrec`、baseline source/outputs、统一数据接口和
  evaluator，但这些都视为 read-only；
- 只写自己的 `systems/<candidate>/**`、`runs/<assigned-prefix>*`、
  `models/<candidate-id>_*`、`artifacts/<candidate-id>_*`、`tmp/<candidate-id>_*`；
- 不修改 `AGENTS.md`、`doc/`、`paper/`、`reports/`、`experiments/`、共享
  `src/myrec/`、`scripts/`、evaluator、manifest 或冻结 C5 文件；唯一共享写入是
  evaluator 在持锁时追加 `reports/dev_eval_log.jsonl`；
- 不提交 commit，不清理 dirty worktree，不覆盖用户或 sibling 文件。

若需要共享修复，agent 必须停止并把最小复现交给主协调者。独立阶段不得为了
方便建立 shared candidate library；胜出后再审查哪些代码可提升到 `src/myrec`。

## 4. Proposal lock（任何 GPU outcome 之前）

每路必须先在自己的 `notes/` 产出：

1. `proposal.md`：Observation → Architecture consequence → Falsification；
2. `mechanism_fingerprint.md`：数学 operator、干预层、状态表示、训练信号、
   推理输入、复杂度、退化版本；
3. `nearest_neighbors.md`：基于原始论文/官方代码的最新检索，至少覆盖
   DIN、SIM/UBR、ZAM/TEM/RTM、SASRec/BERT4Rec、现代 LLM4Rec，以及本路最
   接近的专门机制；不得凭名字宣称 novel；
4. `gate_protocol.md`：数据、hash、seed、阈值、controls、预算、stop-loss、
   evaluator call 数和预期失败解释；
5. `proposal_lock.json`：上述文件 SHA256、git commit/dirty、时间、environment、
   GPU、candidate ID，并声明尚未读取本路 dev outcome；
6. `README.md`、candidate-local configs、tests 和最小执行命令。

若原始文献已经覆盖核心 operator，必须在读取 dev outcome 前于本路搜索轴内
pivot，重新生成 lock；不能只换名字。四路提交后由主协调者比较 mechanism
fingerprint；可通过变量改名相互归约的设计视为碰撞，至少一路必须在 GPU gate
前 pivot。

## 5. 分阶段 GPU 预算与授权

### A. Design / unit / train-internal smoke（已授权）

- 只实现回答本路 falsifier 所需的最小 Transformer/LM path；
- 可以使用 train labels 做训练/internal validation；dev records 保持 label-free；
- 每路最多 2 个实现尝试（包含 debug retry），累计不超过 8 A40 GPU-hours；
- 不得做无界超参搜索、三 seed 完整训练或跨数据集扩展。

### B. 单 seed dev screening（proposal lock 后已授权）

- 固定 seed `20260708`；
- 每路最多 1 次 primary dev evaluator call；
- screening 前冻结 scores contract、candidate hash、config 与 run ID；
- dev evaluator 必须串行：

```bash
flock tmp/pps_dev_evaluator.lock \
  python scripts/evaluate_scores.py \
  --run-id <assigned-prefix><purpose> \
  --candidate-manifest data/standardized/kuaisearch/v0_lite/candidate_manifest.json
```

### C. Full design gate（screening 幸存后，需主协调者登记预算）

完整 gate 至少验证：

1. repeat-present 不得弱于 item-only control；
2. 4,677 non-repeat history-present 请求上相对 D2p 有稳定正值；
3. wrong-user、shuffled-event、query-masked、coarse-only evidence 不得复现；
4. 4,110 no-history 请求逐请求 score/rank/metric 等价于 D2p contract；
5. matched-capacity backbone 与本路最近邻退化版不能解释同等增益；
6. 1000-request deterministic rescore 逐值一致。

具体统计阈值必须由每路在 outcome 前冻结。C01--C04 的专属干预分别是：

- C01：counterfactual certificate/null contract 必须比 plain target attention 多
  提供可证伪价值；正式推理只能输入真实 `(q,c,H)`；
- C02：candidate-conditioned internal `Delta W` 必须超过 ordinary LoRA、output
  gate 与 mean-history residual；no-history 时 `Delta W=0`；
- C03：triadic cycle/null-sink 必须优于 matched softmax target attention，破坏
  query/history 后 evidence mass 应进入 null；
- C04：同一 LM 的 history/no-history internal candidate-logit delta 必须超过
  matched single-pass LM；不得实现成外部 `D2p + delta` 固定混分。

### D. Full implementation/training（当前未授权）

只有某一路完整 gate 通过且主协调者写出显式授权后，才可运行多 seed、dev tuning、
完整 ablation、Amazon/JD 扩展。未过 gate 的 candidate 必须停止；不得追加模块
救场或借 sibling 结果改 premise。

## 6. 输出与选择

每路结束时只在自己的 `notes/final_report.md` 汇报：

- proposal/gate lock hashes；
- source/config/test 文件；
- environment、GPU、commands、run IDs、GPU-hours；
- unit/smoke/determinism 结果；
- dev-eval call 数与日志行；
- common gate 与 track-specific gate 的逐项结论；
- nearest-neighbor novelty verdict：`distinct` / `reducible` / `uncertain`；
- `advance` / `pivot-before-more-dev` / `stop` 建议。

主协调者在四路 lock 和 screening 都完成后统一盲评。进入 full gate 的必要条件是：

1. protocol/integrity 全部通过；
2. 是合格 LLM4Rec/Transformer，且 mechanism fingerprint 与 sibling/最近邻不可
   归约；
3. screening 不显示明显退化，并满足本路预注册 stop-loss；
4. 预算、延迟和显存可接受。

不按 best-dev 数字直接拼装四路。若全部失败，保留四个负结果并回到新问题定义；
不得访问 test 挽救设计。

## 7. 对应 prompts

- `doc/design_prompts/20260710_01_counterfactual_evidence_contract_prompt.md`
- `doc/design_prompts/20260710_02_history_hyperadapter_prompt.md`
- `doc/design_prompts/20260710_03_triadic_transport_transformer_prompt.md`
- `doc/design_prompts/20260710_04_prefix_delta_lm_prompt.md`

四份 prompt 自包含地重复共同红线，但分别限定不同 architecture locus；共同规则
保证公平，专属机制约束保证独立创新。
