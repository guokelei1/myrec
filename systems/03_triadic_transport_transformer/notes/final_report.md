# C03 Final Report — Candidate-Anchored Cycle-Intersection Transport

日期：2026-07-11
最终建议：**`stop`**
Primary dev evaluator calls：**0 / 1**
Test：**保持锁定，未读取**

## 1. 结论

C03 没有在预注册的 8 A40 GPU-hour 上限内完成 screening。冻结训练在 GPU 2 上运行约 7 小时 25 分后仍未产生 checkpoint；计入特征准备，保守累计 GPU wall-time 上界约为 **7.456 小时**。根据已经观测到的吞吐，剩余预算不足以完成训练、内部诊断、1000-request 两次确定性复算和 575,609 个 dev candidate 的 label-free scoring，因此在越过硬上限前主动终止。

这是一个**预算/复杂度 gate 失败**，不是 dev 排名效果的正面或负面结论。没有生成 C03 scores，没有调用共享 evaluator，也没有 C03 dev log 行。不得把“未完成”表述为 non-repeat 无效；可以成立的判断只有：当前三边严格传输及其训练/诊断实现不具备可接受的 falsifier 性价比。

## 2. 冻结机制

共享 Transformer 对 `[Q, H_1, ..., H_m, C]` 编码。三条带 learned dustbin 的部分传输产生 query-history、history-candidate 和 query-candidate 的真实质量：

```text
a_j = (m + 1) P_qh[Q, H_j]
b_j = (m + 1) P_hc[H_j, C]
d_qc = 2 P_qc[Q, C]

u_j = sqrt(a_j b_j)
Delta_cycle = sum_j |a_j - b_j| / (sum_j a_j + sum_j b_j + eps)
g_j = d_qc * exp(-lambda_cycle * Delta_cycle) * u_j
t = sum_j g_j
null = 1 - t
```

只有 `g_j` 的交集质量可以更新 candidate state：

```text
h_bar = sum_j (g_j / (t + eps)) h_j
c_plus = c + t W_o h_bar
r_raw = t * (s(q, c_plus) - s(q, c) + softplus(b_mass))
r_i = gamma * (r_raw_i - mean_k r_raw_k)
score_i = D2p_i + r_i
```

exact item identity 只作为 `h↔c` cost 内的受保护正原子；没有独立 exact score/channel。无历史时 mask 代数保证 `r_raw = r_i = 0`，最终分数严格等于 D2p。

参数相同的冻结退化控制为 `softmax`、`no_null`、`no_cycle` 和 `mean_pool`。专属 falsifier 是：真实 non-repeat 证据应保留三边交集质量；wrong-user、shuffle、query-mask 和 coarse-only 应将质量推向 null；近邻控制不能复现同等行为。

## 3. 新颖性与最近邻结论

全局新颖性 verdict 维持 **`uncertain`**。generic Sinkhorn、dustbin、cycle consistency、多重/多边 OT 和 Transformer recommendation 都已有直接先例。最初的“多分布 Sinkhorn + generic cycle loss”在任何 C03 outcome 前因 Optimal Multiple Transport / multi-marginal OT 近邻而被放弃；冻结后可测试的增量仅是“candidate-anchored partial plans 的 non-null 交集是唯一 history-to-logit 通道”。

因此，即使未来在新 protocol 下有效，也不能把“OT 用于推荐”或“dustbin matching”作为论文创新。详细对照见 [nearest_neighbors.md](nearest_neighbors.md)。

## 4. 实现、环境与数据边界

- 源码：`model/triadic_transport.py`、`train/`、`configs/`、`tests/`。
- 环境：`myrec-c03`；Python 3.10.20；PyTorch 2.6.0+cu124；Transformers 5.12.1；pytest 8.3.5。
- 设备：仅 physical GPU 2；`CUDA_VISIBLE_DEVICES=2`；程序设备 `cuda:0`；NVIDIA A40。
- Seed：`20260708`。
- Candidate manifest SHA256：`94eb667000e0d0f389d0a2a4d4730683b71c129043edbfcf627590376e9c123e`。
- Proposal-lock candidate hash：`7ea1a275bcf2c1fb6e1a7043b55d278e8e5558d006573191faad86303ef509ca`。
- 锁中的逐文件 SHA256 已在终止后重新核对，全部未变；本报告和阶段总结属于 lock 后报告材料，不修改冻结机制、config 或 thresholds。
- 训练只打开 `records_train.jsonl`；dev 特征只打开 label-free `records_dev.jsonl`。candidate code 没有读取 qrels、test records、test qrels 或 test metrics。
- 未读取其他候选的 prompt、设计或代码。

准备结果：

- 12,000 个 train requests 在读标签前按 request-ID hash 固定；10,761 fit、1,239 internal validation。
- train records SHA256：`2c4823c3117cb11e89052b6563ed8fb40c7113400be933f01ce9db51aea9bcd8`。
- label-free dev records SHA256：`caa376ccad79df7b96fe945d1f5f922570a2977f7edaa728d20046751b814533`。
- dev：12,229 requests、575,609 candidate rows；没有读取 labels/qrels。

## 5. 实现尝试与执行记录

### Attempt 1：数值实现失败，冻结前修正

普通有限轮 log-Sinkhorn 在严格守恒测试中的最大误差约为 `3.4e-4`，不满足 gate。由于 C03 的实际 pairwise plan 总有 singleton 侧，改为对同一熵正则缩放方程求 differentiable scalar Newton root。该修正不改变冻结机制，并使 float64 单测达到 `<= 1e-10`。

### Attempt 2：冻结训练触发预算 stop-loss

```bash
CUDA_VISIBLE_DEVICES=2 CONDA_ENVS_PATH=/data/gkl/conda_envs \
  conda run -n myrec-c03 python \
  systems/03_triadic_transport_transformer/train/run_probe.py \
  --config systems/03_triadic_transport_transformer/configs/c03_screening.yaml prepare

CUDA_VISIBLE_DEVICES=2 CONDA_ENVS_PATH=/data/gkl/conda_envs \
  conda run -n myrec-c03 python \
  systems/03_triadic_transport_transformer/train/run_probe.py \
  --config systems/03_triadic_transport_transformer/configs/c03_screening.yaml train
```

- Feature preparation：123.908 秒。
- Frozen training：约 26,717 秒上界；在 checkpoint 前手动中止。
- 累计 assigned-GPU wall-time 上界：26,840.908 秒，即 7.456 小时。
- 观测资源：约 0.9 GB 显存、23%--25% GPU utilization；主要开销来自逐 candidate 的主前向加四个 corruption 前向和三条 transport 求解。
- Checkpoint：无。
- Reserved run ID：`20260710_kuaisearch_c03_tctt_screen_s20260708`；run dir/scores 未生成。
- 原始停止证据：ignored run state 中的 `outputs/c03_screening/attempt_2_stop_report.json` 与 `gpu_ledger.json`。

不进行第三次实现/debug retry；不修改 epoch、subset、threshold 或 gate 来挽救结果。

## 6. 测试、mass/null 与 gate 审计

终止后复核命令：

```bash
CONDA_ENVS_PATH=/data/gkl/conda_envs \
  conda run -n myrec-c03 pytest -q \
  systems/03_triadic_transport_transformer/tests
```

结果：**8 passed**。覆盖：手算 1x1 plan、矩形质量守恒与 null path、exact cost 单调性、padding mask、无历史严格零残差、trusted/null 有界、有限梯度、softmax 强制分配以及 request-centered residual。float64 plan marginal error 的测试上界为 `1e-10`，`trusted_mass + null_mass = 1` 的误差上界为 `1e-12`。

| Gate | 状态 | 证据/原因 |
|---|---|---|
| Proposal/threshold lock before C03 dev outcome | PASS | `proposal_lock.json`；锁文件 hash 复核一致 |
| Numerical/unit gate | PASS | 8/8 tests |
| Separate tiny smoke config | NOT RUN | 未把未冻结的 tiny outcome 当作 screening 证据；长训练过程保持 finite，但未写 checkpoint，不能记为 smoke pass |
| Candidate hash / assigned GPU / run prefix | PASS | prepare/train 入口均 assert；只使用 GPU 2 |
| Frozen feature preparation | PASS | 完整 train/dev text store；dev label-free |
| Train-internal checkpoint | **FAIL — BUDGET STOP** | 7h25m 后仍无 checkpoint |
| Learned mass/null corruption diagnostics | NOT RUN | 无 checkpoint |
| Matched softmax/no-null/no-cycle/mean-pool diagnostics | NOT RUN | 无 checkpoint |
| 1000-request label-free dev corruption | NOT RUN | 无 checkpoint |
| 1000-request byte-identical deterministic rescore | NOT RUN | 无 checkpoint |
| Full label-free dev scoring | NOT RUN | 预算不足；无 scores |
| Primary dev evaluator | **0 calls** | 没有合法 scores；`reports/dev_eval_log.jsonl` 无 C03 行 |
| Repeat-present vs item-only | NOT ASSESSED | 无 evaluator outcome |
| 4,677 non-repeat vs D2p | NOT ASSESSED | 无 evaluator outcome |
| 4,110 no-history full score/rank equality | NOT ASSESSED | unit algebra通过；未做 full score audit |
| Test lock / qrels isolation | PASS | candidate training/scoring 未读取 qrels/test；test 完全未触碰 |

没有 learned checkpoint，因此不能报告训练后的 trusted mass、null increase、corruption score drop 或 ranking delta。不得用 unit-level conservation 代替这些科学诊断。

## 7. 失败含义与建议

本轮最可迁移的结论是：**fail-closed 是正确的系统合同，但“严格三边传输”不是一个已经付得起复杂度租金的实现形式。** 三条相似度的交集/连乘既可能压垮信号，又要求多次逐 candidate 计算；corruption objective 还可能只把 null diagnostic 教出来，而不产生 non-repeat 排名价值。

后续设计应保留三件事：证据不足时可拒绝、repeat 不退化、no-history 严格回到共同 base；同时应：

1. 将安全性 gate（null/corruption/no-history）与有效性 gate（non-repeat 正增益及置信区间）分开；
2. 避免多个脆弱相似度通过唯一乘法通道汇合；
3. 在正式训练前用真实 request 做端到端吞吐外推，覆盖 corruption、determinism 和 full scoring，而不只 benchmark 单次 forward；
4. 用信息流等价的近邻控制判断复杂机制是否退化为 target attention/gating；
5. 若重新研究该原则，必须作为新候选重新冻结，不继承本轮 dev budget，也不能把本轮无 outcome 当作调参依据。

最终状态：**`stop`**，不进入 full gate，不请求 multi-seed、test 或跨数据集预算。
