# 14 - Batch 2b：B4/B5/B6 官方 baseline 接入计划

状态：执行计划。前置：doc 11（数据与实验计划）、doc 12（执行协议）、
doc 13（baseline 实现计划）、`reports/pps_batch2_decision_summary.md`。

## 0. 为什么有这个批次

Batch 2 的 B4/B5/B6 是 hashed-logistic 简化 adapter，全部显著低于 B0b/B7。
这些负结果只能证明"该简化实现弱"，不能支撑论文里"现有个性化方法在此
setting 下不行"的 motivation 表述——审稿人会认定为 strawman。

Batch 2b 的目标是把三个占位 adapter 升级为可辩护的实现：

| ID | 目标实现 | 升级后可支撑的声明 |
|---|---|---|
| B4o | RecBole 官方 SASRec（可选加 BERT4Rec） | "官方实现的强序列 baseline 在此 setting 下的真实水平" |
| B5o | KuaiSearch 官方 DIN/DCNv2 pipeline | "该数据集原作者的工业 ranking baseline 的真实水平" |
| B6o | HEM/ZAM/TEM（官方代码或经外部基准验证的 faithful 复现） | "PPS 经典方法在此 setting 下的真实水平" |

**结论层级（先写下来，防止事后按结果改叙述）**：

- 若 B4o/B5o/B6o 仍不超过 B7-best（0.3305）：motivation 升级为
  "包括官方实现在内的现有个性化方法不超过简单静态混合，而 per-request
  headroom 仍有 +28%" —— 这是"现有系统做得不好"的可辩护版本。
- 若某官方 baseline 超过 B7-best：它成为新的 baseline-to-beat，motivation
  回到 "最强现有方法之上仍有 headroom" 的表述。两种结局论文都成立，
  区别只是措辞强度。**不允许因为结果不利于某种叙述而废弃合格 run。**
- 无论哪种结局，提议系统的主对比表必须包含 Batch 2b 的合格 run。

---

## 1. Phase 0：预算修订（开跑前必须完成，先例：`reports/pps_c2_gate_amendment.md`）

doc 13 §2.5 预算在 Batch 2 开跑前已冻结，因此 Batch 2b 需要一次显式修订，
不允许默默复用或默默新开预算：

1. 写 `reports/pps_batch2b_budget_amendment.md`，内容至少包括：
   - B4o/B5o/B6o 是**新 method ID**，各自获得全新 trainable 预算
     （16 次 dev 评测/方法，官方默认超参算第 1 次）；
   - 旧 B4/B5/B6 adapter 的 run 与结果行**保留不删**，状态改为
     `retired placeholder (superseded by B4o/B5o/B6o)`，论文中只可出现在
     appendix/实现说明，不进主表；
   - 外部验证 run（§3–§5 中在 RecBole 自带数据集、官方 KuaiSearch 格式数据、
     Amazon PPS 基准上跑的对齐性检查）**不读本项目 dev split，不计入
     dev 评测预算**，但每次都要在对应 baseline card 登记；
   - 提议系统未来仍执行同等 16 次预算（07 §9 对称性不变）。
2. 修订文档 commit 后，才允许产生第一次 Batch 2b dev 评测。
3. `experiments/pps_baseline_cards.md` 新增 B4o/B5o/B6o 三张 card
   （状态 = in progress），沿用 doc 13 §5 模板。

---

## 2. 所有方法共同边界（继承，不重新发明）

1. 输入只来自 `data/standardized/kuaisearch/v0_lite/`；candidate manifest
   不变（`94eb667000e0d0f389d0a2a4d4730683b71c129043edbfcf627590376e9c123e`），
   evaluator assert 不变。
2. **qrels 红线不变**：训练/打分代码读 qrels = 该方法全部 run 作废。
3. 公平性矩阵沿用 doc 13 §2.4 的 B4/B5/B6 行，逐字段适用于 B4o/B5o/B6o：
   - B4o：query ✗；history ✓（item_id 序列）；candidate 文本 ✗；
     candidate item_id ✓；train 标签 ✓；dev/test 标签 ✗；
   - B5o / B6o：query ✓；history ✓；candidate 文本 ✓；candidate item_id ✓；
     train 标签 ✓；dev/test 标签 ✗。
4. **推理时的历史 = 该请求 record 里冻结的 ≤50 条 history**（doc 11 §1.2）。
   即使官方代码惯例是"取该用户全量训练序列"，也必须改为喂入当前 record 的
   history 字段——这是所有方法可比的前提，差异写进适配说明。
5. 训练交互的构造统一为（三个方法共用同一个导出脚本，进 `src/myrec/`）：
   - 来源 A：`records_train.jsonl` 每条 record 的 history 事件
     （item_id, event_type, event_time）；
   - 来源 B：`records_train.jsonl` 的 clicked/purchased 候选，作为发生在
     request_time 的正交互；
   - 按 (user_id, item_id, event_time) 去重，升序排列；
   - **禁止**使用 dev/test records 的任何字段构造训练数据；
   - 导出物落盘 `artifacts/batch2b/interactions_train.<fmt>`，记录行数、
     unique user/item 数和 SHA256，写进各方法 metadata。
6. 打分输出仍是全候选覆盖的 `scores.jsonl` + 共享 evaluator，doc 13 §1 不变。
7. 冻结 config 后 3 seeds（doc 12 §6），报 mean±std，登记最优 seed run。
8. 显著性一律用共享 compare 脚本（paired bootstrap 95% CI，doc 11 §1.4）。
   每个方法的固定对照集：vs Random（sanity）、vs B0b（history-only 代表）、
   vs B7-best（当前最强）。全量 dev 对比，不需要 B8 式抽样。
9. run_id 命名：`YYYYMMDD_kuaisearch_<b4o|b5o|b6o>_<impl>_dev_s<seed>`。

---

## 3. B4o：RecBole SASRec / BERT4Rec

### 3.1 环境（先解锁，再谈别的）

已知阻塞：`recbole==1.2.1` 依赖 `ray<=2.6.3`，无 cp313 wheel。解决方案按
doc 12 §1 的 `pps-recbole` 环境组：

```bash
conda create -n pps-recbole python=3.10 -y
conda activate pps-recbole
pip install recbole==1.2.1 torch --index-url <按机器 CUDA 版本>
```

要求：

- 依赖清单落盘 `configs/env/recbole.txt`（`pip freeze` 输出），进 git；
- metadata 记录 python=3.10.x、recbole/torch/ray 版本；
- 若 conda 不可用，用 `python3.10 -m venv`；若机器无 python3.10，
  用官方 pytorch docker 镜像——三选一，选哪个写进 card。
- **禁止**为了迁就 py3.13 改 RecBole 源码跑"魔改版"；那等于又造一个 adapter。

### 3.2 环境 sanity（不计 dev 预算）

正式接数据前，先在 RecBole 自带 ml-100k 上跑通 SASRec 默认配置，确认
loss 下降、评测流程完整。目的只是证明环境可用，结果不登记 results.md，
但在 card 的 Known limitations/setup 里记一行。

### 3.3 数据 adapter（写入 `src/myrec/baselines/recbole_adapter.py` 或同级）

1. 把 §2.5 的统一训练交互导出为 RecBole atomic 文件
   （`.inter`：user_id、item_id、timestamp）；
2. item token 空间 = 训练交互中出现过的 item ∪ 全部 candidate item_id
   出现在训练交互中的部分。**先跑一次统计并写进 card**：unique item 数、
   dev 候选中 in-vocab 的覆盖率（catalog 有 297 万 item，全表 embedding
   不可行也不必要——只需交互过的 item）；
3. RecBole 内部 split 设为不再切分（训练用全部 train 交互；我们自己的
   dev/test 不进 RecBole）；early-stop 如需 valid set，只准从 train 交互
   内部按时间切出 tail，并在 config 声明比例。

### 3.4 候选打分 adapter

1. 对每条 dev record：输入 = 该 record 冻结 history 中 in-vocab 的
   item_id 序列（保序、截断规则与训练一致）；
2. score(candidate) = 模型对该 candidate item 的下一交互得分
   （full-sort 得分表中取候选对应列，不做私自过滤）；
3. **冷启动策略（冻结进 config）**：candidate 不在 vocab → score 取
   该请求内 in-vocab 候选最低分减一个固定 margin（保证排最后且可复现）；
   history 全部 out-of-vocab → 所有候选同分 + 诊断字段（与 B0b 同规则）；
4. `scores.jsonl` 附加诊断字段：`in_vocab`（bool）、`history_in_vocab_len`；
5. evaluator 照常评全候选。冷启动候选占比写进 card——如果占比过高
   （>30%），先停下来在 card 记录并评估是否需要在 §3.3 扩 vocab 策略，
   这算实现修正不算调参，但要记 dev_eval_log 备注。

### 3.5 调参与验收

- 预算 16 次 dev 评测。第 1 次 = RecBole SASRec 官方默认超参。搜索空间
  （hidden size、n_layers、n_heads、dropout、lr、max_seq_len、loss 类型）
  先写进 `configs/baselines/b4o_sasrec_recbole.yaml` 再开跑；
- BERT4Rec 可选：若做，作为同一 card 内第二实现，共享同一预算池，
  不另开 16 次；
- 冻结后 3 seeds；
- 验收：
  1. query-blind 核对（代码路径中无 query 字段）；
  2. B4o 显著高于 Random；
  3. 与 B0b、B7-best 的 compare 结果登记（无论方向）；
  4. **合格判据不是"必须赢 B0b"**，而是：官方实现 + 预算用满或搜索空间
     覆盖声明 + 3 seeds 稳定（std 与 adapter 同量级）。官方实现输给 B0b
     本身就是可写进论文的合法结论；
  5. card 记录 RecBole 版本、config 快照、负采样设置、候选打分方式、
     冷启动占比。

---

## 4. B5o：KuaiSearch 官方 DIN/DCNv2

来源锁定：`https://github.com/benchen4395/KuaiSearch`
commit `7ce0471b659112096f0aa7e892ed0aa4c972246a`（与 Batch 2 card 一致）。
环境组 `pps-kuaisearch`（python 版本以官方 repo 要求为准，依赖清单落盘
`configs/env/kuaisearch.txt`）。

已知阻塞：官方 ranking pipeline 需要 (a) 预计算 query/title embeddings，
(b) 标准化接口之外的原始用户特征。分两阶段处理：

### 4.1 阶段 A：官方复现对齐（不计 dev 预算）

在官方 repo 自带的数据准备流程/数据上，跑通官方 DIN（或 DCNv2）训练与
评测，与官方论文/README 报告的指标对齐到 **±10%**（doc 13 §3 B5 的 C2
标准）。产出 `reports/b5o_official_alignment.md`：官方指标、我方复跑指标、
差异、环境。

**止损规则**：若官方 repo 缺数据/缺训练入口/缺可对齐的官方数字，导致
阶段 A 无法完成，写 protocol-diff report 后**降级处理**——B5o 改标
`official-code, alignment-not-verifiable`，论文中作为 secondary baseline
放 appendix，不作为主表"官方复现"声明。不允许带着不明差异硬标官方。

### 4.2 阶段 B：接入我方数据

1. 只在 adapter 层做格式转换（doc 13 §3 B5 原则不变）：
   - query/title embeddings：用**冻结的公开文本编码器**从 standardized
     records/item_catalog 的文本生成。默认 `BAAI/bge-small-zh-v1.5`
     （与 B2z 同款，保证语义通道口径一致）；模型名、权重版本、截断长度
     写进 config；embedding 缓存 `artifacts/batch2b/`，记 hash；
   - 用户特征：只允许使用 standardized record 中存在的字段
     （user_id、history item/category/event/time）。官方 pipeline 需要
     但 record 中不存在的字段，一律置官方默认值/缺省桶，并在
     `reports/b5o_protocol_diff.md` 逐项列出"官方要求 vs 我方提供"；
2. 训练用 §2 统一交互 + train clicked 标签，按官方代码的样本构造方式
   组 batch；dev 打分时 history 只用当前 record 冻结 history（§2.4）；
3. 输出 fixed candidates 的 score → `scores.jsonl` → 共享 evaluator，
   禁用官方私有 evaluator 出论文数字；
4. 预算 16 次 dev 评测，第 1 次 = 官方默认超参；冻结后 3 seeds。

### 4.3 验收

1. 阶段 A 对齐报告存在（或降级声明存在）；
2. protocol-diff report 存在且逐字段核对过公平矩阵；
3. 显著性对照集（vs Random / B0b / B7-best）登记；
4. card 记录官方 commit、patch diff（如有，`git diff` 落盘）、license。

---

## 5. B6o：HEM / ZAM / TEM

这是三个中历史包袱最重的：原方法为 Amazon review-based product search
设计，官方实现为老版本 TensorFlow。two-path，按顺序尝试：

### 5.1 Path 1：官方代码直跑（优先）

1. 定位原作者官方实现（HEM: Ai et al. SIGIR'17；ZAM: Ai et al. CIKM'19；
   TEM: Bi et al. SIGIR'20；三者共享同一作者组的代码族）。repo URL 与
   commit 写进 card；
2. 环境组 `pps-classic`，允许装老版本 TF（py3.7/3.8 conda env 或 docker）；
3. 阶段 A（不计 dev 预算）：在论文使用的 Amazon 子集上复跑其中至少一个
   方法，与论文数字对齐 ±10%，产出 `reports/b6o_official_alignment.md`；
4. 阶段 B：adapter 接入我方数据（映射规则见 §5.3）。

### 5.2 Path 2：faithful 复现（Path 1 环境不可行时）

1. 按论文公式在 PyTorch 重写（HEM 必做；ZAM/TEM 至少再选一个）；
2. **验证门槛**：复现版必须先在公开 Amazon PPS 基准子集上达到论文数字
   ±10%（不计 dev 预算），才允许接我方数据。达不到就不接，B6o 标记
   blocked 并保留为 related work（doc 13 §3 B6 原则）；
3. card 的 implementation type 如实写 `faithful reimplementation
   (validated on Amazon ±10%)`，论文同样如实标注。**通过外部基准验证的
   复现是可辩护的；未经验证的复现和现在的 adapter 没有区别。**

### 5.3 字段映射（两条 path 共用，写进 config 与 protocol-diff）

| 原方法需要 | 我方提供 | 备注 |
|---|---|---|
| query | record.query | 真实 query，无需构造 |
| item text | title + brand + category（B1 document 模板） | 与其他文本方法同模板（§2.4 补充规则） |
| user 购买/评论序列 | 冻结 history 的 click/purchase 事件 | 无 review 文本；HEM 的 review-based user 表示退化为 history-item 表示，此差异写进 card |
| 训练正例 | train clicked/purchased 候选 | 同 §2 统一交互 |

### 5.4 预算与验收

- 16 次 dev 评测（HEM/ZAM/TEM 共享同一预算池，各自第一次跑 = 论文默认
  超参）；冻结后 3 seeds；
- 验收：外部对齐报告（或 blocked 声明）+ 三个固定对照 compare +
  每个子方法的适配差异说明（doc 13 §2.6 B6 行）。

---

## 6. 优先级与止损（预算按 dev 评测次数计，不按时间计）

执行顺序（一个做完验收再开下一个，避免三线并行烂尾）：

1. **B4o**（最便宜、阻塞最明确：装个 py3.10 环境）；
2. **B6o**（对 PPS 论文定位最关键：审稿人默认要求和 HEM 系比较）；
3. **B5o**（最难对齐；且它与提议系统的证据通道重叠度最高，缺席的
   叙述代价最小）。

全局止损：

- 任一方法的阶段 A（外部对齐）失败 → 按各自小节降级规则处理，
  **不因此阻塞其他方法**；
- 三个方法中**至少 B4o + B6o 之一必须达到"官方或验证过的复现"状态**，
  Batch 2b 才算达成 motivation 升级目标；三个全部降级 = 回到 Batch 2
  的措辞边界（只谈 query 饱和 + headroom，不谈"现有方法不行"）；
- Batch 2b 全部收尾后，重跑一次 M3 三通道 oracle 把合格的新方法纳入
  channel 候选（`best_history_only` 可能换人），沿用 doc 11 M3 噪声护栏；
  多通道 oracle 仍只作 analysis-only。

## 7. 交付物（每方法，doc 13 §2.6 全套 + Batch 2b 附加）

| 产出 | 路径 |
|---|---|
| config（含搜索空间） | `configs/baselines/b4o_*.yaml` 等 |
| 环境清单 | `configs/env/recbole.txt` / `kuaisearch.txt` / `pps_classic.txt` |
| 统一训练交互 + hash | `artifacts/batch2b/`（hash 进 metadata） |
| 外部对齐报告 | `reports/b4o_env_sanity.md`（并入 card 亦可）、`reports/b5o_official_alignment.md`、`reports/b6o_official_alignment.md` |
| protocol-diff | `reports/b5o_protocol_diff.md`（B6o 的写进 card 适配差异节） |
| run 全套 | `runs/<run_id>/`（scores/metrics/per_request/metadata，3 seeds） |
| compare 结果 | vs Random / B0b / B7-best，共享 compare 脚本输出 |
| card 更新 | `experiments/pps_baseline_cards.md` |
| 结果行 | `experiments/pps_results.md`（新行，不覆盖旧 B4/B5/B6 行） |
| 决策摘要 | `reports/pps_batch2b_decision_summary.md`（按 §0 结论层级写措辞） |

## 8. 开发者执行 checklist（按图索骥）

```text
step 0  写并提交 reports/pps_batch2b_budget_amendment.md（§1）；
        三张新 card 登记为 in progress。
step 1  实现统一训练交互导出脚本（§2.5），落盘 + hash + 单测
        （不含任何 dev/test 字段的 assert）。
step 2  B4o：建 pps-recbole py3.10 环境 → ml-100k sanity → 数据 adapter
        → 打分 adapter（含冷启动策略）→ 默认超参第 1 次 dev 评测。
step 3  B4o：预算内调参（每次自动记 dev_eval_log）→ 冻结 → 3 seeds →
        复跑确定性检查（doc 12 §5）→ compare → card/results 登记。
step 4  B6o：Path 1 环境探测；不可行则 Path 2。先过外部 ±10% 门槛，
        再接我方数据；随后同 step 3 流程。
step 5  B5o：阶段 A 官方对齐（或降级声明）→ 阶段 B adapter →
        同 step 3 流程。
step 6  重跑 M3 三通道 oracle（合格新方法进 channel 候选）。
step 7  写 reports/pps_batch2b_decision_summary.md：对照 §0 结论层级，
        明确论文可用的 motivation 措辞；更新旧 B4/B5/B6 卡片状态为
        retired placeholder。
```

## 9. 红线（触发任一条 = 该 run 作废，card 标记并重跑）

继承 doc 13 §7 全部红线，另加 Batch 2b 特有：

- 训练交互混入 dev/test records 的任何字段；
- 推理时用用户全量训练序列替代 record 冻结 history 而未在 card 声明；
- 外部对齐未通过（也未走降级流程）却在论文中标注 official/faithful；
- 为跑通官方代码修改共享 evaluator 或 candidate manifest；
- 修订文档（§1）提交前产生的任何 Batch 2b dev 评测。
