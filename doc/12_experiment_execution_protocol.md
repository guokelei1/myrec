# 12 - 实验执行与资源隔离协议

状态：执行协议。前置：`11_experiment_and_dataset_plan.md` 定义任务、数据、
baseline 和 checkpoint；本文只规定实验如何在本机可复现、可并行、可审计地运行。

目标：

1. 不同 baseline 的依赖互不污染；
2. 多张 GPU 可以并行探索，但任何 run 都有唯一输出边界；
3. 评测只读统一 score 文件和 candidate manifest，不读方法私有状态；
4. 每个重要结果都能追溯到代码版本、环境、GPU、配置、数据 hash 和随机种子。

---

## 1. 环境分组

每个 baseline 只允许在所属环境组中运行。不同组可以用不同 conda/venv/Docker，
但组内脚本必须从同一个统一 JSONL 接口读取数据。

| 组 | 建议环境名 | 覆盖方法 | 说明 |
|---|---|---|---|
| core | `pps-core` | 数据转换、审计、指标、B0a/B0b/B1/B7 | 默认环境，必须最稳定 |
| embed | `pps-embed` | B2 dense bi-encoder、B3 cross-encoder scoring | 可安装 torch/transformers/sentence-transformers/faiss |
| recbole | `pps-recbole` | B4 SASRec/BERT4Rec | RecBole 依赖隔离，避免污染 core |
| kuaisearch | `pps-kuaisearch` | B5 DIN/DCNv2 官方代码 | 以官方 repo 依赖为准，适配层在本仓库记录 |
| pps_classic | `pps-classic` | B6 HEM/ZAM/TEM/MAI-style | 老代码依赖单独隔离 |
| llm | `pps-llm` | B8 raw-history LLM、MemRerank-style | 大模型推理和缓存单独管理 |

规则：

- 任何 run 的 `metadata.json` 必须记录环境组、环境名、Python 版本、关键包版本。
- 依赖文件优先放在 `configs/env/`，例如 `configs/env/core.yml` 或
  `configs/env/recbole.txt`。机器本地路径、token、HF cache 路径不进 git。
- 上游 baseline 若需要修改依赖，修改原因写入对应
  `baselines/<baseline_name>/README.md` 或 manifest。
- 不允许为了跑通某个 baseline 修改统一 evaluator；只能改 adapter 或 score 导出。

---

## 2. GPU 与并行策略

GPU 只负责加速方法运行，不改变实验协议。所有方法必须在同一个 candidate manifest
上导出分数，再由同一个 evaluator 评测。

### 2.1 设备绑定

每个 GPU run 必须显式绑定设备：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/<command>.py --config ... --run-id ...
```

脚本内部统一使用可见设备里的 `cuda:0`。不要在脚本里硬编码物理 GPU 编号。

### 2.2 并行粒度

| 类型 | 默认资源 | 是否适合并行 | 例子 |
|---|---|---|---|
| CPU/light | CPU | 可以大量并行 | B0a、B0b、B1、B7、指标测试、canary |
| embedding scoring | 单 GPU | 可以每卡一个 run | B2、B3 的 dev/test scoring |
| trainable baseline | 单 GPU | 每卡一个 run，优先固定 seed | B4、B5、B6 |
| LLM rerank | 独占 GPU 或独占服务 | 单独排队 | B8、MemRerank-style |
| shared evaluator | CPU | 可以并行，但必须只读 | `evaluate_scores.py` |

多卡机器上的建议原则：

- 不做隐式多卡训练，除非某个 baseline 官方实现必须如此并在 run summary 中说明。
- 同一张 GPU 同时最多跑一个显存占用大的训练或 LLM 推理任务。
- 轻量 baseline 可以和 GPU 任务并行，但不得写入同一个 `runs/<run_id>/`。
- dev exploration 可以多 run 并行；test 只在冻结配置后运行一次。

### 2.3 推荐排队顺序

Phase 0-2 的早期并行顺序：

1. CPU：C0/C1 审计、split、candidate manifest、指标单测；
2. CPU：B0a/B0b/B1/B7；
3. GPU：B2 zero-shot embedding scoring；
4. GPU：B4/B5/B6 训练型 baseline；
5. GPU/LLM：B8 和 MemRerank-style，只在 dev 抽样上做成本-质量曲线。

---

## 3. Run ID 与目录边界

所有 run 使用统一 ID：

```text
YYYYMMDD_<dataset_id>_<method_id>_<short_purpose>
```

例子：

```text
20260708_kuaisearch_bm25_motivation_m1
```

每个 run 只能写入：

```text
runs/<run_id>/
```

建议结构：

```text
runs/<run_id>/
  command.sh
  metadata.json
  config_snapshot.yaml
  stdout.log
  stderr.log
  scores.jsonl
  metrics.json
  run.lock
```

规则：

- `run.lock` 存在且进程仍存活时，脚本必须拒绝复用该 `run_id`。
- 复跑同一设置时不要覆盖旧目录，使用新的 short purpose 或追加 `r2`。
- `scores.jsonl` 是方法输出的唯一评测入口；字段至少包含
  `request_id`、`candidate_item_id`、`score`、`method_id`。
- `metrics.json` 只能由共享 evaluator 生成，不由方法脚本自己写主指标。
- 原始日志、score dump、checkpoint、cache 都留在 `runs/`、`models/` 或
  `artifacts/`，不进 git。

---

## 4. Metadata 必填字段

每个重要 run 的 `metadata.json` 至少包含：

```json
{
  "run_id": "20260708_kuaisearch_bm25_motivation_m1",
  "created_at": "2026-07-08T00:00:00+08:00",
  "git_commit": "unknown",
  "git_dirty": true,
  "dataset_id": "kuaisearch",
  "dataset_version": "v0_lite",
  "split_id": "time_80_10_10_seed20260708",
  "candidate_manifest_sha256": "unknown",
  "method_id": "bm25",
  "method_group": "core",
  "config_path": "configs/baselines/bm25.yaml",
  "config_sha256": "unknown",
  "seed": 20260708,
  "env_group": "core",
  "env_name": "pps-core",
  "python": "3.x",
  "packages": {},
  "cuda_visible_devices": "",
  "gpu_name": "",
  "hostname": "",
  "command": ""
}
```

若某字段当时无法获得，写 `"unknown"`，不要省略字段。后续脚本可以把这些字段
自动补齐。

---

## 5. 评测边界

从 C1 之后，所有方法遵守同一个评测边界：

1. 输入只来自 `data/standardized/<dataset_id>/<version>/` 的统一 JSONL；
2. 候选集来自 candidate manifest；
3. 方法脚本只导出 `scores.jsonl`；
4. evaluator 在评测前 assert `candidate_manifest_sha256` 一致；
5. evaluator 统一计算 NDCG@10、MRR、Recall@10、purchase-NDCG@10 等指标；
6. evaluator 结果写 `runs/<run_id>/metrics.json`，重要摘要再提升到 `reports/`
   或 `experiments/`。

禁止事项：

- 禁止方法脚本私自过滤候选；
- 禁止方法脚本用自己的指标实现生成论文数字；
- 禁止因为某个 baseline 难适配而改 split 或 candidate manifest；
- 禁止在 dev 上看过结果后修改 test split；
- **禁止任何打分/训练代码读取 `qrels_dev.jsonl` / `qrels_test.jsonl`**
  （dev/test records 本身不含标签，见 doc 11 §1.2；qrels 只允许共享
  evaluator 和 M3–M6 分析脚本读取）；违反 = 该方法全部 run 作废。

**dev 评测日志（防止无限调 dev）**：

- 共享 evaluator 每次在 dev 上运行，追加一行到
  `reports/dev_eval_log.jsonl`（字段：timestamp、run_id、method_id、
  split、ndcg@10）；该文件进 git；
- 每个方法进入论文主表时，其 dev 评测次数必须与 baseline card 登记的
  tuning budget（doc 13 §2.5）对得上；对不上要么补记原因，要么按超预算
  处理（结果标注 asymmetric budget）。

**复跑确定性**：

- 每个方法在冻结 config 后，对 dev 抽样 1000 请求复跑一次打分，score 必须
  逐值一致；若方法本身有随机性（如 LLM 采样），必须固定 seed/temperature
  并在 metadata 记录，仍不可复现的要在 baseline card 声明并报告方差。

---

## 6. 调参与种子

阶段纪律：

- Phase 0-2 早期可以单 seed，必须标注 provisional；
- C2 baseline 可信度 gate 之前，不看 M3 oracle headroom；
- 所有 trainable baseline 用相同调参预算；预算的具体数字在
  `doc/13_baseline_implementation_plan.md` §2.5 量化并冻结，run summary 中
  记录搜索空间；
- Tier-2/final claim 阶段，trainable comparison 至少 3 seeds，报告 mean 和波动，
  不报告 best seed。

调参输出：

```text
runs/<run_id>/search_space.json
runs/<run_id>/trials.jsonl
runs/<run_id>/best_config.yaml
```

这些文件是 raw run state，默认不进 git。最终只把小的调参预算摘要提升到
`experiments/` 或 `reports/`。

---

## 7. Tracked Summary

重要 run 完成后，创建一份小摘要，放在 `experiments/` 或 `doc/dev_log/`。

摘要必须包含：

- run ID；
- command；
- config path；
- git commit 和 dirty 状态；
- dataset version、split ID、candidate manifest hash；
- checkpoint 或外部模型引用；
- 主指标和关键次指标；
- 结论：继续、复查、废弃或进入下一个 checkpoint。

不要把完整日志、完整预测、完整 sweep 表复制到 tracked 文档。

---

## 8. Checkpoint 对应关系

| Checkpoint | 执行协议要求 |
|---|---|
| C0 | 数据审计 run 记录下载来源、schema、样本窗口和审计脚本版本 |
| C1 | split、candidate manifest、指标单测和 canary 都有 metadata/hash |
| C2 | 每个 baseline 的 score 文件由同一 evaluator 评测，候选 hash 一致 |
| C3 | oracle/headroom 分析只读 C2 合格 run 的 metrics 和 per-request scores |
| C4 | 三个数据轨共用同一 record schema，代码中无 `if dataset == X` 分支 |
| C5 | insight 复测记录跨数据集、跨 seed、跨 baseline 的完整 provenance |

---

## 9. 最小执行命令模板

轻量 baseline：

```bash
CUDA_VISIBLE_DEVICES="" \
python scripts/score_baseline.py \
  --config configs/baselines/bm25.yaml \
  --run-id 20260708_kuaisearch_bm25_motivation_m1
```

GPU baseline：

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/train_or_score_baseline.py \
  --config configs/baselines/bge_m3_zero_shot.yaml \
  --run-id 20260708_kuaisearch_bge_m3_motivation_m1
```

共享评测：

```bash
python scripts/evaluate_scores.py \
  --run-id 20260708_kuaisearch_bm25_motivation_m1 \
  --candidate-manifest data/standardized/kuaisearch/v0_lite/candidate_manifest.json
```

这些脚本名是协议占位；实际实现时可以调整名称，但必须保留同等输入输出边界。
