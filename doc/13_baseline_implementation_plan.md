# 13 - Baseline 实现与评测计划

状态：执行计划。前置：`11_experiment_and_dataset_plan.md` 定义 baseline 矩阵；
`12_experiment_execution_protocol.md` 定义环境、GPU、run 和评测边界。本文回答：
每个 baseline 如何实现、如何导出分数、如何验收，避免 baseline 只停留在列表层面。

核心原则：

1. 所有 baseline 读取同一个 standardized JSONL；
2. 所有 baseline 导出同一种 `scores.jsonl`；
3. 所有论文数字由同一个 evaluator 产生；
4. 每个 baseline 先通过 sanity checks，再进入 M1/M2/M3 分析；
5. 自实现 baseline 是 control，不包装成 SOTA；强 baseline 需要记录出处、适配和调参预算。

---

## 1. 统一输入输出

### 1.1 输入

每个 baseline 只允许读取：

```text
data/standardized/<dataset_id>/<version>/
  records_train.jsonl        # 含标签（训练用）
  records_dev.jsonl          # blind：candidates 不含任何标签字段
  records_test.jsonl         # blind
  item_catalog.jsonl
  candidate_manifest.json
  manifest.json
```

`qrels_dev.jsonl` / `qrels_test.jsonl`（标签文件，doc 11 §1.2）**不在此列**：
任何 baseline 的打分/训练代码读取 qrels = 该方法全部 run 作废。

禁止回读原始 KuaiSearch/Amazon/JDsearch 表做额外特征，除非该特征先被写入统一
record schema 并重新通过 C1。

**train 统计的边界**：允许用 `records_train.jsonl` 的标签做统计特征（如
popularity、query 点击熵），禁止用 dev/test 的任何标签或行为构造任何特征。

**请求内 z-score 的统一定义**（B0b/B1/B2z 导出、B7 消费，实现进
`src/myrec/`，不许各方法自写）：

```text
z(s_i) = (s_i - mean(request 内所有候选分)) / std(request 内所有候选分)
std = 0（全同分）时，所有 z 置 0。
```

### 1.2 输出

每个 baseline 的 score 文件：

```text
runs/<run_id>/scores.jsonl
```

每行至少包含：

```json
{
  "request_id": "req",
  "candidate_item_id": "item",
  "score": 0.0,
  "method_id": "bm25"
}
```

允许附加诊断字段，例如 `query_score`、`history_score`、`alpha`、`latency_ms`，
但 evaluator 的主指标只读 `request_id`、`candidate_item_id`、`score`。

### 1.3 共享评测

评测命令统一形态：

```bash
python scripts/evaluate_scores.py \
  --run-id <run_id> \
  --split dev \
  --candidate-manifest data/standardized/<dataset>/<version>/candidate_manifest.json
```

evaluator 必须在计算指标前检查：

- `scores.jsonl` 中每个请求的 candidate set 与 manifest 一致；
- 每个候选恰好一个 score；
- 没有未知 `request_id` 或未知 `candidate_item_id`；
- `metadata.json` 中的 `candidate_manifest_sha256` 与实际文件一致。

---

## 2. 实现批次与范围控制

Baseline 不一次性全做。第一批只覆盖每个关键类别的一个代表，目标是在最短路径上
验证方向是否值得继续，而不是把 baseline suite 做完整。

范围纪律：

- Batch 1 未完成前，不启动 RecBole、KuaiSearch official、PPS classic 或 LLM
  的正式接入；
- Batch 1 的每个方法都必须能通过共享 evaluator 产出 dev 指标；
- 只有 C1 通过且 Batch 1 结果说明存在非零 headroom，才进入 Batch 2；
- 如果 Batch 1 已经证伪核心 insight，停止实现更重 baseline，按 doc 11 回退。

### Batch 0 - 仪器

先实现这些，不算正式 baseline：

| 项 | 目的 | 验收 |
|---|---|---|
| Random scorer | 指标 sanity 下界 | NDCG/MRR 接近随机水平 |
| Oracle label leak canary | 检查 evaluator 能识别明显信号 | 正例 title 拼进 query 后指标显著上升 |
| Candidate hash check | 固定候选池 | 任一候选缺失/新增时 evaluator 报错 |
| Per-request metric dump | 支持 M3 oracle | 能输出每个 request 的 NDCG@10 |

### Batch 1 - 第一批最小 baseline

第一批正式 baseline 只做 5 个方法，加 1 个 oracle 分析。每个证据类别只选一个
最重要代表：

| ID | 方法 | 目的 | 环境 |
|---|---|---|---|
| B0a | Popularity | 无个性化下界，确认标签/候选基本可用 | core |
| B0b | Recent-behavior | history-only 代表，检查历史是否有信号 | core |
| B1 | BM25 | query-only 代表，先用最稳的 lexical baseline | core |
| B2z | Dense bi-encoder zero-shot | 现代语义 query-only 代表，用来补 BM25 的语义盲区 | embed |
| B7 | Static mixture | query/history 双通道但非自适应的关键 control | core/embed |
| M3 | Per-request oracle | 分析，不算可部署 baseline；测自适应 headroom | core |

Batch 1 的目标不是追 SOTA，而是回答方向生死问题：query-only、history-only、
static mixture 是否互补失败，以及逐请求选择是否有 headroom。

Batch 1 明确不做：

- 不做 B3 cross-encoder fine-tuning；
- 不做 B4 SASRec/BERT4Rec；
- 不做 B5 DIN/DCNv2 官方复现；
- 不做 B6 HEM/ZAM/TEM；
- 不做 B8 LLM/MemRerank-style；
- 不做多 seed 大调参。

Batch 1 完成定义：

1. Random/canary/candidate hash check 通过；
2. B0a、B0b、B1、B2z、B7 都有 `scores.jsonl` 和 `metrics.json`；
3. B1 显著高于 B0a，B0b 显著高于 random；
4. B7 的 alpha 曲线落盘；
5. M3 oracle 只读取上述合格 run，产出 headroom summary；
6. 形成一页 Batch 1 决策摘要：继续、回退或补查数据。

### Batch 2 - 强 baseline

Batch 1 通过后再做：

| ID | 方法 | 作用 |
|---|---|---|
| B4 | SASRec/BERT4Rec via RecBole | 强 sequence/history baseline |
| B5 | DIN/DCNv2 via KuaiSearch official | 官方工业 CTR/ranking baseline |
| B6 | HEM/ZAM/TEM | PPS 经典 baseline |
| B3 | Cross-encoder reranker | query-only 语义强上界 |
| B6+ | MAI/NAM-style if feasible | 较新 PPS/when-personalize 对照 |
| B8 | LLM raw-history + MemRerank-style | 质量/成本上界 |

### 2.4 公平性边界矩阵（每个方法允许读什么，验收时逐项核对）

"✓" = 允许使用；"✗" = 禁止使用（使用即该方法作废重跑）。所有方法一律使用
records 中冻结的 ≤50 条历史（doc 11 §1.2），可截短不可延长。

| ID | query 文本 | history | candidate 文本/属性 | candidate item_id | train 标签 | dev/test 标签 |
|---|---|---|---|---|---|---|
| Random | ✗ | ✗ | ✗ | ✓（只做 key） | ✗ | ✗ |
| B0a | ✗ | ✗ | ✗ | ✓ | ✓（统计点击量） | ✗ |
| B0b | ✗ | ✓ | ✓（仅类目/item_id 匹配） | ✓ | ✗ | ✗ |
| B1 | ✓ | ✗ | ✓ | ✓ | ✗ | ✗ |
| B2z | ✓ | ✗ | ✓ | ✓ | ✗ | ✗ |
| B3 | ✓ | ✗ | ✓ | ✓ | ✓（仅 fine-tune 版训练） | ✗ |
| B4 | ✗ | ✓（item_id 序列） | ✗ | ✓ | ✓（序列训练） | ✗ |
| B5 | ✓ | ✓ | ✓ | ✓ | ✓（CTR 训练） | ✗ |
| B6 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| B7 | ✗（只读上游 scores） | ✗（只读上游 scores） | ✗ | ✓ | ✗（α 在 dev 指标上选，经共享 evaluator） | ✗ |
| B8 | ✓ | ✓ | ✓ | ✓ | ✗（zero-shot） | ✗ |
| M3 | —（只读合格 run 的 per-request metrics 与 qrels，分析专用） | | | | | |

补充规则：

- 单通道方法（B1/B2z/B3 不读 history；B0b/B4 不读 query）是**设计如此**，
  其与全通道方法的差距按 evidence-access 差距解读，不算调参不公（07 §9）；
- 任何方法不得读取其他方法的 run 目录（B7/M3 例外，二者的输入 run 列表
  必须写进自己的 config）；
- 文本类方法（B1/B2z/B3/B8）的 item 文本统一用 §3 B1 定义的 document 模板，
  谁也不许私自增删字段——语义方法与词法方法必须看到同一段文本。

### 2.5 调参预算（量化，Batch 2 开跑前冻结，之后不许改）

| 类别 | 预算 | 说明 |
|---|---|---|
| 自实现零训练（B0a/B0b/B1） | 1 个声明默认 config + ≤ 8 次 dev 评测的变体 | 全部记入 `reports/dev_eval_log.jsonl`；超出预算的变体结果作废 |
| B7 | α 网格 11 点 × 2 个 query 通道实例 = 22 次 dev 评测 | 网格本身就是它的全部预算 |
| zero-shot（B2z/B3-zs/B8） | 1 个声明 config，0 次调参 | prompt/模板改动算调参，每次改动记日志，≤ 3 次 |
| trainable（B3-ft/B4/B5/B6/B6+/提议系统） | 每方法 **16 次 dev 评测**，搜索空间先写进 config 再开跑 | 提议系统与 baseline 同预算（07 §9）；官方默认超参算第 1 次 |
| 多 seed（Tier-2） | 最优 config × 3 seeds | 只在冻结 config 后做，seed 不算调参次数 |

预算不对称（例如官方代码只能跑默认配置）必须写进 baseline card 和论文，
不许事后声称"都调过了"。

### 2.6 每个方法的期望产出（验收即对照此表，缺一不算完成）

所有方法（含 Random）统一交付：

| 产出 | 路径 | 说明 |
|---|---|---|
| config | `configs/baselines/<method>.yaml` | 进 git |
| run 目录 | `runs/<run_id>/` 含 metadata/config 快照/日志 | 不进 git |
| 分数 | `runs/<run_id>/scores.jsonl` | dev 全量；每候选恰一分 |
| 指标 | `runs/<run_id>/metrics.json` | 只能由共享 evaluator 生成 |
| 请求级指标 | `runs/<run_id>/per_request_metrics.jsonl` | M3/M5 的输入，Batch 1 起强制 |
| baseline card | `experiments/pps_baseline_cards.md` | 含实测 run ID 与验收结论 |
| 结果登记行 | `experiments/pps_results.md` | 数字只准从 metrics.json 复制 |

方法特有的额外产出：

| ID | 额外产出 |
|---|---|
| B0a | train 点击量统计的落盘位置与 hash |
| B0b | 公式权重的变体日志（若调过） |
| B1 | 分词器名称与版本；20 请求人工 top-5 检查记录 |
| B2z | 模型名/版本/权重 hash；截断长度；B1 vs B2z per-request winner 表 |
| B3 | latency/吞吐记录；ft 版另附 trials.jsonl |
| B4 | RecBole config/版本、负采样设置、候选打分方式说明 |
| B5 | 官方 commit/patch/license；与官方论文数字的对齐报告（±10%） |
| B6 | 每个子方法（HEM/ZAM/TEM）的适配差异说明 |
| B7 | 两个实例（见 §3）的完整 α 曲线（`alpha_curve.json`） |
| B8 | prompt 全文、模型/量化/温度、latency、token 成本、解析失败率、三档历史长度对比 |
| M3 | `oracle_choices.jsonl`、`headroom_summary.json`（含 bootstrap CI、split-half、通道选择分布，doc 11 M3） |

---

## 3. Baseline 逐项方案

### B0a Popularity

输入：train records。

实现：

1. 统计 train 中每个 `item_id` 的点击数和购买数；
2. 默认 score = `log1p(click_count)`；
3. 无点击记录的 item 使用 0；
4. 可选 tie-breaker 用 train item exposure count，但必须记录。

输出：对 dev/test 每个 candidate 输出 popularity score。

验收：

- score 不使用当前请求 query/history；
- dev 指标显著高于 random；
- 作为下界，不要求强。

### B0b Recent-behavior

输入：每条 record 的 history 和 candidates。

实现：

1. exact item match：candidate 出现在近期 history 中加最高分；
2. category overlap：三级类目从细到粗给不同权重；
3. event weight：purchase > click；
4. recency decay：越近权重越高；
5. score 标准化到请求内 z-score，供 B7 复用。

建议初始公式：

```text
score = 3.0 * item_match + 1.0 * cat_l3_match + 0.5 * cat_l2_match
        + 0.2 * cat_l1_match
```

以上权重是**声明默认值**；调整属于调参，走 §2.5 预算（≤8 次 dev 评测，
记日志），进入 M3 前冻结最终公式。

验收：

- 不读取 query 文本；
- history 缺失时所有候选同分并设置诊断字段；
- B0b 应显著高于 random，否则优先检查 history join。

### B1 BM25

输入：query 和 candidate item text。

实现：

1. document = `title + brand + seller + cat_l1 + cat_l2 + cat_l3`；
2. 中文分词先用轻量可复现方案；后续可替换为 jieba/pyserini analyzer；
3. 每个请求只对其固定 candidates 打分，不从全库召回；
4. score 请求内保留原始 BM25 和 z-score。

验收：

- 不读取用户 history；
- B1 必须显著高于 B0a，否则 query 字段、分词或 item text join 有问题；
- 抽 20 个请求人工看 top-5 是否符合 query 直觉。

### B2z Dense Bi-encoder Zero-shot

输入：query 和 candidate item text。

实现：

1. 模型候选：`bge-m3` 或同等级中文/多语 embedding model；
2. candidate 文本**必须使用与 B1 完全相同的 document 模板**（字段与拼接顺序
   一致，§2.4 补充规则）；截断长度写进 config；
3. query embedding 与 candidate text embedding 做 cosine/dot product；
4. item embedding 可以缓存到 `models/` 或 `artifacts/`，不进 git；
5. 第一版只做 zero-shot，不微调（模板/截断改动按 §2.5 记日志）。

验收：

- 不读取用户 history；
- 与 BM25 互补：记录 B1/B2z 的 per-request winner；
- 如果显著低于 BM25，抽样检查中文 tokenizer、截断和文本拼接。

### B3 Cross-encoder

输入：query-candidate text pair。

实现：

1. zero-shot reranker 先跑 dev 抽样或 top-k candidates；
2. 若算力允许，再做全 dev top-100；
3. fine-tune 版必须只用 train，dev 只调参，test 最后一次。

验收：

- 作为 query-only 语义强上界；
- 不读取 history；
- 记录 latency 和吞吐，避免只报告质量。

### B4 SASRec/BERT4Rec

输入：train/dev/test records 转换成 RecBole 序列格式。

实现：

1. 用户历史序列来自 record history，不回读原始表；
2. 训练目标使用 train 行为序列；
3. scoring 阶段只给当前请求 fixed candidates 打分；
4. adapter 负责把 RecBole item score 映射回 `candidate_item_id`。

验收：

- query-blind，不读取 query 文本；
- history-only 显著高于 random；
- 记录 RecBole config、版本、负采样设置和候选打分方式。

### B5 DIN/DCNv2 KuaiSearch Official

输入：统一 records 到官方代码格式的 adapter。

实现：

1. 优先复现官方 KuaiSearch baseline；
2. 只在 adapter 层做格式转换；
3. 若官方代码需要额外字段，必须说明这些字段是否已在统一 record 中存在；
4. 输出 fixed candidates 的 score，不使用官方私有 evaluator。

验收：

- 与 KuaiSearch 官方论文数字对齐到合理范围；C2 标准是 ±10%；
- 对不齐时先写 protocol-diff report，不带着不明差异进入 M3；
- 记录官方 commit、patch、license 和 setup。

### B6 HEM/ZAM/TEM

输入：统一 records 转换为 PPS classic 格式。

实现：

1. query 使用真实 query；
2. item text 使用 title/brand/category；
3. history 使用 clicked/purchased item sequence；
4. 若原方法依赖 review 或缺失字段，写明替代策略；
5. adapter-only 优先，不随意改模型结构。

验收：

- 记录每个方法的 paper、官方代码状态、适配差异；
- 若某方法无法公平适配，保留为 related work，不进主表；
- 指标仍由共享 evaluator 产生。

### B6+ Recent PPS / When-personalize Baseline

候选：MAI/Dynamic Multi-attribute Interest、NAM-style、MemRerank-style 的轻量版。

实现原则：

1. 优先选择有公开代码或清晰结构的近期 PPS 方法；
2. 若没有代码，可以做 faithful reimplementation，但必须标注；
3. 只把能接入 fixed-candidate scoring 的方法放进主表；
4. 对私有数据论文，只能做方法思想适配，不能声称官方复现。

验收：

- baseline card 明确：official code / reimplementation / style-adapted；
- 若适配不可靠，放入 appendix 或 related work，而不是主结论支撑。

### B7 Static Mixture

输入：query 通道 score（B1 或 B2z）与 B0b history score，只读上游 run 的
`scores.jsonl`，上游 run ID 写进 B7 config。

实现（**两个固定实例**，消除"query 通道选哪个"的自由度）：

1. `B7-bm25` = z(B1) 与 z(B0b) 混合；`B7-bge` = z(B2z) 与 z(B0b) 混合；
   z-score 用 §1.1 的统一实现；
2. 每个实例 grid search `alpha in {0.0, 0.1, ..., 1.0}` on dev（共 22 次
   dev 评测，即 §2.5 的全部预算）；
3. score = `alpha * z(query_score) + (1 - alpha) * z(history_score)`；
4. 两个实例都报告；dev 更优者记为 "B7-best"，供 M3/E3 引用；
5. test 使用 dev 选出的全局 alpha。

验收：

- B7 是最重要 control：证明“加两个通道”不等于“自适应”；
- 两个实例各自保存完整 alpha 曲线（`alpha_curve.json`）；
- alpha 曲线若近乎平坦，按 doc 11 M2 记为警报并写入 Batch 1 决策摘要；
- M3 oracle 必须把 B7-best 作为候选之一。

### B8 LLM / MemRerank-style

输入：query、历史摘要、top-k candidates。

实现：

1. B8a raw-history：直接给 query、最近历史、候选，输出排序；
2. B8b MemRerank-style：先抽取 preference memory，再 rerank；
3. history length 设 5/20/50 三档；
4. **候选截断规则（冻结）**：LLM 只重排 top-20，top-20 由当时 dev 上最强的
   非 LLM 方法的 scores 决定（哪个 run 写进 B8 config）；其余候选保留该基础
   方法的分数并整体排在重排段之后——这样 `scores.jsonl` 仍覆盖全候选，
   与其他方法在同一 manifest 上可比；
5. dev 抽样 2000 请求（固定 seed 抽样，请求清单落盘），不直接全量；
   抽样子集上的对比方法必须在**同一子集**上重新汇总指标，禁止拿子集 B8
   对比全量 baseline；
6. 输出解析失败的请求回退为基础方法排序，并计入解析失败率；
7. 记录 prompt、模型、量化、温度/seed、latency、tokens、成本。

验收：

- 作为质量/成本上界，不作为轻量生产 baseline；
- 记录解析失败率；
- 若使用闭源 API，记录日期、模型版本和价格口径；
- KuaiSearch 上只能称为 MemRerank-style adapted baseline，不能称为官方 MemRerank 复现。

---

## 4. Oracle 与失败切片

M3 oracle 不训练模型，只读合格 baseline 的 per-request metrics：

```text
oracle(request) = max_by_ndcg10(
  best_query_only,
  best_history_only,
  best_static_mixture
)
```

需要落盘：

```text
runs/<run_id>/per_request_metrics.jsonl
runs/<run_id>/oracle_choices.jsonl
runs/<run_id>/headroom_summary.json
```

验收：

- oracle 只用于分析 headroom，不进入可部署方法主表；
- C2 之前不看 oracle 结果；
- `headroom_summary.json` 必须包含 doc 11 M3 的三个噪声护栏
  （bootstrap CI、split-half 一致性、通道选择分布），缺任一项不得引用；
- headroom 判定按 doc 11 C3 表执行（点估计 ≥ +5% 且 CI 下界 ≥ +2% 且
  split-half 同向），失败按 doc 11 回退。

---

## 5. Baseline Card 必填项

每个 baseline 在 `experiments/pps_baseline_cards.md` 登记：

- method ID；
- method name；
- role：control / lower bound / strong baseline / upper bound / oracle；
- evidence channels：query / history / item text / full features / LLM；
- source：paper、repo、official weights；
- venue/year；
- implementation type：self-implemented / official code / adapter-only /
  reimplementation / style-adapted / zero-shot；
- input fields used；
- output score definition；
- tuning budget；
- known limitations；
- current status。

---

## 6. 完成定义

一个 baseline 算完成，必须同时满足：

1. 有 config；
2. 有 run metadata；
3. 能从 standardized JSONL（blind dev records）读 dev split，全程未触碰 qrels；
4. 能写 `scores.jsonl`（全候选覆盖、无 NaN/Inf、复跑 1000 请求分数一致，
   doc 12 §5）；
5. 共享 evaluator 能生成 `metrics.json` 和 `per_request_metrics.jsonl`；
6. candidate hash assert 通过；
7. §2.4 公平性矩阵逐项核对通过（输入字段没有越界）；
8. dev 评测次数与 §2.5 预算对账通过（`reports/dev_eval_log.jsonl`）；
9. baseline card 已更新（含实测 run ID 与验收结论）；
10. `experiments/pps_results.md` 已登记结果行，数字与 `metrics.json` 一致；
11. 若进入论文主表，必须有调参预算或 zero-shot 声明。

只完成训练脚本、不通过共享 evaluator 的方法，不算 baseline 完成。

---

## 7. 开发者执行流程（按图索骥，逐 baseline 走一遍）

前置：C0、C1 已通过（standardized 数据、manifest、指标单测、canary 就绪）。

```text
step 1  读本文件 §2.4 该方法那一行，确认允许输入字段；读 §3 对应小节。
step 2  写 configs/baselines/<method>.yaml（含搜索空间，若 trainable）。
step 3  在 experiments/pps_baseline_cards.md 登记 card（状态 = in progress）。
step 4  选环境组（doc 12 §1），生成 run_id（doc 12 §3），运行打分脚本，
        产出 runs/<run_id>/scores.jsonl。
step 5  运行共享 evaluator（doc 12 §9 命令模板）→ metrics.json +
        per_request_metrics.jsonl；每次 dev 评测自动记入 dev_eval_log。
step 6  跑 §3 对应小节的验收项（含与参照方法的显著性检验，用共享
        compare 脚本，"显著"定义见 doc 11 §1.4）。
step 7  调参在 §2.5 预算内迭代 step 4–6；预算用完即冻结。
step 8  复跑确定性检查（doc 12 §5）。
step 9  对照 §2.6 交付物表逐项核对；更新 card（状态 = accepted / blocked）；
        在 experiments/pps_results.md 登记结果行。
step 10 写一份 tracked summary（doc 12 §7）；异常和坑记 doc/baseline_notes/。
```

红线（任何一条触发 = run 作废，card 标记并重跑）：

- 读了 qrels 或 dev/test 标签；
- 换了候选池 / split / manifest；
- 用私有指标实现产出数字；
- 超预算调参未声明；
- 越过 §2.4 矩阵读取禁用字段。
