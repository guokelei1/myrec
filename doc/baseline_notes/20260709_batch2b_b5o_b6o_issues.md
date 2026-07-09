# Batch 2b B5o/B6o 问题记录

日期：2026-07-09

目的：单独记录 B6o HEM 和 B5o KuaiSearch official 在 Batch 2b 中遇到的
外部对齐问题，方便后续判断是否值得修复，以及应该怎么修。

## 当前结论

- B6o：官方 HEM 代码真实跑通，但 Amazon 外部基准未复现到论文数字，按
  doc 14 不能接 KuaiSearch formal dev。
- B5o：KuaiSearch 官方 ranking 代码能 import、能 tiny smoke train，但官方
  repo 当前 commit 与公开数据/自身 demo 不自洽，论文 Table 7 对齐不可验证。
- 这两个问题都不是“模型在我们任务上效果差”，而是“进入我们 formal baseline
  前的官方/外部对齐证据不够”。

## B6o：HEM 官方复现失败

### 做了什么

- 下载并固定 HEM 官方 repo：
  `QingyaoAi/Hierarchical-Embedding-Model-for-Personalized-Product-Search`
  commit `cd089d0ecf277e2fcdccb35f4989d05ef3e81032`。
- 建立 `pps-classic` 环境：Python 2.7.18 + TensorFlow 1.4。
- 下载 Amazon Cell Phones & Accessories 5-core review/meta 原始数据。
- 用官方 jar/script 链路做 preprocessing、index、metadata query matching。
- 根据 public Amazon Product Search benchmark 的 `query_text`、
  `train_review_id`、`train/test.qrels` 重建 HEM split。
- 跑完 official-parameter HEM：20 epoch、`simplified_fs`、`embed_size=100`、
  `negative_sample=5`、`subsampling_rate=0.0`、`L2_lambda=0.005`。
- 用同一 checkpoint decode `cosine`、`product`、`bias_product` 三种官方
  scoring mode，并用 TREC qrels 评估。

### 观察到的数字

论文目标（Cell Phones & Accessories）：

| Metric | Target |
|---|---:|
| MAP | ~0.124 |
| MRR | ~0.124 |
| NDCG@10 | ~0.153 |

我方 best：

| Scoring | MAP@100 | MRR@100 | NDCG@10 |
|---|---:|---:|---:|
| cosine | 0.0564 | 0.0565 | 0.0623 |
| product | 0.0757 | 0.0757 | 0.0932 |
| bias_product | 0.0759 | 0.0760 | 0.0885 |

best MAP 只有目标的约 61%，best NDCG@10 也只有目标的约 61%，不满足 ±10%
外部对齐门槛。

### 关键问题

公开 benchmark repo 提供了 qrels、query text、train review id，但没有提供
HEM 训练实际使用的完整 indexed `query_split/` 文件或原 checkpoint。我们只能
从 raw review/meta + public qrels 重建 split。

重建过程中发现：

- public qrels/query id 可完整保留；
- ranklist 覆盖完整，665/665 positive user-query 都有输出；
- preprocessing 确实走了官方 Krovetz stemming；
- 但有 8 条 product-query entry 因 metadata-derived query string 和 public
  benchmark query text 不一致而被丢弃；
- 当前最可能的问题是重建 split 与作者原实验 split/index 状态仍有协议差。

### 修复方向

1. 最好：找到作者原始 indexed split 或 checkpoint。
   - 需要 `query_split/`、`product_query.txt.gz`、`query.txt.gz` 等和论文实验
     完全一致的中间产物。
   - 如果能拿到，直接重跑 decode/eval，判断是否恢复到论文数字。

2. 次选：复查 split 重建逻辑。
   - 对比 official `split_train_test_data.py` 生成 qrels 的精确逻辑；
   - 只保留 public qrels 涉及的 test reviews / user-query pairs；
   - 检查 product query candidate generation 是否和 qrels 构造一致；
   - 但如果 public repo 本身缺失原始随机状态，仍可能无法精确复现。

3. 退路：faithful reimplementation。
   - 按论文公式重写 HEM/TEM；
   - 仍必须先在 Amazon PPS benchmark 达到 ±10%，否则不能接 KuaiSearch。

## B5o：KuaiSearch 官方 ranking 对齐不可验证

### 做了什么

- 下载并固定 KuaiSearch 官方 repo：
  `benchen4395/KuaiSearch` commit
  `7ce0471b659112096f0aa7e892ed0aa4c972246a`。
- 建立 `pps-kuaisearch` 环境，官方 ranking CLI `--help` 通过。
- 用 tiny synthetic bridge 数据跑通官方 `ranking/main.py` DCNv1 1 epoch：
  能 train、valid、test、保存 checkpoint。
- 这个 smoke 只证明代码/环境可运行，不是论文数字对齐。

### 论文目标

KuaiSearch paper Table 7 ranking targets：

| Method | Logloss | ROC-AUC |
|---|---:|---:|
| DNN | 0.1588 | 0.6258 |
| Wide & Deep | 0.1598 | 0.6217 |
| DCN | 0.1611 | 0.6194 |
| DCNv2 | 0.1603 | 0.6239 |
| DIN | 0.1606 | 0.6262 |

官方代码也输出 LogLoss / AUC，所以理论上可以对齐这些指标。

### 关键问题

当前 locked repo 的 ranking pipeline 和公开数据/自身 demo 不自洽：

1. 路径不一致：
   - `ranking/data/process.py` 读取 `data/rank.jsonl` 和 `data/corpus.jsonl`；
   - 它把 `query_emb.npy`、`item_title_emb.npy` 等写到当前工作目录 `./`；
   - `ranking/datasets.py` 却从 `./data/query_emb.npy` 和
     `./data/item_title_emb.npy` 读取。

2. 用户字段不一致：
   - 公开 users 文件字段是 `age_bucket`；
   - loader 读取 `age`；
   - matched user 会得到 `age_idx=None`，missing user fallback 则是
     `gender=2`、`age=9`，但模型 embedding cardinality 分别是 2 和 7，
     fallback 本身也会越界。

3. demo 数据不一致：
   - demo rank 的 target item id 是 38-330 的 reindexed 小整数；
   - demo items 是百万级原始 item id；
   - target item 覆盖率为 0，demo 不能直接用于 official ranking train。

4. encoder 描述不一致：
   - paper 说 query/title embedding 来自 BERT encoder；
   - repo 的 `ranking/data/process.py` 使用 `BAAI/bge-small-zh-v1.5`。

这些问题需要 adapter 或 patch 才能跑 full alignment。因此 B5o 当前状态应是
`official-code, alignment-not-verifiable`，不能硬标 official reproduction。

### 修复方向

1. 最干净：向官方确认 Table 7 的确切 preprocessing commit/config。
   - 需要确认 `rank.jsonl/corpus.jsonl/users.jsonl` 的生成方式；
   - 需要确认 embedding encoder 是 BERT 还是 BGE；
   - 需要确认 last-day split 如何由 public rank_lite 构造。

2. 工程修复：写一个 official-format materializer。
   - 从 public raw train 文件生成 `rank.jsonl`、`corpus.jsonl`、`users.jsonl`；
   - 显式把 `age_bucket -> age`；
   - 把 embedding 文件写到 training loader 实际读取的 `./data/`；
   - 修复或避开 missing-user fallback 越界；
   - 然后跑 DNN/DCNv2/DIN，对齐 Table 7。

3. 若仍对不齐：降级为 adapter baseline。
   - 明确说明不是 official-aligned reproduction；
   - 可以作为 appendix/secondary baseline；
   - formal PPS 主表不能把它称作“KuaiSearch official DIN/DCNv2”。

## 对 Batch 2b 的影响

- B4o 是目前唯一完成 formal KuaiSearch dev 的官方实现 baseline。
- B6o 和 B5o 都不能进入主表 official-aligned baseline。
- Batch 2b 不能写成“官方 B4/B5/B6 全部验证后仍弱”。
- 更稳妥的论文表述是：
  - 官方 RecBole SASRec 在当前 fixed-candidate PPS setting 下没有超过 B0b/B7；
  - HEM/B5 official alignment 暂不可用，作为 protocol limitation 报告；
  - 主结论仍依赖 query-conditioned candidate pool 诊断、B0/B1/B2/B7/B8、以及
    M3 oracle/headroom 分析。
