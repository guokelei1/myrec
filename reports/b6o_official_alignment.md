# B6o HEM Official Alignment

Date: 2026-07-09

Status: failed external alignment. HEM official code path runs end to end,
but the best Cell Phones & Accessories metric is outside the +/-10% target
band. No KuaiSearch dev evaluation has been produced, and B6o must not enter
the formal KuaiSearch baseline table from this run.

## Source

- HEM upstream:
  `https://github.com/QingyaoAi/Hierarchical-Embedding-Model-for-Personalized-Product-Search`
- Upstream commit: `cd089d0ecf277e2fcdccb35f4989d05ef3e81032`
- License: Apache-2.0
- Environment: `pps-classic`, frozen in `configs/env/pps_classic.txt`
- TensorFlow sanity: `tensorflow==1.4.0`, `numpy==1.16.5`, Python 2.7.18.
- HEM CLI sanity: `HEM/main.py --help` passed.

## External Data

Alignment category: Amazon Product Search, Cell Phones & Accessories.

Raw data were downloaded to ignored `artifacts/batch2b/` paths:

| File | SHA256 |
|---|---|
| `reviews_Cell_Phones_and_Accessories_5.json.gz` | `477f09973936c491c4c138436c477ba257d2e1fc556480c80fb75ebae10b4dab` |
| `meta_Cell_Phones_and_Accessories.json.gz` | `71dd91522ece7d4f18184b15f3110d4c77d909c5475745b6019a91497393bb39` |

The HEM indexed data were built with the official jar/script chain:

1. `AmazonReviewData_preprocess.jar false ...`
2. `index_and_filter_review_file.py ... 5`
3. `AmazonMetaData_matching.jar false ...`
4. `scripts/build_b6o_hem_benchmark_split.py`

The final HEM split manifest is
`reports/b6o_hem_benchmark_split_manifest.json`.

## Benchmark Split

The reconstructed split matches the public benchmark qrels counts:

| Quantity | Value |
|---|---:|
| Train reviews | 150,048 |
| Test reviews | 44,391 |
| Train qrels rows | 174,087 |
| Test qrels rows | 667 |
| Benchmark queries | 165 |
| Query strings with missing vocab words | 0 |

Eight product-query entries were dropped because two metadata-derived query
strings differ from the public benchmark query text. This affects product-query
candidate generation for those product lines only; the public benchmark qrels
and query ids are preserved.

## Smoke

A one-epoch, reduced-size HEM smoke run completed:

| Setting | Value |
|---|---|
| `embed_size` | 20 |
| `negative_sample` | 2 |
| `max_train_epoch` | 1 |
| Ranklist rows | 66,500 |
| Ranked queries | 665/665 |
| MAP@100 | 0.0075 |
| MRR@100 | 0.0076 |
| NDCG@10 | 0.0066 |

This smoke run verifies the code/data/evaluator chain only; it is not an
alignment claim.

## Official-Parameter Run

Current command:

```bash
/home/gkl/miniconda3/envs/pps-classic/bin/python -u \
  baselines/pps_classic/hem_official/HEM/main.py \
  --data_dir artifacts/batch2b/b6o_amazon_alignment/cellphones/indexed/min_count5/ \
  --input_train_dir artifacts/batch2b/b6o_amazon_alignment/cellphones/indexed/min_count5/official_query_split/ \
  --train_dir artifacts/batch2b/b6o_hem_cellphones_official/ \
  --similarity_func cosine \
  --net_struct simplified_fs \
  --embed_size 100 \
  --window_size 3 \
  --max_train_epoch 20 \
  --steps_per_checkpoint 400 \
  --negative_sample 5 \
  --subsampling_rate 0.0 \
  --L2_lambda 0.005 \
  --batch_size 64
```

Target paper numbers for HEM on Cell Phones & Accessories: MAP ~= 0.124,
MRR ~= 0.124, NDCG@10 ~= 0.153.

The training command completed successfully and wrote checkpoint
`ProductSearchEmbedding.ckpt-726040`.

## Final Metrics

All decoding runs used the same checkpoint and the official rank cutoff 100.
The official code produced 66,500 ranklist rows in each scoring mode, covering
665/665 positive user-query ids in `test.qrels`.

| Scoring mode | MAP@100 | MRR@100 | NDCG@10 | Missing ranklists |
|---|---:|---:|---:|---:|
| `cosine` | 0.0564 | 0.0565 | 0.0623 | 0 |
| `product` | 0.0757 | 0.0757 | 0.0932 | 0 |
| `bias_product` | 0.0759 | 0.0760 | 0.0885 | 0 |

Best observed MAP is 0.0759, which is 61.2% of the target 0.124 and outside
the accepted +/-10% alignment band. Best observed NDCG@10 is 0.0932, which is
60.9% of the target 0.153.

Verdict: B6o HEM Path 1 is not externally aligned. This is an implementation
and protocol-diff finding, not a KuaiSearch result.

## Diagnostics

- Preprocessing did use the official Krovetz stemming path in
  `AmazonReviewData_preprocess.jar`; the `false` argument only disables
  stopword removal.
- The public benchmark qrels/query files were preserved exactly; the split
  manifest records `test.qrels_sha256 =
  9b7234641f3278f2e0af70bd358c8d9659b894fdbab1ad117ce3a7cfe4085ae1`.
- Ranklist coverage is complete against public qrels, so the failure is not
  caused by missing user-query outputs.
- The remaining likely protocol gap is benchmark reconstruction: the public
  dataset repository provides qrels/query text/train review ids, but not the
  original indexed `query_split/` files or trained model state. Our
  reconstruction drops eight product-query entries whose metadata-derived query
  text does not match the public benchmark query list.

Next action: do not adapt this checkpoint to KuaiSearch. Either obtain the
exact original indexed split/model settings, or run a separate faithful
reimplementation/alignment attempt and require the same +/-10% external gate
before any KuaiSearch B6o dev evaluation.
