# PPS Classic Baselines

This directory holds upstream code used for Batch 2b B6o.

## Upstream Sources

| Local dir | Method family | Upstream URL | Commit | License |
|---|---|---|---|---|
| `hem_official/` | HEM, SIGIR 2017 | `https://github.com/QingyaoAi/Hierarchical-Embedding-Model-for-Personalized-Product-Search` | `cd089d0ecf277e2fcdccb35f4989d05ef3e81032` | Apache-2.0 |
| `prodsearch_tem/` | TEM/RTM and classic variants, SIGIR 2020/2021 | `https://github.com/kepingbi/ProdSearch` | `449335ba652fe7c877a008e154157d7b2a4b0e76` | Apache-2.0 |

## Environment

HEM is old TensorFlow code. The working local environment is recorded in
`configs/env/pps_classic.txt`:

```bash
conda create -n pps-classic python=2.7 -y
conda install -n pps-classic -c conda-forge tensorflow=1.4 numpy=1.16 six bleach=1.5 -y
```

The HEM CLI sanity check passed:

```bash
conda run -n pps-classic python baselines/pps_classic/hem_official/HEM/main.py --help
```

TEM/ProdSearch imports under the repository's base PyTorch environment and the
CLI help path passed:

```bash
python baselines/pps_classic/prodsearch_tem/main.py --help
```

### Local compatibility patch

The tracked upstream tree has narrowly scoped compatibility and input-boundary
patches. For current PyTorch, its own locally generated checkpoint is loaded with
`weights_only=False`.  The upstream checkpoint contains an `argparse`
namespace and optimizer object in addition to tensor weights, which the newer
safe weights-only default cannot deserialize.  No model, objective, data,
selection, scoring, or ranking behavior is changed, and only checkpoints
created by this local run are eligible for loading.
The legacy byte attention mask is converted to boolean at `masked_fill`, as
required by current PyTorch; the mask values and attention computation are
otherwise unchanged.
For an all-empty-history inference batch, the adapter inserts the model's
existing product padding index before tensorization.  Its mask remains false,
so ZAM attends only to its native zero-attention slot and receives no synthetic
history event.

The native-format adapter also enables a local `use_review_query_idx` boundary
mode.  The original benchmark stores a product-level query union, whereas the
unified PPS records contain the exact query for each request/interaction.  In
this mode the unchanged model receives that explicit per-interaction query for
train, validation, and scoring.  It prevents cross-request query expansion and
does not alter ZAM attention, loss, optimization, or scores.

## External Alignment State

HEM Path 1 is active. The Amazon Cell Phones & Accessories benchmark was
reconstructed from:

- QingyaoAi Amazon Product Search benchmark commit
  `64907e59b4ce27738e61607f2d6b16b62dee92ac`;
- SNAP/McAuley raw review file
  `reviews_Cell_Phones_and_Accessories_5.json.gz`;
- SNAP/McAuley raw metadata file
  `meta_Cell_Phones_and_Accessories.json.gz`.

Raw and processed data live under `artifacts/batch2b/` and are not tracked.
The reproducible split manifest is `reports/b6o_hem_benchmark_split_manifest.json`.

The current HEM official-parameter external alignment run uses:

```text
data_dir=artifacts/batch2b/b6o_amazon_alignment/cellphones/indexed/min_count5/
input_train_dir=artifacts/batch2b/b6o_amazon_alignment/cellphones/indexed/min_count5/official_query_split/
similarity_func=cosine
net_struct=simplified_fs
embed_size=100
window_size=3
max_train_epoch=20
negative_sample=5
subsampling_rate=0.0
L2_lambda=0.005
batch_size=64
```

No KuaiSearch dev/test records or qrels are read during external alignment.
