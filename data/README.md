# Data

Local dataset files. Ignored by Git except for this README.

```text
data/raw/<dataset_id>/                 downloaded source files
data/interim/<dataset_id>/             intermediate processing output
data/processed/<dataset_id>/           cleaned, join-ready tables
data/standardized/<dataset_id>/<ver>/  unified JSONL interface records
```

## Datasets

| Dataset | Role |
|---|---|
| KuaiSearch | Main track: real NL query + history + candidates + dual labels |
| Amazon-C4 + Amazon-Reviews-2023 | Secondary track: English validation, MemRerank comparison |
| JDsearch | Anchor track: robustness without plaintext text |

Generated standardized files (`records_train.jsonl`,
`records_dev.jsonl`, `records_test.jsonl`, `item_catalog.jsonl`,
`manifest.json`) belong here and are not committed.

Small provenance summaries, checksums, or dataset cards can be promoted
to `doc/`, `experiments/`, or `reports/` when useful for reproduction.
