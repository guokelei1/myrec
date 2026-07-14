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
| KuaiSearch Full | Binding main track after source/collision/power admission |
| KuaiSAR Full | Functional behavioral replication without a plaintext-semantic claim |
| JDsearch | Pre-registered replication fallback within its anonymized information boundary |
| Amazon-C4 + history companion | Non-binding semantic stress test and legacy pilot |

The existing Amazon-C4 raw files do not make it a binding natural-search
track. The JDsearch GitHub repository contains only tiny schema samples; the
full archive still requires the upstream JD Cloud interactive download and
must not be replaced by those samples in experiments. KuaiSearch Lite is
legacy pilot material and does not substitute for the E0 Full audit.

Raw source datasets are retained for the new direction. Existing
`interim/` and `standardized/` outputs from the previous exploration are
legacy material and are archived before a new E0-approved standardized version
is created.

Generated standardized files (`records_train.jsonl`, `records_dev.jsonl`,
`records_test.jsonl`, `item_catalog.jsonl`, `manifest.json`) belong here and
are not committed.

Small provenance summaries, checksums, or dataset cards can be promoted
to `doc/`, `experiments/`, or `reports/` when useful for reproduction.
