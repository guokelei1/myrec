# Current Motivation data

The active runtime population is KuaiSearch Full:

- `raw/kuaisearch_full/`: permitted source data;
- `standardized/kuaisearch/full_confirm_preceding40k_v11/`: frozen development population;
- `standardized/kuaisearch/full_scout10k_query_history_v1/`: subsequent source-train scout;
- `standardized/kuaisearch/full_confirm_preceding40k_newholdout4k_v12/`: V1.2 confirmation holdout;
- matching intermediate assignment/materialization state under `interim/kuaisearch/`.

All generated records, manifests, and qrels remain local and are consumed only
through the shared V1.2 contracts.
