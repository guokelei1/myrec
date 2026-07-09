# B6o Upstream Issue Draft

Date: 2026-07-09

Status: draft only. Do not post before user confirmation. The HEM repository
currently restricts new issue creation for unauthenticated users, and an older
open issue already asks how to feed the Amazon Product Search Dataset into HEM.

## Proposed Target

Prefer commenting on the existing HEM issue if possible:

`https://github.com/QingyaoAi/Hierarchical-Embedding-Model-for-Personalized-Product-Search/issues/1`

The Amazon Product Search Dataset repository has no open issues and exposes no
release/checkpoint package.

## Draft Text

Title:

```text
Request for original HEM indexed query_split or checkpoint for Amazon Product Search benchmark
```

Body:

```text
Hello,

I am trying to reproduce the SIGIR'17 HEM results on the public Amazon Product
Search benchmark, especially the Cell Phones & Accessories subset.

I followed the public HEM repository data-preparation path:

1. preprocess the Amazon 5-core review file with AmazonReviewData_preprocess.jar
2. run index_and_filter_review_file.py with min_count=5
3. run AmazonMetaData_matching.jar to create product_query.txt.gz
4. build HEM query_split files and train HEM with the paper-like settings

I also checked the public Amazon-Product-Search-Datasets repository. It provides:

- query_text.txt.gz
- train.qrels.gz
- test.qrels.gz
- train_review_id.txt.gz

but I could not find the original indexed query_split/ directory,
product_query.txt.gz, query.txt.gz, train/test_query_idx.txt.gz, or any trained
checkpoint used for the paper numbers.

My reconstructed split preserves the public qrels and query ids, but the HEM
metrics remain far below the reported Cell Phones & Accessories numbers. I
suspect the remaining gap is due to a split/indexing protocol difference rather
than a model-code issue.

Could you confirm whether the original HEM indexed split or checkpoint is still
available, or whether the public benchmark qrels were intended to be converted
back into HEM's query_split format in a specific way?

The most useful artifacts would be:

- query_split/ for Cell Phones & Accessories
- product_query.txt.gz and query.txt.gz from the original indexed data
- train_query_idx.txt.gz and test_query_idx.txt.gz
- the exact random seed/settings used by split_train_test_data.py
- any released HEM checkpoint for Cell Phones & Accessories

Thank you.
```

## Local Evidence Behind This Draft

- `QingyaoAi/Amazon-Product-Search-Datasets` lists only the four dataset
  directories and README.
- Its README says the public files are query text, train/test qrels, and
  train review ids.
- `QingyaoAi/Hierarchical-Embedding-Model-for-Personalized-Product-Search`
  has no releases/checkpoints.
- Existing issue #1 says the public dataset repo is not in the format expected
  by HEM and has no maintainer answer.
- Related `kepingbi/ProdSearch` and `kepingbi/ConvProductSearchNF` repos refer
  back to the same HEM/Amazon data flow; they do not expose HEM's original
  `query_split/`.
