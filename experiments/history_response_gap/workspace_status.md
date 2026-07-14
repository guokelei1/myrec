# Workspace readiness — 2026-07-14

## Ready now

- current doc 34 scientific plan and E0 review draft;
- dataset-independent standardized-record and label-isolation validator;
- shared candidate-hash ranking evaluator;
- true/null/wrong score-bundle contract and shared history-response metrics;
- hand-computed tests for common-mode, candidate-relative activity,
  directional accuracy, signed alignment and incremental NDCG;
- a four-A40 local environment with the current PyTorch/Transformers stack;
- preserved legacy source and outputs for read-only, selective reference.

## Still blocking an actual binding experiment

- KuaiSearch Full raw files are not present; the local KuaiSearch source is
  Lite only;
- KuaiSAR Full is not present;
- the local JDsearch directory contains the repository/schema boundary, not a
  confirmed full admitted track;
- BGE-reranker-v2-m3 and Qwen3-Reranker-0.6B are not in the current local
  cache;
- no dataset admission card, standardized Full version, endpoint/MDE lock, or
  model-family budget has been frozen.

## Readiness decision

The repository is ready to begin **E0 implementation and the optional E-1
instrumentation pilot**. It is not yet ready to launch E1/E2 training or any
binding dev/confirmation evaluation. No wholesale archive restoration is
needed; use `archive_reuse_policy.md` when a specific old utility becomes
necessary.
