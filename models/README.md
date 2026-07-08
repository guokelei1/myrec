# Models

Local model state. Ignored by Git except for this README.

Use it for:

- downloaded model weights (encoder, reranker, LLM);
- trained checkpoints;
- embeddings;
- vector indexes;
- tokenizer/model caches when they should stay inside the workspace.

Record only small checkpoint references, hashes, and selection decisions
in tracked experiment summaries under `experiments/` or `reports/`.
