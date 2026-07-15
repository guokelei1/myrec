# LLM-SRec baseline boundary

This directory intentionally contains no upstream implementation.

## Source audit

- Paper: *Lost in Sequence: Do Large Language Models Understand Sequential
  Recommendation?*, KDD 2025
- Paper: `https://arxiv.org/abs/2502.13909`
- DOI: `10.1145/3711896.3737035`
- Official repository: `https://github.com/Sein-Kim/LLM-SRec`
- Audited commit: `b81019ca655fb759cee895924b8b6c7cc0f0cce9`
- License finding on 2026-07-15: no `LICENSE` or equivalent grant was present
  at the repository root.

Because the audited source has no explicit license, its code is not vendored,
copied, or patched here. The temporary audit checkout is local experiment state
under ignored `tmp/` and is not an implementation dependency.

## Independent implementation

The PPS baseline will independently implement the mechanism described in the
paper:

- a sequential recommender pretrained on training data and then frozen;
- a frozen language backbone;
- lightweight projections that align/distill the sequence representation into
  the language-side user/item representation;
- retrieval/ranking and representation alignment objectives;
- fixed-slate candidate scoring through the shared PPS interface.

For the first resource-controlled implementation, the language backbone is
`Qwen/Qwen3-Reranker-0.6B` rather than the paper's LLaMA-3.2-3B. The true query
is included in the user-side context, and candidates come from the dataset's
fixed request slate. These are explicit PPS task adaptations, so the result is
called **paper-mechanism-faithful, task-interface-adapted LLM-SRec**. It must not
be reported as reproduction of the paper's original numbers.

The independent mechanism source is now implemented in
`src/myrec/baselines/llm_srec_adapter.py`. It contains the paper's fixed-slate
retrieval loss, MSE user-representation distillation, hypersphere uniformity
loss, CF item embedding injection, and frozen-Qwen user/item output-token
extraction. The true query is added only as the declared PPS task adaptation.

The real-record mechanics smoke is
`reports/llm_srec_pps_adapter_lite_smoke.json`. It confirms that the Qwen
backbone remains frozen and lightweight modules receive finite gradients, but
uses deterministic synthetic CF representations and is not scientific model
evidence. A trained SASRec teacher and real frozen content features remain
required before adequacy evaluation.

Generated teachers, checkpoints, embeddings, runs, and score dumps belong under
ignored project state directories.

## Evaluation contract

The implementation must use the same standardized JSONL records, candidate
hashes, score bundle, shared evaluator, true/null/wrong counterfactuals, and
base/history accounting as Qwen, HSTU, and BGE. Training may read train qrels;
model code must never read development/test qrels.
