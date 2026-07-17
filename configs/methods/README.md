# Current Motivation method configs

These five files are the frozen first-round V1.2 baseline recipes:

| Role | Config |
|---|---|
| Q0 Qwen3-Reranker anchor | [`kuaisearch_motivation_v12_q0_qwen3_reranker_06b.yaml`](kuaisearch_motivation_v12_q0_qwen3_reranker_06b.yaml) |
| Q1 InstructRec-style GeneralQwen | [`kuaisearch_motivation_v12_q1_instructrec_generalqwen.yaml`](kuaisearch_motivation_v12_q1_instructrec_generalqwen.yaml) |
| Q2 RecRanker-style GeneralQwen | [`kuaisearch_motivation_v12_q2_recranker_generalqwen.yaml`](kuaisearch_motivation_v12_q2_recranker_generalqwen.yaml) |
| Q3 TALLRec-style GeneralQwen | [`kuaisearch_motivation_v12_q3_tallrec_generalqwen.yaml`](kuaisearch_motivation_v12_q3_tallrec_generalqwen.yaml) |

The shared implementation is in
[`src/myrec/baselines/motivation_v12_ranker.py`](../../src/myrec/baselines/motivation_v12_ranker.py)
and [`motivation_v12_contracts.py`](../../src/myrec/baselines/motivation_v12_contracts.py).
Their result-producing authority is the frozen
[`protocol.yaml`](../../experiments/motivation/protocol.yaml). Diagnostic reuse
is governed by the current
[`mechanism analysis plan`](../../experiments/motivation/mechanism_analysis_plan.md).

W0 is intentionally separate because it is a non-LLM structural witness:
[`../baselines/kuaisearch_motivation_v12_copps_transfer_witness.yaml`](../baselines/kuaisearch_motivation_v12_copps_transfer_witness.yaml).
