# Motivation V1.2 baseline cards

These cards describe the frozen Q0--Q3 methods and W0 witness that establish
the current Motivation observation. Their result-producing boundary is frozen
in `experiments/motivation/protocol.yaml`; mechanism-stage reuse is governed by
`experiments/motivation/mechanism_analysis_plan.md` and `doc/motivation.md`.

## Active method provenance

All five variants below are independent, project-owned minimal
reimplementations. They share the unified KuaiSearch records, explicit input
whitelist, checkpoint/resume boundary, score-bundle contract, and evaluator.
No upstream trainer, data pipeline, checkpoint manager, or private evaluator is
used.

Q1--Q3 share the exact local `Qwen/Qwen3-0.6B` revision
`c1899de289a04d12100db370d81485cdf75e47ca` (Apache-2.0), with frozen weights,
tokenizer, and artifact-manifest SHA256 values recorded in
`experiments/motivation/protocol.yaml`. GPL-3.0 RecRanker and Apache-2.0
TALLRec source were inspected only to identify their load-bearing mechanisms;
no upstream implementation code is copied into the project-owned harness.

### Q0: Qwen3-Reranker-0.6B anchor

- Source: [Qwen3-Reranker model card](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B), [Qwen3-Embedding repository](https://github.com/QwenLM/Qwen3-Embedding/tree/44548aa5f0a0aed1c76d64e19afe47727a325b8f), and [paper](https://arxiv.org/abs/2506.05176).
- Commit and license: documentation repository commit `44548aa5f0a0aed1c76d64e19afe47727a325b8f`, Apache-2.0. The local model revision was not recorded and is therefore identified only by the frozen full-artifact and weight/tokenizer hashes in the V1.2 protocol.
- Inspected: local model card/config, official repository README, and paper Sections 2--3.2.
- Migrated mechanism: instruction-aware chat boundary and normalized yes/no relevance logits; the local adapter adds query/history context and KuaiSearch pointwise BCE.
- Omitted: official training corpus, synthesis/filtering, model merging, training/evaluation stack, and full 32K context recipe.
- Local implementation: `src/myrec/baselines/motivation_v12_ranker.py`, `src/myrec/baselines/motivation_v12_contracts.py`, shared train/score scripts, and the frozen Q0 config.

This is a specialized pretrained reranker anchor with project-owned
KuaiSearch pointwise adaptation; it is neither matched pretraining with Q1--Q3
nor an official Qwen reproduction.

### Q1: InstructRec-style GeneralQwen

- Source: [InstructRec paper](https://arxiv.org/abs/2305.07001), DOI `10.1145/3708882`. No author-maintained public code repository was identified, so code commit and code license are N/A; the paper's CC BY 4.0 status is not treated as a code license.
- Inspected: paper Sections 2.1--2.3 and Appendix C.
- Migrated mechanism: the personalized-search `P + I + T3` instruction boundary, a fixed candidate slate, and candidate-response generation likelihood; the local recipe freezes two prompt phrasings and scores the exact marked candidate line with output-only NLL.
- Omitted: Flan-T5-XL, 39 templates, 252K teacher-generated instructions, preference generation, T0/T2/CoT tasks, Amazon data, and the original evaluator.
- Local implementation: shared V1.2 Q ranker/contracts and train/score scripts plus the frozen Q1 config.

This is an independent InstructRec-style minimal reimplementation of its
P/I/T3 instruction and candidate-likelihood boundary; no official code
repository was identified.

### Q2: RecRanker-style GeneralQwen

- Source: [RecRanker repository](https://github.com/sichunluo/RecRanker/tree/f7fd1f3649fab5fd1ed510300290113c7c22787e) and [paper](https://arxiv.org/abs/2312.16018), DOI `10.1145/3705728`.
- Commit and license: `f7fd1f3649fab5fd1ed510300290113c7c22787e`, GPL-3.0.
- Inspected: repository README/license, `llm recommender/README.md`, `llm recommender/inference.py`, and paper Sections 4.2--4.6. Upstream states that its LLM training framework is not public.
- Migrated mechanism: pointwise/pairwise/listwise ranking boundary; the local shared yes/no candidate score is trained with `0.5 * RankNet + 0.5 * tie-aware ListNet`.
- Omitted: private trainer, adaptive/importance/clustering/repetition sampling, position shifting, prompt enhancement, three generated tasks and utility ensemble, LLaMA2-7B, and upstream data/evaluation systems.
- Local implementation: shared V1.2 Q ranker/contracts and train/score scripts plus the frozen Q2 config.

The joint RankNet/ListNet objective is a RecRanker-style adaptation of the
paper's multi-granularity ranking boundary. The official method instead trains
natural-language task outputs with token cross-entropy and ensembles three
task-specific utility scores at inference.

### Q3: TALLRec-style GeneralQwen

- Source: [TALLRec repository](https://github.com/SAI990323/TALLRec/tree/c1db29ce6501bce32cc8c3e343c4eb14155beeb1) and [paper](https://arxiv.org/abs/2305.00447), DOI `10.1145/3604915.3608857`.
- Commit and license: `c1db29ce6501bce32cc8c3e343c4eb14155beeb1`, Apache-2.0.
- Inspected: repository README/license, `finetune_rec.py`, `evaluate.py`, preprocessing scripts, adapter config, and paper Sections 2.1--2.2 and 3.2.
- Migrated mechanism: recommendation-alignment yes/no output-only NLL and LoRA with rank 8, alpha 16, dropout 0.05, and `q_proj`/`v_proj` targets.
- Omitted: Alpaca first-stage tuning, LLaMA-7B/8-bit stack, movie/book preprocessing, few-shot/cross-domain experiments, and upstream Trainer/evaluator/checkpoint stack.
- Local implementation: shared V1.2 Q ranker/contracts and train/score scripts plus the frozen Q3 config; the PPS adaptation adds the current query and a common causal history/candidate boundary.

This is an independent TALLRec-style recommendation-alignment and LoRA
adaptation on the shared Qwen backbone, not an official reproduction.

### W0: CoPPS-style structural transfer witness

- Source: [author-hosted paper](https://playbigdata.ruc.edu.cn/dou/publication/2023_KDD_CLPS.pdf), DOI `10.1145/3580305.3599287`. No author-maintained public code repository was identified, so commit and code license are N/A.
- Frozen feature source: `BAAI/bge-small-zh-v1.5` revision
  `7999e1d3359715c523056ef9478215996d62a620` (MIT), used only as a frozen
  item/query text encoder. Its encoder fingerprint is
  `0527b3ecdfb3a33ccda8fbd7e434f54c2e0b77bf793e34f366a07ef9fa0e6f16`.
- Inspected: paper Sections 3.2--3.4 and 4.1.4.
- Migrated mechanism: two augmented history views, different-ID/same-category semantically close item replacement, a shared downstream ranker, and in-batch InfoNCE.
- Omitted: the original BERT sequence encoder, full contrastive-pretrain then ranking-finetune pipeline, mask/reorder augmentation, KG/DREM query deletion, Amazon 5-core/automatic-query data, and upstream ranking/evaluation systems.
- Local implementation: frozen BGE item-only vectors instead of DREM/KG,
  query-aware projection/pooling, and joint ranking-plus-contrastive training in
  `src/myrec/baselines/copps_transfer_witness.py`,
  `src/myrec/baselines/frozen_text_features.py`, and
  `src/myrec/baselines/representative_sequence_adapter.py`, with the
  project-owned materialize/train/score path. The released checkpoint is the
  raw-query identity fix v2 checkpoint; its superseded partial scorer run is a
  mechanical non-result, not transfer evidence.

W0 is a non-LLM CoPPS-style structural transfer witness outside the four-LLM
main table. It is not an official reproduction and does not show that CoPPS
itself solves strict transfer.

## Current Motivation frozen first-round execution

The result-producing protocol is
`experiments/motivation/protocol.yaml` at SHA256
`6788d27cce8186be02dae4595129157fcca5032b49c1107ec83fdd2f9ecf8e43`.
All five completed checkpoints, configs, evaluator rules, holdout selection,
and history-assignment recipe were sealed before new-holdout materialization
by `20260717_kuaisearch_motivation_v12_post_selection_release_lock` at SHA256
`a4ae744d78e084685a1c14f6703fe9d4f3f05805da8f5ae2ee8d423f2a3e9d3f`.
The common pilot seed is `20260714`; no result-producing second seed was run.

| Method | Frozen checkpoint | Config SHA256 | Training status | New-4k evidence SHA256 |
|---|---|---|---|---|
| Q0 | `q0_qwen3_reranker_06b@654f929996f7eb09f7b2` | `6e9f7d93dadf2c6946049ba7290912e753ab8a6c9297116e211926748429dd66` | completed, 967 optimizer steps | `659292cc76d408578d322569fc7118cd1300c1b91f141c5d0b87a6541977fbd5` |
| Q1 | `q1_instructrec_generalqwen@9625a8c5a36327cec65f` | `1b68a3f4e79807862a0c1da369caa7b2d12218de1fe0dbf6ab2dfe18a5f15493` | completed, 967 optimizer steps | `02dc4e85066778a44746af67d04f7db58c484dfeba75e56a09b52feb21f9da67` |
| Q2 | `q2_recranker_generalqwen@e207d2213741c16f997a` | `88a463fe48e5a884e99bf72cc3522a82031194f13cdd4b98966b160378e9a11e` | completed, 967 optimizer steps | `155bda5c735435d08642df3f8156aaf49739db840bfd5a100be8c5576780d8ae` |
| Q3 | `q3_tallrec_generalqwen@ea46a89671b63741ada8` | `ea8e0fb2d3421408cc51ecc216bfcfc7c7a0524e14a594d24009c9678235bd91` | completed, 967 optimizer steps | `6359cd51a075642ed3c4b6821705659001e5c693008068503aa8e1ca34a617f9` |
| W0 | `w0_copps_style_transfer_witness@ddee4f219794be9e77f5` | `70c6a0290259f0ef1cef73b3505c111ebac7e8cddeba77c6eb6abb893709d784` | completed, 242 optimizer steps | `a168403ea2be1be34422e8cf27c244c7fd3779bdd88d842ffd93dcfb034ce6e2` |

The registered new confirmation population is
`full_confirm_preceding40k_newholdout4k_v12`, manifest SHA256
`21f4c45b4e796a3808cb0db9de066f1e3fbf8e50a2eff8a4bdf3f1bf17d8f3bb`.
Its candidate, request, and label-free record SHA256 values are respectively
`8b2c859bcd35400bed58b6df2cad4911e043a2c5ba2cac19c243392a3fff4c29`,
`5586653149a4a17fd617f0beabae842170e996713539e4481eb0a08c67352db2`,
and `828127bb611e1b7429e596a1a66977854cb9bbaf64ae443d00f1b1c32c203e8f`.
It contains 4,000 requests and 77,836 candidate rows. Every full/null/wrong
score bundle passed complete-finite candidate identity checks before the
shared evaluator opened confirmation qrels; source test was not opened.

W0 scoring used the descendant frozen feature store fingerprint
`0237c51234320bb8931a20c0b582a19748ca2abb579915f0cd8eaae0fea09be8`.
Its metadata SHA256 is
`ee50aefa2615a9b8114eaf2677e1295ab704f93f2366aa6c213dc5614581f7e0`:
764,473 base-store rows were reused bitwise and 55,633 new rows were encoded,
with `qrels_read=false`.

This holdout is retrospective: it is strictly earlier than the reused 32k
training population and has zero request/session overlap, but later training
histories can contain earlier holdout events. It supports a frozen
recipe/request/session/time-boundary confirmation, not forward-temporal or
user/item/query-isolated generalization. Wrong-user assignments include a
registered global fallback and are diagnostic rather than a
provenance-matched causal control.

The first-round status is: Q0--Q3 show reliable overall and recurrence gains
against both null and wrong-user histories, but none establishes strict
transfer on the new holdout; W0 establishes only a small recurrence response.
Exact metrics, confidence intervals, run IDs, and claim boundaries are in
`experiments/pps_results.md` and
`reports/motivation_current_summary.json`. These results are
single-seed preliminary observations, not official upstream reproductions or
multi-seed robustness claims.
