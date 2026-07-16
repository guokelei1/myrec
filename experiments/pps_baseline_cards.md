# Active baseline boundary cards

These cards describe controls available to the history-response validation.
They do not authorize an architecture or preserve the old B0–B9 leaderboard.

| Role | Candidate implementation | Information boundary | Status |
|---|---|---|---|
| sanity | popularity / source order / random | request and candidate metadata only | reusable |
| lexical | BM25 | query and candidate text | reusable |
| dense | cross-encoder / dense reranker | query and candidate text | reusable |
| sequential | RecBole or SASRec-style control | history identity plus declared fields | reusable control |
| personalized search | ProdSearch or PPS-classic adapter | query, history, candidate fields | control candidate |
| ordinary history model | E-FULL / D-FULL | full standardized record | to be frozen after E0 |
| signal witness | train-only cross-fitted predictor | same information boundary | diagnostic only |

## Current and diagnostic model boundaries

The current three-model observation is governed by
`doc/40_transformer_recurrence_transfer_motivation_v1_zh.md`. HSTU, SASRec,
LLM-SRec, ZAM, MAPS, and HMPPS remain diagnostic or implementation-boundary
cards only; they are not active plans and do not authorize a proposed model.

| ID | Architecture role | Source boundary | PPS adaptation boundary | Status |
|---|---|---|---|---|
| QWEN-QC/FULL | ordinary decoder language Transformer | `Qwen/Qwen3-Reranker-0.6B` | raw query/history/candidate token serialization; matched QC/FULL | Full-source two-seed exploration complete and task-adequate; fresh matched QC/FULL on the disjoint confirmation population passed all five frozen motivation gates |
| HSTU-QC/FULL | rec-native sequential transducer | official Generative Recommenders at `6135bc30398f97e5786674192558d91f2ef2fa90`, Apache-2.0 | official HSTU core plus query-conditioned fixed-slate adapter | Full-source matched bundle complete; response symptom supportive, but QC significantly below strong BGE-QC and target-repeat positive control failed |
| SASREC-MATCHED | HSTU architecture-specific control | SASRec implementation in the same official tree | identical fields, dimensions, scorer, and training budget as HSTU | Lite matched bundle complete; QC below BM25, so control only |
| LLMSREC-FULL | KDD 2025 sequence-enhanced LLM4Rec | paper mechanism; official source audited at `b81019ca655fb759cee895924b8b6c7cc0f0cce9` | frozen Qwen-0.6B + frozen train-only SASRec teacher + independently implemented lightweight alignment for query-conditioned fixed-slate scoring | four train-internal recipes selected `lr=1e-4, epochs=2, distillation=0.5`; tuned confirmation two-seed true/null/wrong complete under the shared evaluator, but endpoint adequacy and reliable target-aware evidence gates fail; HSTU remains deferred |
| QWEN3-RERANKER | advanced decoder-only LLM query/history/candidate ranker | `Qwen3-Reranker-0.6B`, local pretrained checkpoint | official prompt boundary with matched QC/FULL pointwise BCE adapter | two-seed exploration plus one fresh frozen confirmation; FULL true graded NDCG@10 `0.19826` all / `0.40174` positive; repeat `+0.23150`, no-overlap nonrepeat `+0.00324`, contrast `+0.22826` |
| TEM | SIGIR 2020 exact-task PPS Transformer | official ProdSearch tree at `449335ba652fe7c877a008e154157d7b2a4b0e76`, Apache-2.0 | official item-transformer core plus exact-query unified-record adapter | confirmation true graded NDCG@10 `0.18124` all / `0.36726` positive; overall true-null `+0.00032`, repeat `+0.05716`, no-overlap nonrepeat `-0.01633`, contrast `+0.07349`; exact frozen candidate manifest and shared evaluator |
| INSTRUCTREC | TOIS LLM instruction-following personalized-search boundary | paper mechanism; original 3B Flan-T5-XL | natural-language query/history instruction and candidate-likelihood reranking on the fixed slate | formal Flan-T5-XL complete; true graded NDCG@10 `0.17911` all / `0.36295` positive; overall true-null `+0.00061`, repeat `+0.03375`, no-overlap nonrepeat `-0.00017`, contrast `+0.03392`; wrong-history and 2048-token boundary audited |
| MAPS | ACL 2025 LLM-assisted motivation-aware PPS | official code `E-qin/MAPS@e67d599e9d45bc0f61fd11cc6d799773a1316114`, CC BY-NC 4.0; paper’s processed Amazon/PersonalWAB data boundary | query/history/item embedding fusion, MoAE, and query-aware history attention; consultation branch unavailable in KuaiSearch | high-fit conditional structural adaptation; official code and modality boundary audited, no KuaiSearch result claimed |
| HMPPS | ACM Multimedia 2025 multimodal LLM PPS reranker | paper mechanism; reported InternVL2-1B with offline Qwen2.5-14B summarization | query-aware history selection, yes/no likelihood, and multimodal product refinement | high-fit conditional candidate; image coverage and public-code audit pending; text-only execution cannot be called full HMPPS |
| E-QC/FULL | ordinary encoder full-token anchor | `BAAI/bge-reranker-v2-m3` | existing matched QC/FULL cross-encoder | cross-dataset exploratory evidence available |
| ZAM | classic query-selective personalized product search | official ProdSearch tree at `449335ba652fe7c877a008e154157d7b2a4b0e76`, Apache-2.0 | exact frozen query/history/candidates through a native-format adapter; deterministic fillers removed before shared evaluation | Full-source two-rate train-validation selection and exact bundle complete; diagnostic only because task adequacy and repeat positive control failed |

ZAM uses one compatibility-only upstream patch: current PyTorch explicitly
loads the run's own compound checkpoint with `weights_only=False`; the
architecture, training objective, model selection, and scorer are unchanged.
The adapter also switches query lookup from the upstream product-level query
union to the exact query id already stored on each unified interaction; this is
an input-boundary alignment, not a model change.

For each frozen run, record upstream commit, local adapter status, input
fields, candidate manifest hash, qrels boundary, seed, budget, and evaluator
run ID. Historical cards and results are in the dated legacy source archive.

The confirmation Qwen boundary is frozen in
`configs/baselines/kuaisearch_confirmation_qwen3_pointwise.yaml`: one epoch,
seed `20260714`, six recent events for FULL, no history for QC, identical 14,764
training examples and 923 optimizer steps, and no confirmation-label access
during training or scoring. Its decision is
`reports/pps_motivation_confirmation_decision.json`. This card authorizes no
additional confirmation tuning or proposed architecture.
