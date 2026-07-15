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

## Representative architecture motivation matrix

This matrix is governed by
`doc/37_representative_architecture_validation.md`. It is a baseline/failure
localization exercise and does not authorize a proposed model.

| ID | Architecture role | Source boundary | PPS adaptation boundary | Status |
|---|---|---|---|---|
| QWEN-QC/FULL | ordinary decoder language Transformer | `Qwen/Qwen3-Reranker-0.6B` | raw query/history/candidate token serialization; matched QC/FULL | Lite exploratory evidence complete; Full replication pending |
| HSTU-QC/FULL | rec-native sequential transducer | official Generative Recommenders at `6135bc30398f97e5786674192558d91f2ef2fa90`, Apache-2.0 | official HSTU core plus query-conditioned fixed-slate adapter | Lite QC/FULL/true/null/wrong complete; response symptom supportive, but QC below BM25 and adequacy not passed |
| SASREC-MATCHED | HSTU architecture-specific control | SASRec implementation in the same official tree | identical fields, dimensions, scorer, and training budget as HSTU | Lite matched bundle complete; QC below BM25, so control only |
| LLMSREC-FULL | KDD 2025 sequence-enhanced LLM4Rec | paper mechanism; official source audited at `b81019ca655fb759cee895924b8b6c7cc0f0cce9` | frozen Qwen-0.6B + frozen train-only SASRec teacher + independently implemented lightweight alignment for query-conditioned fixed-slate scoring | independent mechanism, teacher store, full Lite training and true/null/wrong complete; adequacy/normal Amazon replication pending |
| E-QC/FULL | ordinary encoder full-token anchor | `BAAI/bge-reranker-v2-m3` | existing matched QC/FULL cross-encoder | cross-dataset exploratory evidence available |

For each frozen run, record upstream commit, local adapter status, input
fields, candidate manifest hash, qrels boundary, seed, budget, and evaluator
run ID. Historical cards and results are in the dated legacy source archive.
