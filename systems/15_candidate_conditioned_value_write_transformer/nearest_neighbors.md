# Nearest-neighbour audit

Only primary papers or project-internal candidate records are used below.

| neighbour | already covers | C15 boundary |
|---|---|---|
| [Dynamic Filter Networks](https://papers.nips.cc/paper_files/paper/2016/hash/8bf1211fd4b7b94528899de0a43b9fb3-Abstract.html) | filters generated dynamically from an input | a candidate/event MLP generating a value transform is the same dynamic-filter pattern |
| [Dynamic Edge-Conditioned Filters](https://openaccess.thecvf.com/content_cvpr_2017/html/Simonovsky_Dynamic_Edge-Conditioned_Filters_CVPR_2017_paper.html) | per-edge filter weights generated from edge information before neighbour aggregation | joint `Phi(z_i,h_j)` is exactly an endpoint/edge-conditioned message/filter |
| [FiLM](https://ojs.aaai.org/index.php/AAAI/article/view/11671) | feature-wise conditioning by learned scale/shift | low-rank Hadamard C15 reduces to FiLM on pooled attention values |
| [Gated Attention for LLMs](https://papers.nips.cc/paper_files/paper/2025/hash/904e89bb4e632e75fb47f093b620b257-Abstract-Conference.html) | query-dependent head/elementwise gate after SDPA | diagonal candidate-conditioned value maps are directly covered |
| [DIN](https://www.kdd.org/kdd2018/accepted-papers/view/deep-interest-network-for-click-through-rate-prediction) | candidate-conditioned history allocation | C15 changes values rather than only weights, but its constrained change is post-pooling FiLM |
| [ZAM](https://arxiv.org/abs/1908.11322) | zero-vector query attention and personalization suppression | NULL/no-history is a safety neighbour, not C15 novelty |
| [TEM](https://arxiv.org/abs/2005.08936) | Transformer query/history personalization | contextual Transformer layers can already make messages endpoint-dependent; generic value MLP is not a distinct insight |
| C02 CHHT | candidate/query/history-conditioned functional Transformer update | bilinear C15 is a simpler dynamic adapter; C02's later mechanically continued gate was outcome-valid but failed because its conditioning path saturated into an almost request-common update |
| C03 triadic transport | candidate-anchored mass, dustbins, three-role intersection | C15 has no transport conservation or triadic agreement and cannot claim C03's structure |
| C06 Hodge flow | constrained conservative candidate direction and local trust | unrestricted pair values can imitate many directions but lack C06's falsifiable conservative constraint |

The closest global prior is edge-conditioned message passing, which fully covers
the only C15 form that survives FiLM factorization.  Restricting its hidden width
or placing it in a Transformer `V` path is an efficiency/application choice, not
a new mathematical primitive.
