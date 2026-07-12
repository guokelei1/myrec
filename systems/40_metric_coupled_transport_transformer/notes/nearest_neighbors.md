# C40 nearest-neighbor audit

| Neighbor | Covered idea | C40 falsifiable boundary |
|---|---|---|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Multi-head Q/K/V attention with independent projections | C40 couples selection, value, transport, and final candidate readout through one map |
| [Do Transformers Need Three Projections?](https://arxiv.org/abs/2606.04032) | Explicit QKV projection-sharing variants | C40 does not claim QKV sharing; `shifted_loop` tests ranker-side closure beyond local sharing |
| [Rethinking QKV Embedding in ViT](https://arxiv.org/abs/2111.10017) | Partly or fully shared nonlinear QKV embeddings | Candidate readout and transported-query score remain outside that sharing boundary |
| [Hopfield Networks is All You Need](https://openreview.net/forum?id=tL89RnzIiCd) | Attention as associative retrieval in one learned energy geometry | C40 must beat one-wide coupled retrieval and cannot claim energy or associative memory as novelty |
| [FiLM](https://ojs.aaai.org/index.php/AAAI/article/view/11671) | Conditional feature modulation | C40 has no pooled-history-conditioned affine map; `single_wide_coupled` is the closest low-rank metric reduction |
| C15 candidate-conditioned values | Pair values reduce to FiLM or generic edge messages | C40 has no candidate/event pair map; the candidate enters only through tied metric readout |
| C16 mixed-gradient energy | Tied attention is covered by Hopfield/energy formulations | C40 does not claim tied attention alone; only a positive matched loop-closure result could survive |
| C38 unprojected transport | One shared adapted metric already carries transferable history signal | C40 must pay rent over this strong one-head predecessor on untouched data |
| C39 halfspace value | Candidate-relative post-selection value constraint | C40 tests the earlier decoupling point and has no projection or candidate-local value law |

Verdict before outcomes: **uncertain but testable**. A synthetic recovery pass
proves only conditional capacity. A real claim requires stable gain over C38
and all same-parameter reductions plus true-over-wrong specificity.
