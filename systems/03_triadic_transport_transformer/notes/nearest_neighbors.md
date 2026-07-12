# C03 Nearest-Neighbor Audit

Search date: 2026-07-11 (before any C03 dev outcome).

Verdict locked for the screening: **`uncertain` global novelty, but not
currently reducible at the full fingerprint level**.  Generic OT, Sinkhorn,
cycle consistency, dustbins, and Transformer recommendation are all prior art.
The testable delta is their candidate-anchored, null-intersection information
flow inside a ranking Transformer.  If later review finds that exact operator,
C03 must stop or pivot before another dev call.

## Mechanism table

| Neighbor | Primary paper / official code | What it already covers | Locked observable difference and rent-paying control |
|---|---|---|---|
| DIN target attention | [KDD 2018 paper](https://www.kdd.org/kdd2018/accepted-papers/view/deep-interest-network-for-click-through-rate-prediction), [author code](https://github.com/zhougr1993/DeepInterestNetwork) | Candidate-conditioned one-way weighting of history. | C03 constrains both sides of three partial plans, can reject mass to null, and requires a direct `q↔c`-consistent intersection. `softmax` degeneration must lose corruption selectivity. |
| UBR4CTR / SIM retrieval | [UBR4CTR paper](https://arxiv.org/abs/2005.14171), [official code](https://github.com/qinjr/UBR4CTR), [SIM paper](https://arxiv.org/abs/2006.05639) | Retrieve candidate-relevant history, then predict/attend. | C03 never hard-retrieves events; all events and null remain in a differentiable mass-conserving plan. Compare the optional mean/retrieval-like control; no hidden top-k branch is allowed. |
| ZAM | [paper](https://arxiv.org/abs/1908.11322) | Query attention with a zero vector that can suppress personalization. | ZAM has one query-to-history normalization and a zero vector; C03 has three capacity-constrained partial plans whose real mass must intersect. `no_cycle` and `softmax` separately remove this delta. |
| TEM | [paper](https://arxiv.org/abs/2005.08936), [official family code](https://github.com/kepingbi/ProdSearch) | Transformer over query and purchase history with dynamic personalization. | TEM self-attention does not expose conserved match mass, a learned dustbin for every pair, or candidate-anchored three-way agreement. Same Transformer states plus `softmax` is the matched degeneration. |
| RTM | [paper](https://arxiv.org/abs/2004.09424), [official family code](https://github.com/kepingbi/ProdSearch) | Transformer fine-grained query/user-review/item-review matching. | RTM uses contextual attention, not partial transport with shared non-null cycle mass. `softmax` tests whether context alone explains C03. |
| SASRec / BERT4Rec | [SASRec paper](https://arxiv.org/abs/1808.09781), [official code](https://github.com/kang205/SASRec), [BERT4Rec paper](https://arxiv.org/abs/1904.06690), [official code](https://github.com/FeiSun/BERT4Rec) | Transformer sequence representation and next-item/Cloze objectives. | They do not jointly condition transport on the current query and candidate or admit null evidence. `mean_pool`/Transformer-only behavior is the capacity control. |
| Sparse Sinkhorn Attention / Sinkformers / ESPFormer | [Sparse Sinkhorn](https://arxiv.org/abs/2002.11296), [Sinkformers](https://arxiv.org/abs/2110.11773), [official Sinkformers code](https://github.com/michaelsdr/sinkformers), [ESPFormer](https://proceedings.mlr.press/v267/shahbazi25a.html) | Sinkhorn for differentiable sorting or doubly-stochastic attention. | These replace/structure attention but do not implement request-level three-role partial matching or a dustbin-gated ranking residual. `no_null` checks whether doubly-stochastic normalization alone suffices. |
| SuperGlue dustbin matching | [CVPR 2020 paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Sarlin_SuperGlue_Learning_Feature_Matching_With_Graph_Neural_Networks_CVPR_2020_paper.html), [official code](https://github.com/magicleap/SuperGluePretrainedNetwork) | End-to-end pairwise partial assignment with a learned dustbin and Sinkhorn. | C03 borrows this known pairwise device; novelty cannot rest on the dustbin. It requires three semantic roles and intersection mass that is the only history-to-logit path. `no_cycle` removes that addition. |
| Unbalanced/null word alignment | [ACL 2023 paper](https://aclanthology.org/2023.acl-long.219/) | OT/partial/UOT null alignment for unmatched tokens. | Pairwise null alignment is known. C03 predicts that null mass changes under history/query perturbations and couples that diagnostic to candidate ranking through a third role. `no_null` must fail that prediction. |
| Optimal Multiple Transport / multi-marginal OT | [OMT manuscript](https://openreview.net/pdf?id=3P87ptzvTm), [unbalanced multi-marginal OT](https://doi.org/10.1007/s10851-022-01126-7) | Three-or-more transports, entropy regularization, and cycle consistency are known mathematical objects. | This finding forced the pre-result pivot. C03 does not claim generic cyclic OT; it uses candidate-anchored partial plans and a Bhattacharyya-style real-mass intersection as the ranking bottleneck. A generic no-null/no-cycle formulation is an explicit degeneration. |
| OT for recommendation/alignment | [partial relaxed OT recommendation](https://arxiv.org/abs/2204.08619), [OT ratings/text alignment](https://proceedings.mlr.press/v286/tran25a.html), [RecGOAT 2026](https://arxiv.org/abs/2602.00682) | User-item denoising, preference-factor alignment, and LM/ID distribution alignment with OT. | These operate globally or across modalities, not per request over query/history/candidate states with a null-gated residual. Their existence prevents any claim that OT-in-recommendation itself is novel. |
| LLM4Rec family | [P5](https://arxiv.org/abs/2203.13366), [P5 code](https://github.com/jeykigung/P5), [TALLRec](https://arxiv.org/abs/2305.00447), [A-LLMRec](https://arxiv.org/abs/2404.11343) | Language-model prompting/tuning and collaborative-semantic alignment for recommendation. | C03 is a discriminative fixed-candidate ranker with zero online LLM calls; the Transformer is load-bearing and transport changes its hidden-state path. Prompt-only or frozen embedding + MLP variants are forbidden. |

## Three claimed additions and observable predictions

1. **Three-way consistency.** A true transferable event must receive both
   `q↔h` and `h↔c` mass while the query and candidate also match.  Removing
   cycle intersection should preserve capacity but weaken query-mask and event
   shuffle sensitivity.
2. **Mass conservation.** One event cannot receive unlimited independent
   attention from all candidates.  Replacing partial transport by row-softmax
   should change row/column sums and allow diffuse/duplicated evidence.
3. **Null sink.** Unsupported evidence is not forced onto a real event.  Wrong,
   shuffled, query-masked, and coarse-only inputs should increase null mass;
   the no-null control cannot satisfy that prediction by definition and must
   not reproduce the primary score drop pattern.

Names and component combinations are not treated as evidence of novelty.
