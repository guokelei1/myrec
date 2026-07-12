# C09 Primary-Source Nearest-Neighbor Audit

Scope: architecture-level prior art found from primary papers, author-hosted
papers, official proceedings, and official publication pages.  This search can
falsify broad novelty claims but cannot prove exhaustive novelty.

## Audit table

| Neighbor | What the primary source already establishes | Difference from C09 | Risk |
|---|---|---|---|
| [Transformer](https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html) | End-to-end self-attention and multi-head attention as the sequence core. | Transformer attention itself is not a C09 contribution; C09 proposes only the residual-margin-derived permission matrix and view barriers. | High if the claim is phrased as “attention-based reasoning.” |
| [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html) | Attention over sets and permutation-aware set modeling. | Candidate-set interaction and permutation equivariance are inherited controls, not novelty. CMA changes how pairwise attention is permitted. | High if the claim is merely “candidate-set attention.” |
| [DIN](https://www.kdd.org/kdd2018/accepted-papers/view/deep-interest-network-for-click-through-rate-prediction) | Candidate-conditioned activation of user history. | DIN is the nearest neighbor of C09's C-first path. It has no candidate-blind Q-first path and no cross-view residual-margin conjunction. C-first-only is mandatory. | High. |
| [ZAM](https://www.amazon.science/publications/a-zero-attention-model-for-personalized-product-search) | Personalized product search can adaptively reduce personalization by attending to a zero option. | ZAM already owns broad “when/how much to personalize” language. C09 must show a non-diagonal candidate-margin mechanism surplus over scalar/zero attention. | High. |
| [TEM](https://arxiv.org/abs/2005.08936) | A Transformer can jointly encode query and purchase history and dynamically control personalization, including history-item interactions. | C09 cannot claim that Transformer history control is new. It uses two shared restricted paths and exact disagreement blocking at candidate residual margins. | Very high; strongest domain collision. |
| [PSMIM](https://aclanthology.org/2024.ccl-1.76/) | Hierarchical Transformers and auxiliary objectives model associations among current query and historical search information; the current-query representation is formed without a candidate in the cited path. | C09's candidate-blind Q-first mediator is not alone novel. The candidate-first mirror and inference-time pairwise conjunction are the remaining distinction. | Medium-high. |
| [Cross-View Training](https://aclanthology.org/D18-1217/) | Auxiliary modules with restricted input views learn to match a full model while sharing intermediate representations. | CVT uses restricted views as a semi-supervised training regularizer. C09 uses deterministic causal masks at inference and does not train views to agree. | High for “restricted shared views”; medium for the exact operator. |
| [R-Drop](https://openreview.net/forum?id=bw5Arp3O3eY) | Two stochastic submodels are explicitly made consistent with bidirectional KL. | C09 views are evidence-path restricted, not dropout samples; no KL agreement reward is used; disagreement blocks an inference-time history update. | Medium. |
| [Multi-view learning in the presence of view disagreement](https://proceedings.mlr.press/r6/christoudias08a.html) | View disagreement can be detected and disagreeing examples filtered before multi-view learning. | “Ignore disagreement” is old. C09 filters directed candidate-pair history corrections inside a shared Transformer while retaining a base ranker, not whole examples. | Very high for the conceptual slogan. |
| [Adaptive Mixtures of Local Experts](https://www.cs.toronto.edu/~fritz/absps/jjnh91.pdf) | A gating network allocates inputs among separate expert networks. | C09 has one parameter-shared ranker, never selects a view, and never mixes expert outputs. Global/diagonal MoE-style gates remain mandatory controls. | Medium. |
| [Products of Experts](https://www.cs.toronto.edu/~fritz/absps/tr00-004.html) | Multiple models can be combined by multiplying and renormalizing their probabilities so all constraints matter. | C09 never multiplies/renormalizes view distributions. Its positive parallel sum only creates a sparse attention permission; the base score remains separate. PoE fusion is a control. | Medium. |
| [Multi-Head Attention with Disagreement Regularization](https://aclanthology.org/D18-1317/) | Attention heads can be explicitly regularized for diversity/disagreement in subspaces, positions, and outputs. | It optimizes head diversity, whereas C09 requires two causal paths to agree before a history update. It nevertheless shows that relations between attention heads are established design territory. | Medium. |

## Nearest-neighbor verdict

The literature audit rejects all broad novelty statements:

- multi-view agreement is not new;
- filtering disagreement is not new;
- restricted views with shared representations are not new;
- candidate-conditioned history attention is not new;
- dynamic control of PPS personalization with a Transformer is not new;
- set attention and permutation equivariance are not new.

The only potentially distinguishable mechanism is the conjunction of these
precise choices:

```text
shared Q-first/C-first causal paths
  -> history residuals against one base
  -> ordered candidate-margin parallel-sum meet
  -> non-diagonal candidate contrast attention
  -> exact base fallback on disagreement.
```

That combination has **high innovation risk**.  A synthetic probe is worth the
small CPU budget only to test whether this exact structure has behavior beyond
single-view attention and matched scalar/diagonal gates.  It is not yet worth a
dev evaluation or full implementation.  A later formal literature review must
search more recent recommendation, robust multi-view, and dynamic-attention
work before any publication novelty claim.
