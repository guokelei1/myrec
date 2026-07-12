# Primary-source Nearest-neighbor Audit

Audit date: 2026-07-11.  Scope: attention kernels/normalizers with negative
weights, sparsity or null attention, competition over output slots/candidates,
candidate-conditioned history, and personalized product search.  This is a
design-risk audit, not a patent search or proof of novelty.  Only primary paper
or official proceedings pages are used below.

## Results

| Work | Relevant primary-source mechanism | Similarity to C07 | Decisive difference / risk |
|---|---|---|---|
| [Attention Is All You Need](https://papers.neurips.cc/paper/7181-attention-is-all-you-need) (NeurIPS 2017) | scaled dot-product logits followed by nonnegative row-softmax | Transformer scaffold and Q/K/V aggregation | C07 replaces only the history-to-candidate normalization; ordinary softmax cannot produce signed conserved mass or an all-zero open region. |
| [Deep Interest Network](https://www.kdd.org/kdd2018/accepted-papers/view/deep-interest-network-for-click-through-rate-prediction) (KDD 2018) | local activation of history with respect to a target ad | candidate-conditioned history representation | DIN handles targets independently and is an Embedding/MLP CTR architecture.  It has no candidate-set competition, signed conservation, or kernel dead zone.  A target-attention ablation is mandatory because superficial similarity is high. |
| [A Zero Attention Model for Personalized Product Search](https://arxiv.org/abs/1908.11322) (CIKM 2019) | query/history attention includes a zero vector so personalization may be ignored | personalization abstention and query conditioning | ZAM abstains by allocating nonnegative simplex mass to a null value and builds a user profile.  C07 obtains zero from pairwise numerator shrinkage and updates jointly competing candidate states.  “Abstention” alone is not novel. |
| [A Transformer-based Embedding Model for Personalized Product Search](https://arxiv.org/abs/2005.08936) (SIGIR 2020) | Transformer encodes query plus purchase history and dynamically controls personalization | load-bearing Transformer and query/history interaction | TEM does not make fixed-set candidates compete for each history event through signed dead-zone normalization.  “Transformer PPS” is established prior art. |
| [Object-Centric Learning with Slot Attention](https://proceedings.neurips.cc/paper/2020/hash/8511df98c02ab60aea1b2356c013bc0f-Abstract.html) (NeurIPS 2020) | inputs normalize attention over slots, explicitly creating competition | closest prior art for reversing the normalization axis so outputs compete for inputs | Slot weights remain a nonnegative assignment and iterative slot update; there is no signed zero-sum candidate margin, pairwise dead zone, query condition, or ranking objective.  Candidate competition by itself is not novel. |
| [From Softmax to Sparsemax](https://proceedings.mlr.press/v48/martins16.html) (ICML 2016) and [Sparse Sequence-to-Sequence Models](https://aclanthology.org/P19-1146/) (ACL 2019) | simplex projections / entmax give exact sparse attention probabilities | exact zeros and differentiable alternatives to softmax | Sparsemax/entmax still return nonnegative unit-sum weights; at least one position receives mass.  C07 can assign negative weights and all candidates can receive zero.  Sparse normalization alone is not novel. |
| [Sparse Attention with Linear Units](https://aclanthology.org/2021.emnlp-main.523/) (EMNLP 2021) | ReLU replaces softmax; heads can attend to nothing | strongest established neighbor for structural all-zero attention | ReLA thresholds individual QK logits and does not impose candidate-relative antisymmetry or candidate conservation.  Null attention is established; C07 must show that pairwise competition, not merely ReLU sparsity, matters. |
| [Differential Transformer](https://arxiv.org/abs/2410.05258) (2024/2025) | subtracts two separately parameterized softmax attention maps to cancel common noise | signed weights, common-mode-noise motivation, Transformer-level primitive | Differential attention uses two Q/K maps and a learned scalar; C07 uses one triplet logit, pairwise candidate margins, an odd dead zone, and per-event zero-sum conservation.  Signed subtraction is not novel and is a required matched neighbor. |
| [More Expressive Attention with Negative Weights](https://arxiv.org/abs/2411.07176) (Cog Attention, preprint) | signed exponential normalization with absolute-sum stabilization | negative attention and L1-like signed normalization | Very close at the normalization level.  Cog signs individual token logits and has no candidate-pair dead zone or per-history candidate conservation.  Any C07 claim phrased merely as “signed normalized attention” is invalid. |
| [XCTFormer](https://arxiv.org/abs/2605.18534) (TMLR 2026) | CRAB uses a signed absolute-sum activation and relational mask for cross-channel/time attention | closest accepted neighbor for signed absolute-sum normalization | C07 cannot claim AbsAct/L1 signed normalization.  Its only distinct algebra is the pairwise odd shrinkage before normalization and fixed-set candidate conservation.  This substantially raises novelty risk. |
| [Spark Transformer](https://arxiv.org/abs/2506.06644) (NeurIPS 2025) | learned predictor plus statistical top-k/soft-threshold sparsifies attention | thresholded attention inside a full Transformer | Spark's goal is compute sparsity; selection is nonnegative top-k over keys.  C07 thresholds signed candidate margins for evidence fidelity and currently has worse \(O(HC^2)\) cost.  “Soft threshold inside attention” is not novel. |
| [Integral Transformer](https://arxiv.org/abs/2508.18387) (EMNLP 2025) | signed-softmax integration balances denoising against deletion of useful information | warns that negative attention can over-cancel useful content | Different construction, but a direct risk signal: C07's negative competition may suppress valid shared evidence.  Repeat preservation and false-negative abstention must be measured. |

## Reduction-oriented conclusion

The literature leaves no defensible broad novelty claim for any of the
following: signed attention, absolute-sum normalization, sparse or null
attention, query-conditioned personalization, candidate-conditioned history,
or competition over output slots.

The narrow unresolved fingerprint found in this audit is:

> Within a joint fixed-candidate ranking Transformer, apply an odd dead-zone to
> every candidate pair's query/history evidence margin, aggregate those
> pairwise wins/losses into per-history zero-sum candidate mass, and use that
> signed mass as the history-to-candidate attention normalization.

Even this may be judged an application-specific composition of Cog/XCTFormer,
ReLA, and Slot Attention.  Therefore the proposal is **high novelty risk**.  A
positive synthetic result is necessary but not sufficient; the direction must
stop if a scalar-gated centered attention or target-attention control matches
it.  Any eventual paper wording must claim a tested evidence-fidelity inductive
bias, not invention of negative/sparse/competitive attention.
