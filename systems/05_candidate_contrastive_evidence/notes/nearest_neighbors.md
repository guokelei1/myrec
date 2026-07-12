# C05 nearest-neighbor audit

Status: pre-outcome audit, 2026-07-11.  Only primary paper/official project
sources are used below.  The current novelty verdict is **uncertain**.

## Mechanism comparison

| Neighbor | Closest overlap | Non-reducible C05 claim to test | Required degeneration |
|---|---|---|---|
| [DIN](https://arxiv.org/abs/1706.06978) | target item conditions history aggregation | CCEB first removes support common across the request's candidates and permits signed abstaining mass; DIN builds one positive target-aware user vector | `target_softmax` |
| [SIM](https://arxiv.org/abs/2006.05639) / [UBR](https://arxiv.org/abs/2005.14171) | candidate-conditioned history retrieval | CCEB does no external top-k retrieval and changes an internal Transformer residual only through within-candidate-set relative evidence | target-attention/retrieval control if G1 survives |
| [ZAM](https://arxiv.org/abs/1908.11322) | personalization can choose a zero vector | CCEB has no zero expert competing in a simplex; generic support cancels across candidates and signed evidence can promote or demote | `target_softmax` plus zero token |
| [TEM](https://arxiv.org/abs/2005.08936) | Transformer jointly models query and purchase history | TEM's ordinary sequence attention can personalize without candidate-relative evidence; CCEB changes one attention normalization and injects candidate-specific residuals | ordinary Transformer block |
| [RTM](https://arxiv.org/abs/2004.09424) | fine-grained query/user/item interaction in a Transformer | RTM contextualizes reviews but does not impose per-event candidate-set common-mode cancellation or a strict signed L1 budget | ordinary joint self-attention |
| [Denoising Attention](https://arxiv.org/abs/2308.15968) | ReLU threshold can filter every history source and return a zero context | it thresholds query/user alignment independently; CCEB thresholds support *after centering the same event across candidates* and uses signed mass | `no_candidate_contrast` |
| [Differential Transformer](https://arxiv.org/abs/2410.05258) | subtracting attention maps cancels common noise | Diff Transformer subtracts two learned softmax maps for each token; CCEB subtracts the candidate-set mean for the same event, then applies a strict evidence budget tied to ranking candidates | two-map differential-attention control if needed |
| [ReLA](https://arxiv.org/abs/2104.07012) | rectified, non-softmax attention can be sparse | ReLA changes generic Transformer normalization but has neither triadic candidate contrast nor the per-candidate evidence budget | rectified-attention control |
| [GSF](https://arxiv.org/abs/1811.04415) / [PRM](https://arxiv.org/abs/1904.06813) | scores depend on other candidates in a list | CCEB uses candidate context only to decide whether a history event is discriminative; it is not a generic groupwise scorer or post-ranking list Transformer | history-free groupwise control |
| [SetRank](https://arxiv.org/abs/1912.05891) / [Context-Aware LTR](https://arxiv.org/abs/2005.10084) | permutation-equivariant candidate-set interactions | CCEB's candidate context modulates history evidence rather than generic document context, but candidate-context gains must be separated explicitly | history-free groupwise and nested-pool controls |
| [STARank](https://arxiv.org/abs/2308.02860) | listwise candidate context is combined with user history | CCEB claims per-event common-mode cancellation and abstention rather than a general set-to-list scorer; this difference is unproven | STARank-style groupwise-history control |
| [SASRec](https://arxiv.org/abs/1808.09781) / [BERT4Rec](https://arxiv.org/abs/1904.06690) | Transformer models sequential history | CCEB is query- and candidate-set-conditioned at scoring time and can structurally abstain; generic next-item attention is the backbone control | query/history Transformer control |
| [HSTU / Generative Recommenders](https://arxiv.org/abs/2402.17152) | target-aware pointwise attention inside a scalable sequential Transformer | CCEB's claim is evidence fidelity on a fixed query-conditioned candidate set, not long-history scaling or generative transduction | matched pointwise-attention block |

## Audit conclusion

Thresholding, abstention, candidate-aware history attention, signed/differential
attention, and groupwise ranking all have direct precedents.  C05 therefore
must not claim any of those ideas individually as novel.

The only potentially distinct operator is their narrow information-flow
constraint:

> the same history event receives no ranking budget unless its joint
> query/candidate support differs from its support for the other candidates in
> that request; surviving support is signed and has strict L1 mass below one.

This may still prove equivalent in effect to Denoising Attention plus a
groupwise scorer.  If `no_candidate_contrast`, `target_softmax`, or a matched
history-free groupwise control reproduces the gain, the mechanism claim is
closed even if C05's aggregate metric is positive.
