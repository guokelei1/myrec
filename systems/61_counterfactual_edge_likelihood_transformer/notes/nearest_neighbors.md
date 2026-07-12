# C61 nearest-neighbor audit

Status: pre-outcome; global novelty unestablished.

| Neighbor | Shared mechanism | Binding distinction/control |
|---|---|---|
| RankNet / Bradley-Terry | Pairwise logistic ranking with a base-margin prior | Pairwise learning is not new. `candidate_only_edge` tests generic pair capacity; C61's bounded claim requires the same-network history likelihood ratio to pay rent. |
| Differentiable Sorting Networks ([Petersen et al., 2021](https://proceedings.mlr.press/v139/petersen21a.html)) | Local compare-exchange ranking | Sorting is not claimed new. C61 retains C60's fixed edge graph/capacity and tests only the counterfactual history verifier. |
| RankRefine++ ([ICLR 2026](https://openreview.net/forum?id=b2tBRLic4V)) | Bayesian update of a strong base with calibrated pairwise evidence; identifies scale/curvature failure | This is a close conceptual neighbor. C61 does not claim Bayesian refinement novelty; it binds evidence magnitude to base-edge capacity and directly tests factual-minus-NULL personalization. |
| Pairwise Ranking Prompting ([Qin et al., 2024](https://aclanthology.org/2024.findings-naacl.97/)) | LLM pairwise comparison and efficient aggregation | C61 is a trained internal LLM4Rec ranker, not prompt-based inference. Pairwise LLM scoring remains prior art. |
| PSMIM / MAPS | Transformer history representation and personalized-search alignment | `ordinary_candidate_attention` tests whether the edge counterfactual adds anything beyond history attention. C61 uses no consultation-specific or dataset-specific fields. |
| C28 / C55 / C56 | Margin-local learned comparison, residual target, factual-null token contrast | C28's free comparator was seed-nonidentifiable; C55 predicted full-list residuals; C56 formed a candidate-common carrier. C61 targets adjacent base-error likelihood directly and caps the resulting write. |
| C60 | Identical one-sided edge interface with fixed semantic evidence | Fixed C60 is mandatory at A1; success cannot be attributed to the transport contract alone. |

The composition is only a falsifiable candidate.  A positive internal-A result
would still require a fresh second-domain confirmation and a deeper novelty
comparison; no global novelty claim is preregistered.
