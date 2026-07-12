# C45 nearest-neighbor and reduction audit

| Neighbor | Overlap | Binding distinction / control |
|---|---|---|
| Feedback Transformer | high-level past representations feed later steps | C45 does not claim recurrence; `factual_state` is the recurrent-memory control. |
| Recurrent Memory Transformer | explicit Transformer memory passed across segments | C45's possible rent is the local factual-minus-NULL event token, not memory persistence. |
| DeltaNet / Gated DeltaNet | input-dependent recurrent memory edits | C45 does not use a delta-rule fast-weight matrix; `ordinary_delta` tests whether a normal state update suffices. |
| BERT4Rec / SASRec | contextual history-item states for sequential recommendation | `factual_state` and `raw_event` cover contextual and ordinary event-token reads. |
| CauseRec | counterfactual sequence perturbations for robust user representations | C45 synthesizes no positive training sequence and uses no counterfactual data augmentation; NULL is a shared local transition intervention. |
| classifier-free / paired-prefix guidance and C04 | conditional-minus-NULL model outputs | C45 subtracts per-event transition states before query/candidate attention; it never subtracts final candidate logits. |
| anchored functional ANOVA and C25 | subtract a NULL-anchored function evaluation | C45 uses a two-variable anchored difference at causal event formation; C25's pooled `q,c,h` third difference is an explicit local negative result. |
| C29/C30 authenticated mediation | recipient-prefix provenance controls event use | C45 uses no user-ID authentication mask; provenance can emerge only through the learned factual prefix state. |
| token attribution/decomposition | assigns hidden-state contribution to input tokens | C45 changes the forward representation and ranking function; it is not a post-hoc decomposition of an unchanged Transformer. |

Primary sources checked before implementation:

- Feedback Transformer: https://openreview.net/forum?id=OCm0rwa1lx1
- Recurrent Memory Transformer: https://arxiv.org/abs/2207.06881
- Parallelizing Linear Transformers with the Delta Rule:
  https://arxiv.org/abs/2406.06484
- Gated Delta Networks: https://arxiv.org/abs/2412.06464
- BERT4Rec: https://arxiv.org/abs/1904.06690
- CauseRec: https://arxiv.org/abs/2109.05261
- Token-wise decomposition of autoregressive LM hidden states:
  https://aclanthology.org/2023.acl-long.562/

Verdict: the full composition is not shown to be globally novel. It is eligible
for a minimal falsifier only because the proposed contribution is tied to one
specific representation restriction and must beat its closest reductions.
