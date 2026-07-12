# C57 nearest-neighbor and reduction audit

Status: pre-outcome; global novelty **unestablished**.

| Neighbor | Shared mechanism | Binding boundary |
|---|---|---|
| Slot Attention ([paper](https://arxiv.org/abs/2006.15055)) | Inputs normalize across exchangeable slots so slots compete for evidence | Candidate-axis normalization is established prior art. `slot_budget_no_null` is the direct reduction. C57 cannot claim novelty from changing the softmax axis alone. |
| Slot normalization analysis ([paper](https://arxiv.org/abs/2407.04170)) | Shows assignment/aggregation normalization affects varying slot counts | C57 fixes one cardinality normalization before outcomes and does not sweep it. A tie to the no-NULL reduction removes architectural rent. |
| DIN / ZAM / TEM | Query/candidate-conditioned history attention, sometimes with a zero vector | These normalize over history independently for each target. `history_softmax` is the direct axis degeneration with identical encoders/parameters. |
| Set Transformer ([paper](https://openreview.net/forum?id=Hkgnii09Ym)) / PRM ([paper](https://arxiv.org/abs/1904.06813)) | Permutation-equivariant set/list interaction | Generic set attention can carry raw candidate values; `raw_candidate` and the history-only value contract test this shortcut. |
| C24/C53/C54/C56 | Candidate list competition and/or history-only V streams | They form per-candidate carriers before list competition. C57 makes candidates compete while the event-to-candidate assignment is formed. Same-checkpoint axis ablation tests the distinction. |
| Optimal-transport/Sinkhorn slot variants | Doubly normalized assignment matrices | C57 uses one softmax and a NULL sink, not iterative Sinkhorn. No optimal-transport novelty is claimed; adding Sinkhorn is outside this gate. |

The exact softmax-axis idea is therefore not new.  The gate asks a narrower
architectural question: whether treating the *actual ranked candidates* as
data-dependent slots, retaining an explicit NULL evidence budget, and reading
only strong-base-centered history assignments solves PPS's candidate-common
write failure.  This can be a useful architecture result only if it beats the
registered Slot/DIN/pooled/raw reductions and later transfers unchanged across
domains.
