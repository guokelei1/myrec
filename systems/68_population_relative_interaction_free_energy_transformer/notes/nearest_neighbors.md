# C68 nearest-neighbor and reduction boundary

The novelty verdict before outcome is `distinct-with-high-uncertainty`.

| Neighbor | Shared mechanism | Binding distinction / reduction |
|---|---|---|
| ordinary target/query-aware attention | candidate or query selects history events | no population reference and no exposed conditional partition ratio; `user_only_free_energy` is the reduction |
| RESUS | separates global and residual user preference | RESUS predicts a residual from pooled nearest-neighbor/ridge features; C68 differences event-level conditional log partitions before ranking; `mean_interaction` binds the first-moment reduction |
| MrTransformer preference editing | separates common and unique preferences between user sequences | preference editing is a training operation over latent preference tokens; C68 uses an explicit population reference in every inference-time interaction energy |
| C04/C65 paired factual/NULL paths | shared-model conditional-minus-null effect | a one-token reference is an exact C68 degeneration; `single_null_interaction` must lose |
| C25 anchored Möbius interaction | removes lower-order query/history/candidate terms | C25 is eventwise anchored finite differencing; C68 additionally contrasts two empirical event distributions through a log partition.  A later real gate must run C25's eventwise reduction |
| C03 partial transport / C44 candidate-axis flow | normalized evidence and abstention | C68 solves no transport plan, conserves no mass, and exposes a free-energy interaction rather than moving values along a plan |
| C47 KRR / C51 covariance | compares a query/candidate with a history distribution | high-temperature C68 approaches a mean interaction, but finite-temperature log partition contains higher cumulants and has no inverse solve or second-moment-only form |
| attention energy / modern Hopfield retrieval | log-sum-exp is the attention energy | log-sum-exp itself is known; C68's only possible contribution is the four-way population-relative conditional interaction and must pay rent over ordinary attention |

Exact reductions:

1. `R={NULL}` gives the single-reference counterfactual control.
2. As `tau -> infinity`, after removing common constants, C68 converges to the
   difference of mean event energies (`mean_interaction`).
3. If `H=R`, all four partition terms cancel exactly.
4. If `F=a(q,c)+b(q,e)`, the four-way difference is exactly zero.
5. A universal DeepSets/Transformer can approximate the same set function;
   C68 claims an inductive bias, not an expressivity separation.

Primary sources reviewed:

- RESUS: https://arxiv.org/abs/2210.16080
- MrTransformer preference editing: https://arxiv.org/abs/2106.12120
- Modern Hopfield/attention energy: https://arxiv.org/abs/2008.02217
- Set Transformer: https://arxiv.org/abs/1810.00825
- CAGrad/gradient and TTT families are excluded by the post-C67 audit rather
  than treated as C68 neighbors.

The primitive is not paper-ready because each ingredient is known.  If any
first-moment, single-NULL, user-only, pooled, or ordinary-attention control
matches it, the distinction is empirically empty and C68 closes.
