# Nearest neighbours, novelty, and complexity

## Direct neighbours

| neighbour | already-covered mechanism | C14 verdict |
|---|---|---|
| softmax + zero NULL | unit simplex with unused mass assigned to a zero value | exact forward/Jacobian reparameterization |
| ordinary attention × scalar gate | conditional event allocation multiplied by candidate/head support | exact equation `o=rho Attention_p(V)` |
| [sigmoid self-attention](https://arxiv.org/abs/2409.04431) | non-sum-to-one independent attention weights; normalization/LayerScale stability practices | stronger absolute-support alternative already systematized |
| [Gated Attention for LLMs](https://papers.nips.cc/paper_files/paper/2025/hash/904e89bb4e632e75fb47f093b620b257-Abstract-Conference.html) | input-dependent head-specific sigmoid gate after SDPA | directly covers per-head output suppression before/around `W_O` |
| [SigGate-GT](https://arxiv.org/abs/2604.17324) | learned per-head input-dependent sigmoid gates drive attention output near zero | direct cross-domain output-gate neighbour |
| [Multiscreen](https://arxiv.org/abs/2604.01178) | absolute query-key relevance, independent thresholds, rejection without sum-to-one competition | stronger real-event screening mechanism |
| [sparsemax](https://proceedings.mlr.press/v48/martins16.html) / [entmax](https://aclanthology.org/P19-1146/) | differentiable attention with exact zero weights | covers exact event/null sparsity if C14 leaves dense softmax |
| DIN target attention | candidate-conditioned one-way history weighting | C14 is target attention plus null/scalar gate |
| ZAM | query attention includes a zero vector that can suppress personalization | C14 generalizes the same zero-value normalization across candidates/heads |
| TEM | Transformer history/query personalization | adding a renamed support scalar does not create new information flow |
| C03 dustbin | partial mass can leave real events for a learned dustbin | one-way C14 is the degeneration C03's own audit warned could become null target attention |

Ramapuram et al. also identify stabilization of large initial sigmoid-attention
norms as critical; LayerScale/initial normalization therefore cannot carry a
new C14 claim.  Qiu et al. compare many gate positions and find head-specific
post-SDPA sigmoid gating effective.  Nakanishi's screening directly targets the
same “absolute rather than relative relevance” motivation without restoring a
unit subprobability through NULL.

## Matched controls

Any future probe would require all of these from the same Transformer states:

1. exact transformed softmax+zero-NULL using the same support/allocation heads;
2. ordinary softmax attention plus candidate/head sigmoid output gate;
3. independent sigmoid attention with registered norm stabilization;
4. sparsemax/entmax over real+NULL events;
5. ZAM-style one-way zero-vector target attention;
6. C03-style one-way dustbin degeneration without triadic transport;
7. zero, small-nonzero, and standard LayerScale initializations.

The first control is mathematically identical, so statistical superiority is
impossible absent implementation/optimization asymmetry.

## Parameters and FLOPs

If `rho` is derived from ordinary real/NULL logits, factorization adds no
parameters and only `O(H)` reductions per candidate/head.  Explicitly appending
NULL adds one key/value position and negligible `O(C d)` work.

An independent support projection adds roughly one scalar per candidate/head
from the query (`O(C d heads)` and corresponding parameters), exactly matching
head-specific gated-attention controls.  Independent per-event screening costs
the same order as attention logits and moves C14 toward sigmoid/Multiscreen.
Candidate centring and the global bound add `O(Cd)`.

LayerScale contributes `d` or `heads` parameters depending on granularity and is
matched verbatim in every control.  Extra capacity cannot establish novelty.

## Absolute-mass interpretation

The attention mass is not identifiable as effective support while `W_V`, `W_O`,
and LayerScale are free to rescale.  A model can decrease `rho` and increase
value/output norms.  Reporting mass without effective write magnitude is
insufficient; constraining these norms would add another mechanism and still
would not break the null/gate equivalence.
