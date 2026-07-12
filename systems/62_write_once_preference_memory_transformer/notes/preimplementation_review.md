# C62 pre-implementation review

Decision: **conditionally authorize only the synthetic G0 implementation**.
Real GPU ranking training is authorized only if G0 passes every binding check.

## Design-gate findings

- **Problem trace:** pass.  The primitive addresses instability caused by
  query/candidate-conditioned history selection, not a post-hoc metric slice.
- **LLM4Rec/Transformer family:** pass.  History formation, latent writing,
  candidate reading, list interaction, and score generation are Transformer
  computations in the end-to-end ranker.
- **Single primitive:** pass.  The claim is immutable history-only write followed
  by query-candidate read.  Multi-slot state is part of that bottleneck, not an
  independent router.
- **No dataset branch:** pass.  One config and decision rule cover KuaiSearch and
  Amazon-C4; input projection width is the only mechanical domain difference.
- **Nearest-neighbor audit:** conditional pass.  Latent slots themselves are not
  novel.  Direct attention, query-conditioned writing, and one-slot pooling are
  binding controls; failure against any closes the architecture claim.
- **Strong-base protection:** conditional pass.  Empty history and repeat are
  exact structural fallbacks.  Non-repeat residuals are centered but not
  manually clipped, so fit-holdout and fresh A0 must detect both rank-inertness
  and base overwrite.
- **Outcome isolation:** pass.  The real formulation gate uses exposed fit only;
  C26 internal-A and C39 reserve labels remain closed.

## Synthetic G0 requirements

Three fixed seeds must satisfy all of the following before real training:

1. primary memory is exactly invariant to query and candidate substitution;
2. no-history and repeat fallbacks are exact;
3. candidate permutation and repeated scoring are within frozen tolerance;
4. wrong history changes non-repeat rankings;
5. gradients reach history encoder, slot writer, memory reader, candidate-set
   Transformer, and score head;
6. primary solves a planted two-interest task and exceeds its single-slot
   ablation by the frozen margin;
7. all modes instantiate exactly the same parameter count.

Passing G0 establishes only that the primitive is mechanically learnable.  It
does not authorize fresh labels or a paper claim.
