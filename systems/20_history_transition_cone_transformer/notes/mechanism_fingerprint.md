# C20 mechanism fingerprint

## Identity

- primitive: request-local nonnegative reconstruction of query-to-candidate
  hidden displacements from chronological history-transition vectors;
- intervention: candidate-token residual between lower and upper Transformer
  blocks;
- state: relation displacement `r_i`, transition dictionary `D`, nonnegative
  coefficients `alpha_i`, supported vector `p_i`;
- solver: fixed-depth projected gradient, initialized at zero, with a
  request-derived safe step size and no outcome-dependent stopping;
- inputs: query, ordered history, candidates, masks and exact-identity relation;
- no-history: exact structural no-op;
- complexity: `O(K C H d)` for fixed unroll depth `K`.

## Algebraic witnesses

1. For one nonzero history transition `d`, the cone contains `d` but not `-d`;
   an unconstrained span contains both.  This is the primary sign witness.
2. If transition columns are nonorthogonal, the NNLS normal equations contain
   off-diagonal Gram terms.  Elementwise ReLU retrieval cannot in general
   produce the same coefficient vector.
3. Softmax/simplex retrieval constrains coefficients to sum to one.  HTCT
   permits any nonnegative magnitude and therefore can represent a positive
   multi-step composition outside the convex hull but inside the cone.
4. Adding one common translation to every query, candidate and history state
   changes no displacement and therefore no conic write.
5. Candidate permutation only permutes independent NNLS problems; request-mean
   centring commutes with the same permutation.
6. With no valid transition, `D` is empty, `p=0`, and the final `torch.where`
   returns the exact base tensor.

## Degenerations and controls

| mode | coefficient law | question |
|---|---|---|
| `cone` | repeated projected gradient, `alpha>=0` | proposed positive-composition law |
| `span` | same iterations without projection | does directionality matter? |
| `relu1` | one projected step only | is the solver just ReLU attention? |
| `simplex` | softmax coefficients summing to one | does ordinary retrieval suffice? |
| `pooled_mlp` | matched transition-pool residual | does generic late capacity suffice? |

Every mode instantiates the same named parameters and begins from the same
state.  A positive claim requires the cone mode to beat every control; solver
complexity alone is not a contribution.
