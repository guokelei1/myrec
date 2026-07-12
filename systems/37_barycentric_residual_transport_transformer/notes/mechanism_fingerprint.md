# C37 mechanism fingerprint

- **Operator:** active-set candidate-axis barycenter subtraction.
- **Locus:** authenticated history-to-candidate value/residual write before
  candidate scoring.
- **State:** one shared query-tangent displacement plus a zero-mean
  candidate-indexed displacement field.
- **Parameters:** one shared rank-16 residual LM adapter (`16,384`); the
  conservation law is parameter-free.
- **Training:** complete-list listwise and correction-direction losses shared
  exactly by all modes.
- **Inference:** query, strictly prior authenticated history, candidate set;
  no labels, user ID, category, query type, or dataset ID.
- **Complexity:** `O(n_candidates * n_history * d)`.
- **Reductions:** delete `delta` for global-only; omit active-set centering for
  uncentered additive; delete `g` for relative-only.
