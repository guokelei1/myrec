# C36 mechanism fingerprint

- **Operator:** active-set candidate barycenter subtraction followed by one
  request-wise max-norm trust coefficient.
- **Intervention locus:** the value/residual write of authenticated
  history-to-candidate cross-attention, before candidate scoring.
- **State:** one shared query-tangent history displacement plus one
  candidate-indexed, zero-mean displacement field.
- **Trainable parameters:** a shared rank-16 residual LM adapter (`16,384`
  parameters); all conservation and trust operations are parameter-free.
- **Training signal:** complete-list listwise loss plus clicked-vs-unclicked
  correction direction loss, identical for all modes.
- **Inference input:** query, strictly prior authenticated item-history states,
  and the current candidate set; no labels, categories, user ID, or dataset ID.
- **Complexity:** `O(n_candidates * n_history * d)` plus candidate-axis
  reductions; no candidate-specific parameters.
- **Exact reductions:** `delta=0` gives global transport; `lambda=1` gives
  unbounded barycentric transport; omitting active-set centering gives
  uncentered trust transport; omitting `g` gives C35 relative-only transport.
