# C45 mechanism fingerprint

- **Information object:** a prefix-conditioned per-event state intervention,
  `F(m,h)-F(m,NULL)`, not a raw item vector, pooled profile, score delta, or
  candidate-event edge ledger.
- **Intervention locus:** history representation formation inside a shared
  recurrent Transformer transition.
- **Propagated state:** factual prefix state only; the local NULL state is
  discarded after forming the event token.
- **Ranking path:** a Transformer read token jointly formed from current query
  and candidate cross-attends to the innovation-token sequence and emits the
  end-to-end candidate correction.
- **Training signal:** identical listwise ranking plus query/candidate base loss
  for all modes; no certificate, oracle, or architecture-specific auxiliary
  target.
- **Inference inputs:** query, strictly-prior ordered history, candidates, and
  evidence masks only.
- **Degenerations:** ordinary adjacent state delta, factual state, and raw event
  token, all with identical trainable parameters and transition executions.
- **Complexity:** `O(H d^2 + C H d)` for bounded history length; two shared
  transition evaluations per event and no online LLM call.
- **Novelty status before outcome:** `uncertain composition`. Recurrent memory,
  NULL pairing, and token-level contributions are established separately; only
  their event-representation role and required matched degeneration are under
  test.
