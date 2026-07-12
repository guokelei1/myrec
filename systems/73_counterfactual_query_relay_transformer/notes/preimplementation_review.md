# C73 preimplementation review

Decision: `authorize_data_free_design_gate_only`.

Positive findings:

- the information path is mathematically distinct from C45, C54, C65/66, and
  pooled C31/32;
- the Transformer is load-bearing and produces the ranking correction end to
  end;
- query/history/candidate interaction uses one graph and evidence masks, with
  no dataset-only field;
- three parameter-identical reductions bind both architectural and capacity
  attribution;
- no-history, repeat, wrong-history, shuffle, coarse-value, query-mask,
  determinism, and permutation contracts are preregistered.

Risks:

- attention bottlenecks and query-aware history refinement are known families;
  only the paired internal operator difference is provisionally distinct;
- C22 already showed that deeper structural safety need not pay utility rent;
  C73 must beat late and pooled reductions rather than claim value from path
  restrictions alone;
- a synthetic pass cannot validate real cross-item personalization.

No repository data, pretrained LM training, dev, test, or qrels access is
authorized until the locked data-free gate passes.
