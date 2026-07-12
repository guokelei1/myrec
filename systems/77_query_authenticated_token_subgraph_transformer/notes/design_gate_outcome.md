# C77 data-free design-gate outcome

Decision: `close_c77_before_repository_data`.

All modes and seeds passed G0, used 219,648 trainable parameters, reduced loss,
kept the frozen-anchor hash unchanged, and preserved no-history, query-mask,
repeat, determinism, and candidate permutation exactly.  The primary admitted
about 14% of tokens; unsupported nuisance-token gradient was exactly zero and
both C-H and H-C edges were active.

The primary solved the held-out shortcut reversal in all three seeds:

- clean supported accuracy `1.0000`;
- wrong-history supported accuracy `0.0000` with negative clean-margin
  retention near `-1.0`;
- query-mask behavior returned to the tied base (`0.1091`) with zero margin;
- repeat and no-history accuracy `1.0000`.

Two independent gates nevertheless failed.  First, event permutation retained
only `0.351/0.362/0.392` of clean margin (supported accuracy
`0.752/0.744/0.877`), below the frozen 0.80 margin-retention contract.  This
creates unsupported chronology dependence even though the motivating Amazon
token HSO found no load-bearing order effect.

Second and more important, the full query-authenticated C-H triangle paid no
rent over simpler token admission.  `query_candidate_filter` tied primary
supported accuracy exactly in all three seeds.  Primary minus frozen pairwise
C-H filtering was only `+0.0063/+0.0636/+0.0063`, below the all-seed `+0.02`
rule in two seeds.  It did beat `ungated_full` and `query_history_filter` by
`+1.0`, confirming that excluding candidate-only nuisance tokens—not the
shared Q-C-H triangle—caused the success.

C77 is closed.  Removing positional embeddings, weakening shuffle, or
promoting the tied query-candidate filter as C77 would be a post-outcome
rescue.  No repository data, label, dev, test, qrel, or evaluator was opened.
The durable lesson is narrower: a frozen query-side candidate-token admission
boundary can block synthetic label shortcuts, but current evidence does not
make the triadic subgraph a unique architecture contribution.

Authoritative report: `reports/pps_c77_design_gate.json`.
