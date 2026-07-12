# C74 locked data-free design gate

C74 reuses the byte-locked C73 associative generator but selects a new train
and validation seed (`20265100/20265101`) not used for C73 outcome or the
post-terminal diagnostic.  Reuse keeps the information problem fixed while
the architecture changes; both the external generator and C73 proposal lock
are included in C74's lock.

All four equal-parameter modes train for 400 steps with identical examples,
batch order, initialization seed, optimizer, and candidate sets:

- `semantic_conservative_relay`;
- `coupled_value_relay`;
- `pooled_semantic_relay`;
- `factual_semantic_relay`.

The primary must reach the frozen absolute and base-gain thresholds and beat
all three reductions in every seed.  Wrong history uses deterministic
cross-example donors; shuffle reverses events while position coordinates stay
fixed; coarse corruption removes preference values; query mask disables the
personalized path.  Repeat/no-history identities, finite loss/gradients,
matched parameter counts, deterministic rescoring, and candidate permutation
are binding.

Any failed condition closes C74.  No threshold, rank, temperature, scale,
position rule, steps, seed, or generator change is permitted after outcome.
