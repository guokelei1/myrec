# C44 frozen data-free design gate

Status: pre-outcome; immutable after design lock.

## D0 structural contract

- all four modes have identical tensors, parameter counts, and initialization;
- both factors receive finite nonzero gradients;
- outputs are finite, deterministic, and candidate-permutation equivariant;
- no-history, absent-query, and exact-repeat corrections are exactly zero;
- primary candidate corrections sum to zero per head and after aggregation;
- primary candidate mass plus null mass equals one for every event;
- controls remove exactly their registered information-flow edge.

## D1 planted mechanism task

Across all three seeds, primary must:

1. reach clean NDCG@10 at least `0.80` and improve over base by `0.25`;
2. exceed each matched control by at least `0.02`;
3. exceed its wrong-history NDCG by at least `0.25`;
4. assign at least `0.50` mass to the planted candidate on signal events;
5. assign at least `0.50` null mass to irrelevant and wrong events;
6. keep correction sum error at most `1e-6`.

Failure is terminal. No teacher, noise amplitude, candidate/history count,
temperature, seed, epoch, threshold, or loss rescue is allowed. Passing only
authorizes a separately frozen real-data formulation.
