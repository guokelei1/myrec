# C18 locked synthetic-gate protocol

Status: **thresholds frozen before implementation outcome**.  Exact executable
constants must match `configs/synthetic_gate.yaml` and the later
`proposal_lock.json`.  Any learned-run failure closes C18; no threshold or data
generator may be revised under this candidate ID.

## G0 — structural/unit gate

CPU tests must prove:

1. protected margins are satisfied to `1e-6` in float64;
2. projection is idempotent to `1e-6` and preserves candidate mean;
3. candidate permutations commute with projection to `1e-6`;
4. an active constraint gives equal/opposite corrections and removes common
   mode;
5. empty history returns bitwise base scores for every trainable mode;
6. projection, direct and soft-penalty modes have identical parameters and
   byte-identical initialization;
7. finite nonzero gradients cross the Transformer and projection;
8. the runner has no repository-data, evaluator, qrels, or test path.

G0 failure permits implementation repair before the proposal lock because it
has no learned outcome.  The final passing source is then hash-locked.

## G1 — one-shot learned synthetic GPU gate

- physical GPU: `2`, exposed as `cuda:0`;
- environment: `/data/gkl/conda_envs/myrec-c18`;
- seeds: `20260718`, `20260719`, `20260720`;
- per seed: 8,192 clean train requests and 2,048 independently generated eval
  requests, balanced across no-history, exact-repeat conflict, and supported
  non-repeat strata as fixed in config;
- eight candidates, eight history slots, 16 raw dimensions;
- `projection`, `direct`, and `soft_penalty` each use the same 800 optimizer
  steps, batches, AdamW settings, parameter count and initial state;
- no early stopping, sweep, retry, repository record, label, dev/test data or
  evaluator call;
- all corruptions are evaluation-only: wrong history, event shuffle, query
  mask and coarse/style removal.

Every condition must pass in all three seeds:

1. projection exact-repeat top-1 accuracy is at least `0.98` and no more than
   `0.01` below `anchor_only`;
2. supported non-repeat accuracy is at least `0.75`, exceeds `base_only` by at
   least `0.20`, and is no more than `0.01` below the best trainable control;
3. `min(repeat_accuracy, supported_accuracy)` exceeds the best of `direct` and
   `soft_penalty` by at least `0.03`;
4. projection beats `direct` on repeat accuracy by at least `0.05` without
   losing more than `0.01` supported accuracy;
5. the clean supported target-margin improvement over base is positive; every
   corruption retains at most `0.35` of that improvement;
6. at least 10% of repeat requests activate the projection and at least 20% of
   all history-present requests have score-delta range at least `0.05`;
7. maximum protected-margin violation is at most `1e-5`;
8. every no-history score is bitwise base, candidate permutation error is at
   most `1e-5`, and all training/scores/gradients are finite;
9. trainable modes have identical parameter counts and initial-state hashes.

The gate intentionally requires utility beyond a structural guarantee.  A
projection that merely preserves repeat constraints but cannot learn useful,
corruption-specific non-repeat transfer must stop.

## G2 — possible train-internal real gate

G2 has zero current budget.  It may be designed only if G1 passes.  It must use
a new request-ID-hash cohort disjoint from exposed C02/C05/C06 cohorts, first
establish exact D2p/item-anchor parity, and freeze repeat/non-repeat,
wrong/shuffle/query-mask/coarse, no-history, load-bearing order-change,
throughput and matched-control thresholds before labels are opened.  A G2 pass
would still not authorize dev; dev requires a separate explicit lock.
