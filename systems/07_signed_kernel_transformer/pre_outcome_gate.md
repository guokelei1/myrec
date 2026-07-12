# C07 Pre-outcome Gate: PDSK Synthetic Falsifier

Status: **frozen before semantic synthetic outcomes; not run**
Allowed compute: CPU only
Allowed inputs: candidate-local synthetic tensors only
Forbidden: repository data, standardized records, cohorts, labels/qrels,
shared evaluator, dev/test calls, GPU, model downloads

Unit tests are algebra/instrument checks, not outcomes for this gate.  Passing
them authorizes this one synthetic probe; it does not count as G1 passing.

## G0 — structural admission (already checkable)

All checks use float64 where an exact algebraic tolerance is stated.

| Contract | Pass rule |
|---|---|
| Hand arithmetic | for scores `(2, 0, -1)`, `tau=.5`, `kappa=1`, balances equal `(2, -.5, -1.5)` and weights equal `(.4, -.1, -.3)` within `1e-12` |
| Candidate conservation | maximum `abs(sum_candidate weight)` <= `1e-10` |
| Common-mode | adding any event-wise scalar to all candidate logits changes weights by <= `1e-10` |
| Open-set abstention | all ranges <= `tau` implies bitwise-zero balances and weights |
| No history | logits are bitwise equal to base logits; delta and weights are bitwise zero |
| Information barrier | altering masked history values/identity bits changes outputs by <= `1e-6`; forward API contains no labels/qrels/dataset field |
| Permutation | candidate equivariance and history tuple invariance within `1e-6` end to end |
| Gradients | float64 gradcheck passes away from kinks; active query/history/candidate gradients are finite and nonzero |
| Linear degeneration | `tau=0` equals `C/(C-1) * centered(scores)` within `1e-12` |
| Non-factorization witness | active three-candidate balance is not collinear with centered logits |

Any failure blocks G1 and requires a pre-outcome code/protocol revision.

## G1 — one semantic synthetic probe

### Fixed generator

Use exactly seeds `20260711`, `20260712`, and `20260713`.  Each seed creates
independent train/internal and held-out sets for four worlds.  Fixed dimensions:

```text
candidates C = 5
history events H = 8
latent/model width d = 16
train requests per world per seed = 4096
held-out requests per world per seed = 4096
tau = 0.50
kappa = 1.00
```

All latent vectors are generated from a local seeded CPU RNG.  Synthetic target
indices are created by the generator and never serialized as repository qrels.
No hyperparameter sweep is allowed.  One smoke run may use 32 requests before
the three fixed runs; it may inspect only finiteness and tensor shapes.

Fixed model/training budget for every trainable method:

```text
Transformer width = 16
heads = 4
layers = 2
FFN width = 32
dropout = 0
loss = listwise cross-entropy over the five synthetic candidates
optimizer = AdamW(lr=3e-4, betas=(0.9, 0.999), weight_decay=0.01)
batch size = 64 (16 requests from each world per update)
updates = 512 exactly
gradient clipping = global norm 1.0
schedule / warmup / early stopping = none
checkpoint selection = none; evaluate the final update
dtype = float32 for training, float64 for algebra audits
CPU threads = 1
```

Set PyTorch deterministic algorithms on.  Each method starts from the same
seed-indexed shared-Transformer tensors wherever shapes coincide; method-only
tensors use the same seed after shared initialization.  Training and held-out
sets are disjoint RNG streams.  No result may change the optimizer, update
count, threshold, null mass, generator, or control list.

### Worlds

1. **R — exact recurrence.** One candidate has one exact history identity;
   cross-item logits are zero-mean distractors constrained below the dead zone.
   The target is the recurrent candidate.  This checks that the mechanism does
   not suppress the reliable anchor.
2. **S — supported non-repeat.** There is no identity match.  The query selects
   two history events whose latent values agree on one candidate through
   above-threshold margins.  Other events contain (a) event-wise common shifts,
   (b) many sub-threshold pair differences, and (c) one isolated contradictory
   pair.  The target is determined before noise is sampled.  This is the only
   positive cross-item world.
3. **U — unsupported transfer.** Preserve all marginal norms and common shifts
   from S, but independently permute query coordinates, user/history bundles,
   and event/value pairing.  No target-changing history update is justified.
   Report each corruption separately as `wrong_history`, `shuffled_event`, and
   `query_masked`; do not pool them into an easier average.
4. **N — no history.** History mask is all false while the hidden history and
   identity tensors contain large canary values.  This is an exact fallback
   contract, not a statistical metric.

The generator must assert that R/S contain at least one active pair and U's
sub-threshold slice contains none before invoking the model.  It must not tune
the generator after results are observed.

### Fixed methods and equalization

All trainable methods use the same Transformer depth/width, token inputs,
optimizer steps, seed, and total parameter count (pad smaller controls with an
active candidate-local FFN, not unused parameters):

| ID | Method |
|---|---|
| `PDSK` | proposed pairwise soft-threshold signed kernel |
| `CENTER0` | exact `tau=0` linear-centered degeneration |
| `GATED_CENTER` | centered-softmax residual with a scalar dead-zone amplitude and a positive request-conditioned scalar temperature, both computed from the same evidence summary |
| `TARGET_NULL` | ordinary candidate-to-history target attention with one learned null value |
| `DIFF_ATTN` | difference of two softmax maps with matched head width |
| `BASE_FFN` | no history path plus parameter-matched Transformer/FFN capacity |
| `ITEM_ONLY` | direct exact-identity residual; evaluated only on R and N |

`GATED_CENTER` is the primary novelty control.  If it matches PDSK, the pairwise
normalizer has not paid rent even if both beat the base.

### Metrics

- top-1 accuracy and pairwise target margin on R and S;
- mean absolute history-induced logit change and top-1 flip rate on every U
  corruption;
- exact maximum logit/rank mismatch on N;
- active-pair fraction and nonzero-gradient fraction in every world;
- maximum conservation/common-mode/permutation error;
- mean and per-seed values only; never best seed.

### Pass rules (all required)

1. **R preservation:** PDSK top-1 >= `0.99`, and its target margin is no worse
   than `ITEM_ONLY` by more than `0.01` in every seed.
2. **S positive action:** PDSK top-1 >= `0.75` in every seed and exceeds the
   best of `CENTER0`, `GATED_CENTER`, `TARGET_NULL`, and `DIFF_ATTN` by at least
   `0.05` absolute in the three-seed mean.  Every per-seed difference against
   `GATED_CENTER` must be positive.
3. **Corruption specificity:** for each U corruption separately, history-induced
   top-1 flip rate relative to `BASE_FFN` <= `0.01`, and mean absolute logit
   change <= `0.01`.  S-minus-U top-1 accuracy gap must be >= `0.20` for every
   seed.
4. **N fallback:** maximum pointwise logit difference, score-order mismatch, and
   rank mismatch versus `BASE_FFN` are all exactly zero.
5. **Optimization viability:** active-pair fraction and nonzero signed-branch
   gradient fraction on S each lie in `[0.05, 0.80]`; no NaN/Inf occurs.
6. **Algebra after training:** conservation/common-mode errors <= `1e-8` and
   permutation error <= `1e-6` for every seed.

Threshold equality is failure.  No confidence interval is claimed for this
mechanism-admission probe; consistency across all three fixed seeds is the
preregistered stability requirement.

## Stop rules

C07 stops, without real-data or GPU escalation, if any of the following occurs:

- any required pass rule fails;
- `GATED_CENTER` is within 0.05 mean top-1 of PDSK on S;
- positive action appears only in R, indicating that `ITEM_ONLY` is sufficient;
- learned/probed effective threshold approaches the `tau=0` centered
  degeneration (no post-outcome smoothing or threshold sweep is allowed);
- corruption gains track S gains, showing unsupported transfer rather than
  evidence fidelity;
- the result depends on two-candidate cases, a category/query-type rule, or a
  fixed expert route;
- PDSK only wins through more parameters, steps, or a different input channel;
- active-gradient coverage violates its bound.

If the gate fails because the dead zone blocks learning, replacing it with a
smooth nonzero activation is a new mechanism and requires a new proposal,
nearest-neighbor audit, and pre-outcome lock.

## Decision boundary after G1

- **Pass:** authorize design of a new, separately locked train/internal smoke
  implementation.  Real records, dev, GPU, and full training still require the
  repository's coordinator and shared protocol.
- **Fail:** close C07 as either centered/generic-gate-reducible, exact-only, or
  optimization-inviable according to the first failed rule.
