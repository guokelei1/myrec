# C69 proposal — semantic-null behavioral relation signal

Status: pre-outcome signal prerequisite. C69 cannot be claimed as the final
architecture.

## Corrected observation

C42/C43 post-terminal function analysis shows that KuaiSearch metric transport
already changes substantially under wrong history, but those changes are not
relevance-aligned. C46 learned real sequence structure relative to shuffled
pairing yet tied frozen semantic mean. Random cross-user negatives therefore
let a behavioral model relearn ordinary item-text similarity without learning
the residual relation needed by personalized ranking.

Direct item-ID transition modeling is not a cross-domain solution: on the
opened C47-A cohorts, only 3.62% of Kuai candidate rows and 1.53% of Amazon
candidate rows occur in the corresponding 6,000-fit history vocabulary.

## Signal primitive

One shared two-item Transformer receives `[REL, SOURCE_ITEM, TARGET_ITEM]` LM
states and returns `f(h,c)`. The scored relation is its anchored two-way
interaction

```text
r(h,c) = f(h,c) - f(0,c) - f(h,0) + f(0,0).
```

Thus source-only, target-only, bias, and token-role shortcuts cancel exactly.
Positive pairs are adjacent events in one user's fit history. For each positive
`(h_i,c_i)`, the primary chooses a target `c_j` from another fit history that
jointly matches:

- `cos(c_i,c_j)`, and
- `cos(h_i,c_j)` to the positive `cos(h_i,c_i)`.

The nearest match inside the fixed training batch is deterministic. A matched
model trained with only a cyclic random cross-history negative is the binding
control. Both own identical parameters, initialization, batches, optimizer,
steps, and source histories.

At outcome scoring, a fixed query-to-history semantic softmax selects events;
only the anchored behavioral relation supplies candidate value:

```text
a_j = softmax(cos(q,h_j) / 0.1)
d_c = sum_j a_j r(h_j,c).
```

Ordinary semantic attention uses the identical `a_j` but value
`cos(h_j,c)`. Missing history returns a zero relation. This scorer is a signal
instrument; a later proposed system must internalize query selection and
behavioral relation in one end-to-end LM/Transformer path.

## Falsification

C69 advances only if the semantic-matched relation, on both KuaiSearch and
Amazon-C4:

1. beats ordinary semantic attention with a positive paired interval and all
   fixed-fold signs;
2. beats the equal-parameter random-negative relation;
3. true history beats matched wrong history;
4. clicked relation direction is positive;
5. all three seeds agree and all algebra, gradients, determinism, candidate
   permutation, source-zero, and no-history contracts pass.

A one-domain pass is failure. A pass authorizes a separate catalog-open
Transformer architecture formulation on fresh roles, not automatic training.
Failure closes semantic-hard-negative relation learning as the missing signal;
no negative-cost, temperature, width, step, aggregation, seed, or cohort rescue
is allowed.
