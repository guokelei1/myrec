# C07 G1 Execution Blocker

Date: 2026-07-11
Decision: **STOP before runner implementation and before any semantic outcome**

## Integrity check

`PRE_OUTCOME_LOCK.json` was verified before this audit.  All seven normative
file hashes matched and the combined manifest was:

```text
66308db14f00e20de860a2060d147329fb93fa07b806951c227a5499746c2edd
```

The lock states `semantic_probe_run=false`, `repository_data_read=false`,
`labels_or_qrels_read=false`, and `gpu_used=false`.  This audit did not change
any locked normative file.

## Why execution must stop

The frozen document fixes dimensions, seeds, optimizer, budget, broad world
semantics, method names, metrics, and pass thresholds.  It does not fix several
choices that directly determine relative method performance.

### Outcome-determinative generator gaps

1. No distributions or scales are specified for query, candidate, history,
   base-signal, common-shift, or distractor tensors, nor whether vectors are
   normalized.
2. World R does not specify the exact-match logit/support scale, distractor
   distribution inside the dead zone, or whether the base features reveal the
   recurrent target.
3. World S does not specify how query selection is encoded; the sizes of the
   two supported margins; history value directions; sub-threshold noise count
   and distribution; or the contradictory pair's candidates, sign, and
   magnitude.  These values can trivially favor PDSK, centered attention, or
   target attention.
4. World U does not define the precise permutation axes/derangements, whether
   each corruption is used in training or only held out, or whether the fixed
   batch's 16 U requests are divided among three corruptions or duplicated for
   each.  `query_masked` also lacks a tensor-level definition.
5. World N says only “large canary values”; magnitude, dtype-safe bounds, and
   candidate/base construction are unspecified.

### Outcome-determinative control gaps

1. `GATED_CENTER` has no equation or architecture for the evidence summary,
   amplitude, temperature, exact dead-zone behavior, normalization axes, or
   value aggregation.
2. `TARGET_NULL` does not define its null key/logit/value, history-normalization
   axis, or how its output enters the candidate state.
3. `DIFF_ATTN` does not define its two Q/K maps, subtraction coefficient,
   initialization/constraint, normalization axis, or value path.
4. `BASE_FFN` does not define the active parameter-matching FFN or where it is
   inserted.  Consequently “same total parameter count” cannot be enforced.
5. `ITEM_ONLY` does not define residual magnitude, whether it is learned, or
   how query compatibility affects it.
6. The frozen prototype has one signed kernel after a shared Transformer, while
   the gate fixes four heads/two layers but does not say which heads/layers are
   replaced for any method.

### Outcome-determinative measurement gaps

1. “Pairwise target margin” could mean target minus maximum, mean, or every
   non-target logit.
2. “History-induced logit change” does not identify whether the reference is
   the same method with history masked or the separately trained `BASE_FFN`.
3. Active-pair and nonzero-gradient fractions lack denominators, numerical
   nonzero tolerance, measurement step(s), and aggregation rules.
4. Top-1/rank tie-breaking and score-order mismatch are undefined.
5. The number and construction of post-training common-mode/permutation audit
   samples are unspecified.

## Consequence

Filling these gaps now would create a new protocol after the lock.  In
particular, supported-margin magnitude and the implementation capacity of
`GATED_CENTER` can each reverse the primary pass rule.  Therefore no compliant
runner can be implemented from the frozen text alone.

No smoke run, training run, raw outcome artifact, or result metric was created.
There is no G1 pass/fail result; status is **blocked by pre-outcome protocol
underspecification**, not mechanism failure.

## Required next action

Before execution, a coordinator must freeze a revised protocol that gives:

- executable tensor-level generator equations and numeric distributions for
  R/S/all three U corruptions/N;
- exact equations and insertion points for every control plus a mechanical
  parameter-count assertion;
- the U training-batch composition;
- metric, tie, gradient-coverage, and audit-sample definitions;
- a new normative hash lock created before any smoke or outcome.

The current lock must remain preserved as the audit record; it must not be
silently edited and reused.
