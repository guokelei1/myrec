# C07 G1 Semantic Synthetic Result

Date: 2026-07-11
Decision: **FAIL — stop C07 without real-data, GPU, dev, or test work**
First failed rule: `rule_1_R_preservation`

## Provenance and integrity

- Old normative lock: `66308db14f00e20de860a2060d147329fb93fa07b806951c227a5499746c2edd`
- G1 execution lock: `65e8dbe35f56df3347e4df5c5e870efcc040aa886093f2edad7911bf1fe3f3e3`
- Raw ignored result: [result.json](/data/gkl/myrec/artifacts/runs/20260711_c07_pdsk_g1_cpu/result.json)
- Raw SHA-256: `363400ec174f6b4f6c49f5b2bdcbc1d9cd479b36ca4d531ef92890845cfca16d`
- Raw size: 60,493 bytes
- Fixed execution: three seeds, seven equalized 5,840-parameter methods,
  512 updates each, CPU FP32 training/FP64 algebra audits, one thread.
- Repository data, labels/qrels, shared dev/test evaluator, and GPU were not
  read or used.

Both lock layers remained valid after the run.  An independent arithmetic audit
recomputed the primary comparisons and first-failure decision from raw JSON.

## Frozen-rule outcome

| Rule | Result |
|---|---|
| R exact-recurrence preservation | **FAIL** |
| S supported non-repeat action and novelty | **FAIL** |
| U corruption specificity | **FAIL** |
| N exact no-history fallback | PASS |
| Optimization viability | PASS |
| Post-training algebra | **FAIL** |

The first failure is binding; later results are diagnostics only.

## R — exact recurrence

| Seed | PDSK top-1 | ITEM_ONLY top-1 | PDSK margin minus ITEM_ONLY |
|---:|---:|---:|---:|
| 20260711 | 0.992920 | 0.994141 | +0.018148 |
| 20260712 | 0.965576 | 0.987305 | -0.362081 |
| 20260713 | 0.992432 | 0.992432 | +0.050044 |

The frozen rule required PDSK top-1 strictly above 0.99 and margin no more than
0.01 below ITEM_ONLY in every seed.  Seed 20260712 fails both conditions.

## S — supported non-repeat

| Method | 20260711 | 20260712 | 20260713 | Mean top-1 |
|---|---:|---:|---:|---:|
| PDSK | 0.542969 | 0.711914 | 0.708252 | 0.654378 |
| CENTER0 | 0.705811 | 0.801270 | 0.725586 | 0.744222 |
| GATED_CENTER | 0.507568 | 0.492188 | 0.588379 | 0.529378 |
| TARGET_NULL | 0.780518 | 0.743408 | 0.725098 | **0.749674** |
| DIFF_ATTN | 0.547119 | 0.717529 | 0.705566 | 0.656738 |

PDSK is below the best control, TARGET_NULL, by `-0.095296` mean top-1 and is
not above 0.75 in any seed.  Thus the pairwise dead-zone kernel does not pay its
preregistered novelty rent.  CENTER0 also exceeds PDSK by `+0.089844` mean,
which is direct negative evidence against the thresholded pairwise primitive in
this synthetic contract.

## U — unsupported/corrupted evidence

PDSK history-induced flip rate / mean absolute logit change:

| Corruption | 20260711 | 20260712 | 20260713 | Frozen upper bound |
|---|---:|---:|---:|---:|
| wrong history, flip | 0.068848 | 0.052734 | 0.069580 | <0.01 |
| wrong history, MAE | 0.067519 | 0.036770 | 0.036780 | <0.01 |
| shuffled event, flip | 0.034668 | 0.068848 | 0.148438 | <0.01 |
| shuffled event, MAE | 0.036596 | 0.054123 | 0.085087 | <0.01 |
| query masked, flip | 0.048828 | 0.012451 | 0.021729 | <0.01 |
| query masked, MAE | 0.050134 | 0.009666 | 0.012830 | <0.01 |

The specificity rule fails broadly.  The required S-minus-U accuracy gap also
fails for wrong-history seed 20260711 and for shuffled-event in all seeds.

## Exact and structural diagnostics

- N no-history maximum PDSK logit mismatch: exactly `0` in all seeds; rank and
  score-order mismatches: exactly `0`.
- S active-pair fractions: `0.056534 / 0.070386 / 0.069601` — inside the frozen
  open interval.
- Nonzero evidence-gradient fractions: `0.166016 / 0.202734 / 0.194531` —
  inside the frozen open interval.
- Candidate-conservation maxima: `2.78e-17` to `3.47e-17`; common-mode maxima:
  `5.55e-17` to `9.71e-17` — both pass.
- Full-model permutation errors:
  `4.41e-6 / 1.61e-6 / 2.62e-6`, all above the frozen strict `1e-6` threshold;
  therefore the combined post-training algebra rule fails despite the exact
  kernel-level invariants.

## Decision

C07 fails its first preregistered semantic gate and is closed.  The result does
not support a real-data probe, train/internal escalation, GPU allocation, dev
screening, or test access.  No rerun, threshold adjustment, smoothing change,
or generator/control revision is permitted under this lock.
