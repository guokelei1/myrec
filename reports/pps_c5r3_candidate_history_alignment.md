# C5-R3 Candidate-History Alignment Motivation Gate

Status: **TERMINAL FAIL**.

This is the complete adjudication of the finite C5-R3 recovery ladder. The protocol and the sole fallback were frozen before component scores or outcomes were generated. Materialization and scoring did not read qrels; test was not read and no model was trained.

## Label-free Decomposition Audit

All **575,609** candidate rows were checked. The maximum absolute error against both the public scorer and the actual upstream B0b score file was **7.105e-15**; there were zero tolerance violations.

| Component | Nonzero requests | Nonzero candidates |
|---|---:|---:|
| Exact item | 3,442 | 9,577 |
| Category | 5,974 | 253,404 |

## History-present NDCG@10

| Seed | D2p | Item only | Category only | Full D2s |
|---:|---:|---:|---:|---:|
| 20260708 | 0.318614 | 0.350654 | 0.319208 | 0.345271 |
| 20260709 | 0.319141 | 0.351281 | 0.319673 | 0.346070 |
| 20260710 | 0.319550 | 0.352185 | 0.319522 | 0.345849 |

## Frozen Paired Comparisons

| Comparison | Seed | Delta | 95% CI | A significantly > B |
|---|---:|---:|---:|---|
| item only − D2p | 20260708 | +0.032040 | [+0.028302, +0.035715] | yes |
| item only − D2p | 20260709 | +0.032140 | [+0.028489, +0.035802] | yes |
| item only − D2p | 20260710 | +0.032635 | [+0.028942, +0.036269] | yes |
| category only − D2p | 20260708 | +0.000594 | [-0.002895, +0.004049] | no |
| category only − D2p | 20260709 | +0.000532 | [-0.002914, +0.004023] | no |
| category only − D2p | 20260710 | -0.000028 | [-0.003475, +0.003443] | no |
| full − item only | 20260708 | -0.005383 | [-0.008143, -0.002567] | no |
| full − item only | 20260709 | -0.005210 | [-0.007943, -0.002349] | no |
| full − item only | 20260710 | -0.006336 | [-0.009088, -0.003481] | no |
| full − category only | 20260708 | +0.026063 | [+0.022906, +0.029250] | yes |
| full − category only | 20260709 | +0.026398 | [+0.023233, +0.029585] | yes |
| full − category only | 20260710 | +0.026327 | [+0.023185, +0.029550] | yes |

## Decision

Outcome: **TERMINAL_FAIL**. Category-only three-seed mean relative gain over D2p: **0.115%**.

Both the primary claim and the only predeclared fallback fail. Motivation therefore terminates as benchmark/analysis-only; no proposed-system design is authorized from this dev evidence.

The supported diagnostic insight is narrower: the history gain is concentrated in exact repeat-item memory. Category-only alignment has no significant gain in any seed, and full D2s is significantly worse than item-only in every seed. The item-only control is therefore the current static benchmark waterline at a three-seed mean NDCG@10 of **0.345376**. This observation is reportable, but it does not retroactively pass the frozen architecture gate.

## Integrity

Overall integrity: **passed**. No-history rank mismatches: 0; metric mismatches: 0. All six new evaluations are present exactly once in the dev-eval log, and every evaluator artifact uses the same candidate and qrels hashes.
