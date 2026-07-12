# C18 synthetic gate outcome

Status: **failed 0/3 seeds; terminal stop before repository data**.

The locked command ran once on physical A40 GPU 2 for 219.75 seconds.  Each of
the three trainable modes completed the same 800 steps in each seed with finite
losses, identical parameter count and byte-identical initialization.  The lock
aggregate was
`ff3d31855113ba63f42a1e8d5ec8d607cfbb1ef391334d3835097fd2f2ed37bb`.

## Result

| seed | projection repeat | projection supported non-repeat | base supported | clean margin gain over base | max protected violation |
|---:|---:|---:|---:|---:|---:|
| 20260718 | 1.0000 | 0.5052 | 0.5052 | `-3.44e-8` | `5.36e-7` |
| 20260719 | 1.0000 | 0.5156 | 0.5156 | `-7.40e-8` | `4.77e-7` |
| 20260720 | 1.0000 | 0.5039 | 0.5039 | `-9.62e-9` | `5.96e-7` |

The projection was active on every repeat request and the history-present score
delta passed the frozen load-bearing magnitude check in every seed.  Candidate
permutation error was at most `2.87e-6`; deterministic rescore and bitwise
no-history fallback passed.  Thus the failure is not an inactive layer,
constraint violation, numerical instability, or implementation no-op.

The transfer direction was useless.  On supported non-repeat requests the
projected model was effectively its own base and clean target-margin gain was
zero.  Consequently the positive-gain and corruption-retention denominators
failed, and the worst-subset advantage over direct/soft controls failed in all
seeds.  C18 successfully prevents speculative transfer from overriding exact
recurrence, but it does not create candidate-aligned transferable evidence.

## Decision and boundary

C18 is closed.  Do not change centring order, proposal nonlinearity, repeat
bonus, synthetic construction, loss, steps, thresholds or controls and rerun
under this ID.  No standardized record, repository label, dev evaluator, test
data or qrels was accessed; `reports/dev_eval_log.jsonl` remained unchanged.

Raw report:
`artifacts/c18_evidence_constrained_order_transformer/synthetic_gate_v1.json`
(`9efd7f0b223295720928e132aa8b211f43300126e2c5341495740a65fb73ecaf`).

The next candidate must impose a candidate-aligned **transfer law** before the
score proposal, rather than another final safety projection or magnitude
normalizer.
