# C02 mechanical-continuation authorization request

Status: **awaiting coordinator approval**.  This document does not authorize
or apply a source change, a third training attempt, or a dev evaluation.

## Why authorization is required

The locked implementation attempt 2/2 reached a deterministic all-no-history
batch and stopped before checkpointing.  There are 16 such batches in the
frozen epoch-1 order; the first is batch 134.  In `corruption_loss`, the valid
mask is empty and each corruption branch computes `.mean()` over an empty
tensor, producing NaN.  No C02 score file or dev metric exists, and the single
dev-evaluator allowance remains unused.

The prompt and locked protocol cap C02 at two implementation attempts.  A
source edit plus another training run therefore needs explicit new authority;
automatic goal continuation is not that authority.

## Requested scope

Authorize exactly one **mechanical continuation attempt**, without changing
the scientific proposal or screening protocol:

1. change only the empty-valid-mask branch of `corruption_loss` so a batch with
   no history-bearing candidate returns a differentiable scalar zero;
2. add a unit regression test covering an all-no-history batch under the full
   CHHT training loss;
3. keep seed, model, rank, layers, data sample, optimizer, epochs, controls,
   thresholds, subsets, run ID, candidate hash, and score definition unchanged;
4. generate a new source hash and continuation lock before GPU training;
5. restart all five variants from their frozen common initialization—do not
   resume or reuse the failed partial attempt;
6. run train-internal validation; if it is numerically valid, produce the one
   label-free dev score file, deterministic rescore, and the still-unused
   single shared-evaluator call;
7. stop and report after the gate, whether it passes or fails.  No multi-seed
   or full-gate training is included.

The intended executable behavior is equivalent to:

```python
valid = candidate_mask & history_mask.any(dim=-1)[:, None]
if not valid.any():
    return true_norm.sum() * 0.0
```

The exact patch must be tested before it is locked; this snippet is a review
aid, not an applied change.

## Unchanged integrity envelope

- environment: `myrec-c02`;
- physical GPU: 1, exposed as `cuda:0`;
- deterministic CuBLAS workspace: `:4096:8`;
- seed: `20260708`;
- candidate-manifest SHA-256:
  `94eb667000e0d0f389d0a2a4d4730683b71c129043edbfcf627590376e9c123e`;
- evaluator calls already consumed: 0;
- C02 dev-log rows already written: 0;
- held-out test access: prohibited;
- sibling design/outcome access: prohibited;
- remaining original GPU ceiling: more than 7.97 of 8 A40 GPU-hours.

## Approval language

A coordinator can authorize this precisely by replying:

> Authorize one C02 mechanical continuation attempt under
> `notes/mechanical_continuation_request.md`; all scientific settings and the
> single dev-call budget remain frozen.

Anything broader requires a separately specified protocol.
