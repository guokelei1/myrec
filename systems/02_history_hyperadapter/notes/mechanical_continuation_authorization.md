# C02 mechanical-continuation authorization

Date: 2026-07-11 (Asia/Shanghai)

Status: **authorized once, before any valid C02 train-internal or dev outcome.**

The user explicitly requested that the current architecture exploration be
finished after the portfolio audit established that C02 had stopped on an
empty-mask implementation defect rather than a scientific gate.  The
coordinator therefore authorizes exactly the continuation requested in
`mechanical_continuation_request.md`:

> Authorize one C02 mechanical continuation attempt under
> `notes/mechanical_continuation_request.md`; all scientific settings and the
> single dev-call budget remain frozen.

The only permitted source behavior change is a differentiable zero from
`corruption_loss` when its complete batch contains no history-bearing
candidate, together with a regression test covering that full-loss case.
Seed, features, model, rank, optimizer, epochs, variants, thresholds, internal
subsets, candidate manifest, evaluator budget and test lock remain unchanged.
All five variants restart from their frozen initialization.  A new
continuation lock must be written after tests pass and before GPU training.

If the train-internal gate fails, C02 closes before dev scoring.  If it passes,
only the already registered label-free score/determinism path and sole shared
dev-evaluator call are permitted.  Multi-seed, test, rescue modules, threshold
changes and full training remain unauthorized.
