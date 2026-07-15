# Open exploration protocol

Status: active from 2026-07-14.

## Purpose

Explore whether ordinary LLM4Rec rankers exhibit a reproducible gap between
history response and correct query-conditioned candidate-relative direction.
No expected result is frozen. The process may discover that the hypothesis is
wrong, data-specific, objective-specific, underpowered, or more interesting in
a narrower form.

## What may change during exploration

- dataset order and whether Lite, Full, Small, or an auxiliary dataset is the
  cheapest next source of information;
- diagnostic metrics, cohorts, serialization, ordinary training recipes, and
  model capacity, provided each change and its motivation are logged;
- the working explanation and even the scientific question when evidence
  contradicts it.

A Lite result has Lite scope. It may motivate a Full probe but cannot admit or
reject Full. Likewise, a Full result does not retroactively change what Lite
showed.

## What does not change

- raw source data are immutable and history must precede the target request;
- candidate identity and order are asserted across counterfactual conditions;
- scoring/training code cannot read development, confirmation, or test qrels;
- label diagnostics use the shared evaluator and every dev call is logged;
- test and independent confirmation remain closed;
- no proposed architecture or architecture training precedes a surviving
  Failure Card.

## Observation cycle

For every material step, record:

1. question and reason for asking it now;
2. exact action, input boundary, command or artifact;
3. direct observation, separated from interpretation;
4. plausible explanations, including at least one explanation that would
   weaken the current thesis;
5. uncertainty, integrity issues, and what the observation cannot establish;
6. correction to prior belief, if any;
7. cheapest reversible probe that best separates the explanations.

Raw outputs live under `runs/`, `artifacts/`, or `data/`. Concise observations
live in `doc/dev_log/`; machine-readable state lives in `pipeline_state.yaml`.

## Moving from exploration to confirmation

Only after a pattern survives ordinary implementation and data checks do we
write a new confirmation lock. The lock freezes the eligible population,
dataset roles, endpoints, thresholds, checkpoint-selection rule, and analysis.
Anything already inspected during exploration remains development evidence and
is not represented as independent confirmation.
