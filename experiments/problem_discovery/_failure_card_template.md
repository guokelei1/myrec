# Fxx Failure Card

Status: `draft | rejected | passed`

This template follows `doc/31 §5`. Replace `Fxx` only when registering a real
failure; do not create an Hxx source tree while status is `draft` or `rejected`.

## Strong Baseline

- Method/config:
- Dataset/version/split:
- Candidate manifest SHA256:
- Dev evaluator calls used/budget:
- Selected-config rule:
- Seeds:
- Baseline metrics source:

## Failure

- Affected request surface:
- Counterfactual/intervention:
- Expected behavior:
- Observed ranking failure:
- Paired effect and confidence interval:
- Minimum claimable effect:
- Power analysis:

## Replication

- Independent split or dataset:
- Information-object comparability:
- Replicated effect and confidence interval:
- Direction/heterogeneity decision:

## Simpler Repairs

| Repair/control | Budget | Result | Why insufficient |
|---|---:|---|---|
| capacity/compute matched | | | |
| context/token budget | | | |
| ordinary optimization | | | |
| ordinary attention/existing mechanism | | | |

## Localization

- Representation/attention/objective/ranking locus:
- Weight-frozen or same-checkpoint intervention:
- Utility consequence:
- Alternative explanations not excluded:

## Architecture Entry

- Cheapest falsifier:
- Nearest simpler mechanism:
- Allowed claim if repaired:
- Claims explicitly not supported:
- Proposed Hxx ID, only after pass:

## Review

- Evidence-hygiene checks:
- Reviewer:
- Decision timestamp:
- Decision and reason:
