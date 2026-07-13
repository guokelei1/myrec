# Fxx Failure Card

Status: `draft | rejected | passed`

This template follows `doc/31 §5` and is only for an R0 survivor seeking
architecture entry. Replace `Fxx` only when registering a real failure; do not
create an Hxx source tree while status is `draft` or `rejected`. Use artifact
links and concise facts; do not fill sections with speculative prose.

## Strong Baseline

- Motivation Brief:
- One-sentence paper problem:
- Method/config:
- Model-family adequacy artifact/verdict:
- Dataset/version/split:
- Candidate manifest SHA256:
- Dev evaluator calls used/budget:
- Selected-config rule:
- Seeds:
- Baseline metrics source:

## Failure

- Affected request surface:
- Prevalence:
- Severity and recoverable overall contribution:
- Cross-family failure matrix:
- Transformer asset that must be preserved:
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
- Nearest prior method and reviewer alternative:
- Allowed claim if repaired:
- Claims explicitly not supported:
- Proposed Hxx ID, only after pass:

## Review

- Evidence-hygiene checks:
- Paper-value gate:
- Shared-blind-spot gate:
- Native-Transformer-shortfall gate:
- Asset-preservation gate:
- Reviewer:
- Decision timestamp:
- Decision and reason:
