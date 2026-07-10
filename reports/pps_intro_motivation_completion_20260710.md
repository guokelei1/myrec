# Intro and Motivation Completion Report

> **Current supersession / 当前解释（2026-07-10）.** 下文的
> benchmark-only/no-design 表述是当时或该特定 gate 的结论。当前以
> [`doc/15_proposed_system_design_principles.md`](../doc/15_proposed_system_design_principles.md)
> 和 [`reports/pps_architecture_readiness.md`](../reports/pps_architecture_readiness.md)
> 为准：motivation complete，design formulation ready；implementation/training
> 仍由新的、design-specific pre-outcome falsifier 把关。C5-R3 FAIL 及全部数字不变。

Date: 2026-07-10

Status: **complete and internally consistent; terminal outcome is
benchmark/analysis-only, with no proposed-system architecture authorization**.

## Final Motivation Conclusion

The complete repaired evidence supports one narrow diagnostic insight:

> In the frozen KuaiSearch fixed-candidate protocol, the stable gain from the
> tested history heuristic is concentrated in exact candidate-item recurrence.
> Coarse category alignment has no significant independent gain and degrades
> the stronger item-only control when added under the frozen mixture.

This replaces the earlier, broader query-anchored personalized-residual story.
It is a valid motivation result because it identifies what the current static
gain actually measures and what it does not measure. It is not promoted into an
architecture premise after the fact.

## Final Evidence Chain

```text
fixed candidate pool is already query-conditioned
  -> fine-tuned text + legal train popularity gives D2p (0.3240)
  -> full D2s appears to add history (0.3416)
  -> temporal control does not establish stable same-query identity specificity
  -> pre-outcome B0b decomposition exactly separates item and category evidence
  -> item-only + D2p reaches 0.3454 and wins in all three seeds
  -> category-only adds approximately zero in all three seeds
  -> full D2s is significantly worse than item-only in all three seeds
  -> apparent static history value is therefore dominated by exact-item recurrence
  -> motivation closes as benchmark/analysis evidence; design remains unstarted
```

## C5-R3 Results

All paired comparisons use NDCG@10 on the frozen 8,119 history-present request
IDs with 10,000 bootstrap samples.

| Comparison | Seed 20260708 | Seed 20260709 | Seed 20260710 | Decision |
|---|---:|---:|---:|---|
| Item only − D2p | +0.03204 | +0.03214 | +0.03263 | all CIs positive |
| Category only − D2p | +0.00059 | +0.00053 | -0.00003 | all CIs cross zero |
| Full D2s − item only | -0.00538 | -0.00521 | -0.00634 | full significantly worse in all seeds |
| Full D2s − category only | +0.02606 | +0.02640 | +0.02633 | all CIs positive |

The multi-granular primary fails because category-only is not independently
useful and full D2s does not beat item-only. The sole predeclared fallback fails
because category-only has zero significant seeds and only **0.1148%** mean
relative gain, far below its frozen 2% threshold. The formal outcome is
`TERMINAL_FAIL`, meaning benchmark/analysis-only—not a broken run.

## Current Static Waterline

| Control | Three-seed mean NDCG@10 | Role |
|---|---:|---|
| D2p | 0.3239501 | strongest frozen non-personalized base |
| Full D2s | 0.3416290 | historical complete static mixture |
| **C5-R3 item-only D2s** | **0.3453755** | current strongest static benchmark |
| C5-R3 category-only D2s | 0.3241931 | no significant history-present gain |

The future 2% reference, if a later protocol retains that rule, is approximately
0.3522831. This number is not a current design target because no architecture is
authorized; it only prevents future work from comparing against the weaker
0.3416 waterline.

## Integrity

- `doc/23` and its executable config were frozen before component scores or
  results existed.
- On all 575,609 candidate rows, item plus category reproduces both the public
  B0b scorer and the actual upstream B0b score file within `1e-12`; maximum
  absolute error is `7.1054e-15`, with zero violations.
- Materialization and scoring read standardized dev records and upstream score
  files but no qrels; only the shared evaluator read `qrels_dev`.
- All methods use candidate hash
  `94eb667000e0d0f389d0a2a4d4730683b71c129043edbfcf627590376e9c123e`.
- All metrics use qrels hash
  `518eab43850c6fbc841cfa5f047602a1e41761960bc80d244c93fb379b0029bc`.
- All six new C5-R3 evaluator calls are logged exactly once.
- Item-only and category-only controls are rank- and metric-equivalent to D2p
  on all 4,110 history-absent requests.
- Test was not read, evaluated, or used for selection.

## Scientific Boundary

Supported:

- tested query/text/popularity controls leave a candidate-aligned repeat-item
  signal;
- exact repeat-item history is strong and stable in the current protocol;
- tested coarse category history is not independently useful;
- full B0b/D2s can underperform a simpler removal ablation;
- no-history requests require exact non-personalized fallback;
- simple query-attentive residuals, entropy routing, and the original oracle
  argument are not positive evidence.

Not supported:

- all useful personalization is exact repetition;
- semantic history transfer, stable same-query identity specificity, or a
  randomized causal effect;
- query-conditioned event selection as an established mechanism;
- an oracle-shaped router or any untrained proposed architecture;
- using test to confirm, rescue, or select the motivation.

## Terminal Decision

Motivation work requested by the current goal is complete. The truthful paper
position is a benchmark/analysis result about repeat-item concentration and the
failure of coarse semantic history alignment. Proposed-system design and
training have not begun. If design is requested later, it must start from a new
question and a new pre-outcome protocol rather than reinterpret C5-R3.

Primary artifacts:

- `doc/23_c5r3_candidate_history_alignment_protocol.md`
- `configs/analysis/c5r3_candidate_history_alignment.yaml`
- `reports/pps_c5r3_candidate_history_alignment.{json,md}`
- `reports/pps_c5r3_consistency_audit.json`
- `reports/pps_c5_insight_audit.json`
- `reports/pps_architecture_readiness.md`
- `reports/pps_intro_motivation_dev_eval_reconciliation.json`
- `experiments/pps_results.md`
