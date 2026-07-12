# Intro and Motivation Completion Report

> **Current supersession (2026-07-13).** “Motivation complete” below describes
> the C5-R3 evidence state, not current architecture readiness. C01--C80 later
> closed; current work is R0 problem discovery under
> [`doc/31`](../doc/31_problem_discovery_and_architecture_iteration_protocol.md).

> **Terminal supersession / 当前解释（2026-07-11）.** 当前以
> [`doc/15_proposed_system_design_principles.md`](../doc/15_proposed_system_design_principles.md)
>、[`reports/pps_architecture_readiness.md`](../reports/pps_architecture_readiness.md)
> 和 [`terminal closure`](../doc/dev_log/20260711_architecture_exploration_terminal_closure.md)
> 为准：motivation complete；后续 C01--C16 已全部关闭，未得到经过验证的
> architecture primitive，也未授权 proposed-system dev/full/test evaluation。
> C5-R3 FAIL 及全部数字不变。

Date: 2026-07-10

Status: **historical motivation completion; the later C01--C16 architecture
portfolio is terminally closed without a validated primitive**.

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
  -> history evidence has unequal empirical fidelity
  -> four independent LLM4Rec/Transformer formulations may test distinct mechanisms
  -> a candidate surviving its minimal probe may enter the full design gate
  -> only a candidate passing that full gate may enter full training
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
`TERMINAL_FAIL` for that preregistered multi-granular/coarse-category recovery
ladder—not a broken run and not a prohibition on a newly scoped design hypothesis.

## Current Static Waterline

| Control | Three-seed mean NDCG@10 | Role |
|---|---:|---|
| D2p | 0.3239501 | strongest frozen non-personalized base |
| Full D2s | 0.3416290 | historical complete static mixture |
| **C5-R3 item-only D2s** | **0.3453755** | current strongest static benchmark |
| C5-R3 category-only D2s | 0.3241931 | no significant history-present gain |

The 2% reference over this static waterline is approximately 0.3522831. It is a
performance reference for later full-system evaluation, not evidence that any
current candidate has passed its design gate; current minimal probes and their
authorization rules are frozen separately in `doc/15` and `doc/24`.

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

## Current Stage Decision

Motivation work is complete. The truthful paper position remains a bounded
result about repeat-item concentration, unequal history-evidence fidelity, and
the failure of the tested coarse semantic alignment. That result now motivates
four deliberately different LLM4Rec/Transformer candidate formulations under
`doc/24`; each may build only the minimal prototype needed for its frozen
dev-only falsifier. Full implementation, tuning, and training are authorized
separately for a candidate only after that candidate passes the common contract
and its design-specific gate in `doc/15`/`doc/24`. This transition does not
reinterpret C5-R3, does not revive its failed category claim, and does not
unlock test.

Primary artifacts:

- `doc/15_proposed_system_design_principles.md`
- `doc/24_parallel_llm4rec_design_protocol.md`
- `doc/23_c5r3_candidate_history_alignment_protocol.md`
- `configs/analysis/c5r3_candidate_history_alignment.yaml`
- `reports/pps_c5r3_candidate_history_alignment.{json,md}`
- `reports/pps_c5r3_consistency_audit.json`
- `reports/pps_c5_insight_audit.json`
- `reports/pps_architecture_readiness.md`
- `reports/pps_intro_motivation_dev_eval_reconciliation.json`
- `experiments/pps_results.md`
