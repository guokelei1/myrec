# External Audit Prompt: Introduction and Motivation

> **Current supersession (2026-07-13).** Do not reuse this prompt. Current
> architecture-readiness audits must use
> [`doc/31`](../31_problem_discovery_and_architecture_iteration_protocol.md) and
> the C01--C80 terminal retrospective.

> **DO NOT REUSE — HISTORICAL / SUPERSEDED PROMPT (2026-07-10).** This mandate
> describes a pre-C5-R3 evidence state (including the then-current D2s comparator
> and 127-entry log) and must not be used to audit or control later architecture
> work. Current authority is
> [`doc/15_proposed_system_design_principles.md`](../15_proposed_system_design_principles.md),
> [`reports/pps_architecture_readiness.md`](../../reports/pps_architecture_readiness.md),
> and the 2026-07-11
> [`terminal closure`](../dev_log/20260711_architecture_exploration_terminal_closure.md):
> motivation remains complete, but C01--C16 are now closed without a validated
> architecture primitive or proposed-system dev/full/test authorization. The
> body below is preserved verbatim as a historical reviewer mandate.

Copy the prompt below into a fresh reviewer/agent session.

---

You are conducting an independent, adversarial audit of the PPS research
repository at `/data/gkl/myrec`. The audit covers all work from the problem
definition through the completed Introduction/Motivation stage. It does not
cover implementation of the proposed system.

Your job is to determine whether the current evidence is correct, reproducible,
fairly compared, logically sufficient, and strong enough to authorize proposed-
system development. Do not assume that existing completion reports or prose are
correct. Treat every headline number and interpretation as a claim to falsify.

## Operating Rules

1. Start by reading the repository `AGENTS.md` and obey it.
2. Work read-only by default. Do not revert, rewrite, or clean the existing dirty
   worktree. Do not make a commit.
3. Do not run new model training, hyperparameter search, or any test evaluation.
   Deterministic unit tests, parsers, score audits, hash checks, and recomputation
   from existing dev artifacts are allowed.
4. Do not read `qrels_test.jsonl`, compute test metrics, or use test records for
   analysis. Merely confirm that test isolation remains intact.
5. Do not use the current paper draft as the source of truth. Trace paper claims
   back to evaluator outputs, frozen configs, protocols, and code.
6. If you find a major issue, report it before proposing a repair. Do not silently
   reinterpret a failed gate or replace a metric after seeing results.
7. Preserve historical negative results. In particular, do not rehabilitate M3,
   M4, entropy laws, or query attention unless independent evidence actually
   supports doing so.

## Claimed Current State to Challenge

The repository currently claims the following. These are audit targets, not
assumptions:

- The fixed KuaiSearch candidate pool is already query-conditioned.
- Raw ranking `recently_*` histories failed the registered future-leakage check
  and are not used. Standardized histories are rebuilt from same-user recall
  events with `event_time < request_time`.
- D2p is the strongest registered non-personalized control, with three-seed mean
  dev NDCG@10 approximately `0.323950`.
- D2h is a valid intermediate static control at approximately `0.335213`, but it
  omitted the popularity component already present in D2p.
- The post-result fairness repair D2s freezes D2p, combines it with B0b using a
  train-only selected `beta=0.3`, and reaches three-seed mean dev NDCG@10
  `0.3416289845` with sample SD `0.0003711265`.
- At seed 20260708, D2s exceeds D2h by `+0.0063627404`, with paired-bootstrap
  95% CI `[+0.0037327413, +0.0090111998]`.
- On 8,119 history-present requests, true D2s exceeds matched wrong-history D2s
  by mean `+0.0353653` across seeds; on the 2,709-request same-query donor subset,
  the mean is `+0.0276284`. Every seed-level CI is positive.
- On all 4,110 no-history requests, D2s and seed-matched D2p have exactly equal
  NDCG@10, MRR, and Recall@10.
- D1m/D1a do not stably improve D1q; query-attentive event selection is therefore
  a design hypothesis, not an established observation.
- M3/M4 oracle evidence is construct-invalid because Random-channel controls are
  at least as strong. It is retained only as a negative methodological result.
- B9 ZAM/TEM numbers are supplementary and non-load-bearing while human-review
  provenance and external alignment remain incomplete.
- The current binding dev comparator is D2s. A full performance claim for the
  proposed system requires significant improvement over D2s and at least 2%
  relative gain, approximately NDCG@10 `0.3485`.
- Test metrics have never been used. The current conclusion is a dev-stage
  authorization to design the system, not final paper confirmation.

## Required Reading

Read at minimum:

- `AGENTS.md`
- `doc/07_paper_design_constraints.md`
- `doc/11_experiment_and_dataset_plan.md`
- `doc/12_experiment_execution_protocol.md`
- `doc/13_baseline_implementation_plan.md`
- `doc/14_official_baseline_plan.md`
- `doc/17_intro_motivation_repair_protocol.md`
- `doc/18_supervised_motivation_diagnostics_protocol.md`
- `doc/19_finetuned_nonpersonalized_control_protocol.md`
- `doc/20_d2h_static_history_waterline_protocol.md`
- `doc/21_d2s_static_full_waterline_protocol.md`
- `paper/introduction_and_motivation.md`
- `paper/introduction_motivation_sentence_plan.md`
- `doc/15_proposed_system_design_principles.md`
- `experiments/pps_results.md`
- `experiments/pps_baseline_cards.md`
- `reports/pps_intro_motivation_completion_20260710.md`
- `reports/pps_architecture_readiness.md`
- `reports/pps_c0_data_audit.json`
- `reports/pps_m3_m4_random_canary_audit.json`
- `reports/pps_supervised_diagnostics_summary.json`
- `reports/pps_d2_d2h_summary.json`
- `reports/pps_d2s_summary.json`
- `reports/pps_d2s_score_audit.json`
- `reports/pps_d2s_protocol_lock_manifest.json`
- `reports/pps_d2s_calibration_semantics_verification.json`
- `reports/pps_c5_insight_audit.json`
- `reports/pps_intro_motivation_dev_eval_reconciliation.json`
- `reports/dev_eval_log.jsonl`

Inspect relevant implementation rather than trusting metadata, especially:

- `src/myrec/eval/`
- `src/myrec/baselines/core.py`
- `src/myrec/analysis/history_identity.py`
- `src/myrec/analysis/supervised_diagnostics.py`
- `src/myrec/analysis/finetuned_query_tower.py`
- D1/D2/D2h/D2s scripts under `scripts/`
- metric, evaluator, history-identity, D2, D2s, and adapter tests under `tests/`

## Audit Workstreams

### 1. Data and Leakage

- Verify why raw `recently_*` failed and that no current method still consumes it.
- Verify standardized history construction is strictly prior by code, including
  tie handling, split boundaries, maximum length, and early-window emptiness.
- Check whether any history, popularity, donor, embedding, or calibration feature
  can incorporate dev/test labels or future interactions.
- Distinguish a temporal construction guarantee from a deployed causal-effect
  claim and check that the paper does the same.

### 2. Evaluator and Result Provenance

- Independently recompute or validate every load-bearing number from existing
  `metrics.json`, `per_request_metrics.jsonl`, score files, and comparison JSONs.
- Verify three-seed means and sample standard deviations; ensure no best-seed
  number is presented as a multi-seed result.
- Verify all paired-bootstrap comparisons use identical request sets and the
  intended preselected bootstrap seed.
- Assert candidate-manifest and qrels hashes for every relevant evaluator call.
- Reconcile all 127 current dev-eval log entries, including the two documented
  duplicate R1 run IDs and the six D2s evaluations.
- Check `experiments/pps_results.md` against evaluator outputs, not summaries.

### 3. Train/Dev Isolation and Ordering

- Verify D1/D2 epochs and D2p/D2h/D2s mixture weights were selected only from
  train-side calibration data.
- Audit the invalid first D2p alpha calculation and prove that it did not affect
  final D2p/D2h/D2s dev scores or model selection.
- Verify protocol/config/calibration/final-config/score/evaluation ordering from
  hashes and timestamps, especially for the post-result D2s repair.
- Confirm D2s calibration and final scorer implement equivalent ranking semantics.

### 4. Baseline Fairness

- Determine whether D2s is a defensible current registered static waterline.
- Specifically search for another obvious, already-available static combination
  or feature that was unfairly omitted. Do not invent an unlimited sweep; explain
  whether the frozen hierarchical D2p-plus-B0b construction is sufficient.
- Verify B4o/B5o/B9 identity labels and caveats are preserved.
- Check that B8 subset evidence is not presented as a direct full-dev D2s
  comparison and that provisional B9 evidence is non-load-bearing.

### 5. Statistical and Construct Validity

- Check whether matched wrong-user donors truly exclude the target user and use
  train-only histories. Audit same-query subset construction across all seeds.
- Assess what the permutation supports: identity-specific predictive value, not
  randomized causal effect.
- Verify no-history equality at both score/ranking and metric levels.
- Re-evaluate the M3/M4 Random-canary argument and confirm no surviving positive
  claim depends on the invalid oracle.
- Check whether D1 negative results justify only a bounded statement about the
  tested representation family.

### 6. Argument Audit

For every paragraph in `paper/introduction_and_motivation.md`, classify each
empirical sentence as supported, overstated, ambiguous, or unsupported. Check
the complete chain:

`query-conditioned pool -> strong non-personalized control -> complete static
correct-history gain -> matched identity dependence -> no-history boundary ->
simple learned residual failure -> query-anchored personalized-residual design
hypothesis`.

Determine whether the final architecture consequence follows as a justified,
falsifiable hypothesis without pretending that query attention has already
worked. Identify any alternative design implication equally supported by the
same evidence.

### 7. Reproducibility and Repository Hygiene

- Run the complete unit-test suite and compile relevant Python source.
- Validate all JSON/YAML and check `git diff --check`.
- Verify tracked/untracked source does not include checkpoints, raw datasets,
  score dumps, credentials, or other prohibited large experiment state.
- Confirm all paper-facing paths exist and all current/superseded reports are
  clearly labeled.

## Required Deliverable

Write `reports/pps_intro_motivation_external_audit_20260710.md` with this exact
structure:

1. **Verdict**: `GO`, `CONDITIONAL GO`, or `NO-GO` for proposed-system design.
2. **Findings first**, ordered Critical / High / Medium / Low, each with precise
   file and line references, affected claim, and required action.
3. **Claim matrix** covering every bullet in “Claimed Current State to Challenge”
   with verdict `verified`, `bounded`, `failed`, or `not independently verified`.
4. **Independent number table** containing D2p, D2h, D2s, D2s-wrong means/SDs,
   key paired deltas/CIs, request counts, candidate hash, and qrels hash.
5. **Protocol-integrity table** covering leakage, label isolation, ordering,
   dev-eval accounting, and test isolation.
6. **Logic assessment** explaining whether the evidence supports the proposed
   design direction and which statements must remain explicitly bounded.
7. **Residual risks** separating blockers from nonblocking paper-completion work.
8. **Commands run** and tests/checks that could not be run.

If there are no Critical or High findings, say so explicitly. Do not give a GO
merely because existing reports say “passed.” A GO must follow from independent
recomputation and code inspection. Do not alter the paper or protocols during
this first-pass audit.

---
