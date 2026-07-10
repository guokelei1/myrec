# 07 - Paper Design Constraints

This document is normative. It defines constraints that any future paper design
must satisfy. It does not choose the current method, current main insight, or
current architecture direction.

PPS audit amendment (2026-07-10): an oracle that selects and evaluates a
per-request winner on the same labels must include a Random-channel null or an
independent selection/evaluation split. Split-half stability after taking the
maximum is insufficient. Without this control, oracle headroom cannot satisfy
Tier 1 evidence hygiene.

The role of this document is to prevent a common failure mode:

```text
many useful mechanisms + many special cases + many controls
=> a system that works like several stitched papers, not one paper.
```

## 0. Constraint Phasing (Read This First)

The constraints below are not all equally urgent. Enforcing all of them during
idea exploration kills iteration speed, which is its own failure mode. Use two
tiers.

### Tier 1 — obey from day one

These few constraints shape the design itself, and skipping them makes every
fast experiment invalid or unreusable later:

1. **Insight template (Section 1).** One sentence, one primitive, one cheap
   falsification test. If the template cannot be filled, do not start building
   the system. This is a 30-minute exercise, not completeness work.
2. **Experiment ordering (Section 7).** Data check, strongest single-channel
   control, oracle headroom, cheapest falsifier — in that order. This IS the
   fast-iteration loop, not overhead on top of it.
3. **Unified interface as a design habit (Sections 2–3).** Costs nothing when
   designing; retrofitting it after per-dataset branches exist forces a
   rewrite. Just do not write `if dataset == D1` from the start.
4. **Minimal evidence hygiene.** A fixed split, identical candidate sets
   across methods, and all decisions on dev. Without these three, every fast
   result is throwaway and must be rerun before it can support anything.
5. **One honest control per claim.** Strongest single channel plus static
   mixture. At idea stage, one reasonably-tuned control is enough; the full
   baseline suite is not required yet.

### Tier 2 — defer until the idea survives Tier 1

Full baseline tuning (Section 9), multi-seed statistics (Section 11),
efficiency/Pareto protocol (Section 12), behavior-preservation finals gate
(Section 13), complexity budget enforcement (Section 14), the attack table
(Section 15), writing constraints (Section 8), and the full control families
(Section 6). Run these once, on the surviving design, before finals and
writing.

Early-stage relaxations that are explicitly allowed:

- single seed, marked as provisional;
- one dataset per claim family instead of all;
- default-config baselines, labeled as such;
- rough latency numbers or none at all.

Rule of thumb:

```text
Tier 1 exists so that fast experiments produce valid, reusable evidence.
Tier 2 exists so that the surviving design becomes a paper.
Never let Tier 2 slow down Tier 1.
Never promote Tier 1 results to paper claims without Tier 2.
```

## 1. Core Insight Constraint

The paper must have one or two load-bearing insights. A load-bearing insight is
not a component list, not a result table, and not a training trick. It is a
compact observation that explains why the architecture should exist.

RPG-style example:

```text
Semantic information can be represented as a sequence and modeled by a
sequential recommender.
```

Any future LRM1 paper insight must satisfy all of these:

- it can be stated in one sentence;
- it is not dataset-specific;
- it implies a concrete architecture primitive;
- it predicts at least one failure mode;
- it can be falsified by a cheap control before the full system is built;
- it is stronger than "we combine lexical, semantic, and collaborative signals";
- it remains meaningful if some individual dataset result is negative.

Required template:

```text
Observation: <one-sentence insight>.
Architecture consequence: therefore the model uses <one primitive>.
Falsification: if <cheap control> happens, the insight is false or too weak.
```

If this template cannot be filled, the design is not paper-shaped yet.

## 2. Unified Architecture Constraint

The method must be one architecture for query and no-query records. It may
condition on evidence availability, but it must not become separate per-dataset
or per-surface code paths.

Allowed conditioning:

- query present or absent;
- history present or absent;
- candidate text present or absent;
- item ID available or unavailable;
- history length;
- query length;
- candidate count;
- text coverage;
- evidence confidence or uncertainty.

Disallowed conditioning:

- if dataset is D1, use module A;
- if dataset is D2, use module B;
- if dataset is D3/D4, bypass the sequence architecture;
- if query dataset, use a separately tuned scorer;
- if no-query dataset, use a separate recommender and report the two as one
  unified method.

The paper must be explainable without saying:

```text
For query data we do X, but for sequence data we do unrelated Y.
```

Acceptable form:

```text
All records are converted into the same model interface. Missing evidence is
masked. The same reasoning primitive processes the resulting instance.
```

## 3. Shared Record Interface Constraint

Every supported task should fit one abstract record:

```text
x = (optional query evidence, optional history evidence, fixed candidate set)
```

The method may represent this as a sequence, graph, table, set, or tensor, but
the representation must preserve:

- fixed candidate identity;
- query evidence when present;
- history evidence when present;
- candidate text or text-missing flags;
- identity/behavior evidence when present;
- evidence availability masks;
- candidate-set-relative features or scores if used.

Query and no-query examples should differ by missing evidence, not by a
different architecture contract.

## 4. Primitive Before Components

The method section must introduce one primitive before listing modules.

Bad structure:

```text
We have BM25, semantic encoder, ID memory, graph module, router, certificate,
and several fallback rules.
```

Required structure:

```text
The paper's primitive is <P>. BM25, semantic encoders, ID memory, and
certificates are instances, channels, or diagnostics inside <P>.
```

If the contribution lives only in a hand-designed fusion layer over named
experts, the paper will read as an ensemble. That is not automatically invalid,
but the claim must shrink to a systems/analysis claim unless the fusion itself
has a clear, falsifiable primitive.

## 5. Claim Separation Constraint

Do not let one result table imply more than it proves.

Claims must be separated by evidence type:

| Claim type | Required evidence |
|---|---|
| Sequence/behavior claim | no-query datasets with valid history and identity signal |
| Query semantic claim | query datasets with true query-item labels |
| Unified architecture claim | same model contract across query and no-query records |
| Dynamic/reasoning claim | surplus over static mixture and strongest single channels |
| Explanation claim | attribution/certificate correlates with drop-channel effects |
| Efficiency claim | latency includes online feature construction and scoring |

Rules:

- A D2-like dataset with missing item text cannot support semantic language
  claims.
- D3/D4 final diagnostics cannot be used for architecture selection unless they
  have a proper dev protocol.
- Query gains must be reported separately from sequence gains.
- A unified method may have different performance tradeoffs across surfaces,
  but it must not hide separate mechanisms behind one name.

## 6. Required Control Families

Every serious paper design must include controls that attack the central claim.

### 6.1 Component controls

Run controls for:

- strongest single channel or component;
- static mixture;
- same-encoder pooled scoring;
- same-encoder late-interaction or BAP/QSM-style scoring if relevant;
- lexical/title/full-query retrieval;
- identity-only sequence scoring;
- popularity/source-order where artifacts are plausible;
- parameter/compute-matched variant of the backbone, so gains are attributable
  to the mechanism, not extra capacity.

### 6.2 Query causality controls

For any query claim:

- true query;
- wrong query;
- no query;
- selected/partial query if selection is used;
- background/entity-only if role decomposition is used;
- selector-masked query if selection is used.

If wrong-query or no-query equals or beats true-query, the query-understanding
claim is withdrawn or narrowed.

### 6.3 Unified-architecture controls

To prove the method is not two systems stitched together:

- remove dataset ID from the model;
- test query-only and sequence-only specialist baselines;
- test a shared model contract with missing-evidence masks;
- include mixed-evidence records when the claim involves arbitration;
- report results by evidence profile, not only by dataset.

### 6.4 Explanation controls

For any explanation/certificate claim:

- drop each channel or evidence family;
- compare attribution share with score/rank movement;
- report insufficient-observation and constant-rank cases honestly;
- include failure examples where explanation confidence is low.

## 7. Experiment Ordering Constraint

Do not build the full system before cheap falsification.

Minimum order:

1. Verify data, split, and candidate equality.
2. Reproduce strong single-channel controls.
3. Measure oracle headroom for any adaptive mechanism.
4. Audit whether routing/reasoning features correlate with oracle decisions.
5. Run the cheapest learner or ablation that can falsify the core insight.
6. Only then build the full architecture.
7. Freeze configs and run final tests once.

If oracle headroom is near zero, do not build an adaptive mechanism. If the
features that drive adaptation are uninformative, do not make a dynamic
reasoning claim.

## 8. Writing Constraints

The paper must obey these writing rules:

1. The abstract contains the core insight in one sentence.
2. The introduction states the failure mode that the insight predicts.
3. The method introduces one primitive before components.
4. Query and no-query handling are described through the same interface.
5. Negative controls appear in the main paper when they decide the claim.
6. The main table separates query and no-query results.
7. Datasets with missing semantic evidence are labeled honestly.
8. Explanations cite measured faithfulness, not only visualization.
9. Latency claims state exactly what is included.
10. Failed gates shrink claims instead of being explained away.

## 9. Baseline Strength Constraint

The most common reviewer rejection is not a weak method but a weak baseline.
Doc 01 names four serious baselines (HSTU, RPG, LLM-SRec, A-LLMRec); this
section defines what "serious" means operationally.

Rules:

- Every named baseline receives the same hyperparameter search budget as the
  proposed method (same number of dev evaluations, or the asymmetry is
  documented). Keep a tuning log per baseline.
- Where published numbers exist on a comparable setup, reproduce them or
  document the gap and its cause before any comparison is used as evidence.
- All methods share preprocessing, splits, candidate sets, and metric code.
  A method-specific data pipeline invalidates the comparison.
- "Beats an untuned or default-config baseline" is not evidence.
- A gain over a baseline that lacks access to an evidence channel (e.g., item
  text, query text) must be labeled as an evidence-access gain, not a modeling
  gain. Give the baseline the same evidence when the architecture permits.

## 10. Evaluation Protocol Constraint

Fix all of the following before any model comparison, and never change them
after results are seen:

- Split protocol: prefer temporal splits; whatever is chosen, state it and
  apply it identically to every method.
- Candidate protocol: all methods rank the identical candidate set per record.
  Per-method candidate pools are forbidden.
- Sampled-metric caution: sampled negatives can invert method ordering. Use
  full ranking, or a fixed shared negative sample generated once and audited
  against full ranking on at least one dataset.
- Metric declaration: one primary metric per claim, declared in advance. The
  primary metric decides claims; secondary metrics only describe.
- Dev decides everything; test runs once per frozen config (extends 7.7).
- Report every attempted dataset. Dropping a dataset requires a stated,
  data-level reason recorded before its test results were seen.

## 11. Statistical Validity Constraint

- Any trainable comparison used in a claim runs at least 3 seeds; report mean
  and variability. Never report the best seed.
- Declare a minimal claimable effect per primary metric before finals.
  Differences below it are reported as ties, not wins.
- A claim supported by only one dataset must be labeled dataset-specific.
- If a gate threshold and the measured effect are within noise of each other,
  the gate is treated as failed.

## 12. Efficiency Claim Protocol

"Lightweight" is part of the core claim (doc 01), so efficiency is a
first-class result, not an appendix table.

- Separate cost axes: offline precompute, per-request online latency, memory,
  online LLM calls (target: zero — prove it, do not assert it), and training
  cost.
- State hardware, batch size, sequence lengths, and caching assumptions for
  every latency number.
- Include a quality-vs-online-cost comparison (Pareto view) that contains at
  least one heavyweight point (e.g., an online LLM reranker) when feasible, so
  "lightweight" is demonstrated relative to the alternative it displaces.
- Comparisons at matched latency are stronger than absolute quality wins and
  should be preferred when claiming a better tradeoff.
- Reusable-state claims (cached user/item representations) must report cache
  hit assumptions and staleness policy.

## 13. Behavior Preservation Gate

Doc 01 requires that adding semantic control must not destroy no-query
sequence quality. Make this a quantitative gate, not a hope:

- Before finals, fix a non-degradation threshold epsilon on the no-query dev
  primary metric, relative to the strongest sequence specialist trained with
  the same budget.
- If the unified model stays within epsilon, the unified/behavior-preservation
  claim is allowed.
- If it does not, the claim shrinks to "query-surface gain at a stated
  sequence cost", and the cost is reported in the main table — not hidden in
  an appendix or explained away.

## 14. Complexity Budget

Every module must pay rent.

- Every named module has an ablation. A module whose removal changes the
  primary metric by less than the minimal claimable effect (11) is removed
  from the method (it may move to an appendix as a negative result).
- Method-section budget: one primitive plus at most three named components.
  Exceeding the budget triggers a re-check of Section 4 — the design is
  probably an ensemble again.
- Fallback rules, special-case heuristics, and per-surface thresholds count
  against the budget.

## 15. Anticipated-Attack Constraint

Before writing, list the strongest reviewer objections and the experiment that
answers each. At minimum:

| Objection | Required answer |
|---|---|
| "This is encoder X + backbone Y stitched together" | primitive statement (1) + unified controls (6.3) |
| "Gains come from extra parameters/compute" | parameter/compute-matched backbone control (6.1) |
| "Queries are synthetic or repository-authored" | dataset provenance table; such data excluded from primary claims (doc 01 non-goals) |
| "Baselines are undertuned" | tuning budget log (9) |
| "Metrics are sampled and unreliable" | candidate protocol statement (10) |
| "No-query quality was sacrificed" | behavior preservation gate result (13) |
| "The explanation module is decoration" | faithfulness controls (6.4) or the module is dropped |

If an objection has no answering experiment, either add the experiment or
shrink the claim before writing. Do not leave the answer to the rebuttal.

## 16. Paper Readiness Checklist

A design is paper-ready only if all are true:

- The core insight is one sentence.
- The insight implies one architecture primitive.
- The primitive has a cheap falsification test.
- Query and no-query records use the same model contract.
- Dataset ID is not used for branching.
- Strong single-channel controls are measured.
- Static mixture is measured if any dynamic/reasoning gain is claimed.
- Same-encoder controls are measured for query claims.
- Wrong-query and no-query controls are measured for query claims.
- Explanation faithfulness is measured if explanations are claimed.
- Latency includes online work.
- Negative or incomplete results are represented as claim shrinkage.
- Baselines are tuned with an equal, logged search budget.
- All methods rank identical candidate sets under one declared split protocol.
- Seeds and variability are reported; the minimal claimable effect was
  declared before finals.
- The behavior-preservation threshold epsilon was fixed before finals and the
  gate result is reported.
- Every named module has a rent-paying ablation; the method fits the
  complexity budget.
- Every anticipated attack in Section 15 has an answering experiment.

If any item fails, the work may still be valuable, but the paper claim must be
narrowed before writing.
