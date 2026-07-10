# 23 - C5-R3 Candidate-History Alignment Motivation Protocol

Status: locked before generating or evaluating any C5-R3 component score on
2026-07-10 21:26 +08:00.

This protocol is the finite motivation-recovery path after C5-R2 failed its
same-query identity requirement. It does not retry user-identity donor designs
and does not reinterpret that failure. Instead it tests a narrower mechanism
already present in the surviving static result: whether correct rolling history
helps because candidates can align with past events at fine item and coarse
taxonomy granularity.

The executable configuration is
`configs/analysis/c5r3_candidate_history_alignment.yaml`. Its SHA256 is recorded
in every generated run and in the final report.

## 1. Surviving Evidence and Candidate Insight

The following facts survive prior audits:

- D2p is a strong non-personalized query/text/popularity base;
- D2s adds the frozen candidate-specific B0b history score and is stronger;
- history absence gives exact D2p ranking/metric fallback;
- unconditioned mean-history and query-attentive D1 residuals do not improve
  stably;
- user-identity specificity is not established and is not used below.

The primary insight candidate is:

```text
Observation: inside a query-conditioned candidate pool, useful behavioral
             evidence is candidate-aligned at multiple granularities: exact
             item memory and coarse taxonomy affinity each add nonredundant
             signal beyond a strong query-candidate base.

Architecture consequence: use one query-anchored candidate-history evidence
             matching residual. Each candidate queries the same masked history
             memory at fine and semantic granularity; this is one matching
             primitive, not a router over fixed scorers.

Falsification: removing either item alignment or category alignment from the
             frozen D2s history channel must hurt, while each retained alignment
             must beat the no-history D2p base on history-present requests.
```

This claim concerns alignment inside the observed correct-history bundle. It
does not claim a randomized user-identity effect, established query-to-event
attention, or deployed causal personalization.

## 2. Frozen Component Decomposition

C5-R3 follows the executable B0b implementation exactly. For every history
event, let

```text
w = 1/sqrt(reverse_position) * (1.5 if purchase else 1.0)
item_component(candidate) = sum 3.0*w for exact item_id matches
category_component(candidate) = sum w * deepest-exclusive category match
```

The deepest-exclusive category weight is 1.0 for level 3, else 0.5 for level
2, else 0.2 for level 1, else zero. It is deliberately not the additive prose
interpretation in the old B0b config. A score audit must prove, candidate by
candidate, that

```text
frozen_B0b = item_component + category_component
```

before any gate result is accepted.

The two ablations are:

- **item-only D2s**: frozen D2p plus the item component;
- **category-only D2s**: frozen D2p plus the category component.

Both reuse the existing D2s `beta=0.3`, request-level z-score semantics,
candidate manifest, D2p seeds, and true rolling history. No new weight is
selected. This is a removal ablation, not a baseline tuning sweep.

## 3. Evaluation

- Seeds: 20260708, 20260709, 20260710.
- Primary population: the frozen 8,119 history-present request IDs.
- Metric: NDCG@10 from the shared evaluator.
- Statistics: 10,000-sample paired bootstrap, seed 20260708.
- Required comparisons per seed:
  - item-only D2s versus D2p;
  - category-only D2s versus D2p;
  - full D2s versus item-only D2s;
  - full D2s versus category-only D2s.
- On 4,110 history-absent requests, both ablations must remain rank/metric
  equivalent to seed-matched D2p.

Materialization/scoring may read standardized train-free dev records and frozen
upstream score files, but never qrels. Only the shared evaluator reads dev
qrels. No model training, calibration, test record, or test qrels access is
allowed.

## 4. Frozen Finite Decision Ladder

### Primary: multi-granular alignment

The primary insight passes only if:

1. item-only versus D2p has positive mean delta and CI lower bound above zero
   in at least two of three seeds;
2. category-only versus D2p has positive mean delta and CI lower bound above
   zero in at least two of three seeds;
3. full D2s versus item-only has positive mean delta and CI lower bound above
   zero in at least two of three seeds;
4. full D2s versus category-only has positive mean delta and CI lower bound
   above zero in at least two of three seeds;
5. decomposition, candidate coverage, label isolation, and no-history fallback
   all pass.

A pass authorizes the multi-granular candidate-history evidence-matching
primitive for proposed-system design, with D2s retained as the numeric
waterline.

### Predeclared fallback: coarse semantic alignment only

If the primary fails, exactly one fallback is allowed. It passes only if:

1. category-only versus D2p has CI lower bound above zero in all three seeds;
2. its three-seed mean relative improvement over D2p on history-present
   requests is at least 2%. The frozen calculation is the arithmetic mean of
   the three per-seed ratios
   `(mean_ndcg_category_only - mean_ndcg_d2p) / mean_ndcg_d2p`, with both
   means computed on exactly the 8,119 frozen request IDs;
3. full D2s is not significantly worse than category-only in any seed and its
   mean delta is nonnegative;
4. all integrity and no-history checks pass.

A fallback pass authorizes the narrower primitive **query-anchored coarse
candidate-history semantic matching**. Exact repeat-item memory may remain an
auxiliary feature but is not the paper insight.

If both paths fail, exact-item recurrence alone is not sufficient to authorize
a paper system. Motivation terminates as benchmark/analysis-only. No new donor,
threshold, component, or dev-driven fallback may be introduced after seeing
C5-R3 outcomes.
