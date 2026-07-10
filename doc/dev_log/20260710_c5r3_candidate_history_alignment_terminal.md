# 2026-07-10 C5-R3 Candidate-History Alignment Terminal

## Purpose

C5-R2 repaired temporal asymmetry but did not establish same-query identity
specificity. The remaining question was whether the valid bundled-history gain
could support a narrower, candidate-aligned mechanism. The terminal requirement
was to finish a finite, pre-outcome recovery ladder rather than stop after one
control.

## Ordering

1. At 21:26 +08, before any component score or outcome, freeze
   `doc/23_c5r3_candidate_history_alignment_protocol.md` and
   `configs/analysis/c5r3_candidate_history_alignment.yaml`.
2. Freeze a primary multi-granular item+category claim, one coarse-category
   fallback, and benchmark/analysis-only if both fail.
3. Implement and unit-test exact executable B0b decomposition and gate logic.
4. Materialize item/category scores without qrels and mix each with the frozen
   three-seed D2p scores at the existing beta=0.3.
5. Invoke the shared dev evaluator exactly six times.
6. Run all 12 frozen history-present paired bootstrap comparisons.
7. Run the independent finalizer and apply the frozen decision ladder.

The decision ladder was fixed at 21:26. Before materialization, the executable
files received two non-outcome clarifications: the exact formula for the
fallback relative-gain average and the upstream full-B0b run identifier needed
for reconstruction. Their final mtimes are 21:32:30, still before the 21:33:32
materialization manifest and 21:36:03 first evaluator output; neither changed a
threshold, candidate path, or fallback count.

No proposed model was trained. Test records/qrels were not read.

## Commands

```text
python scripts/run_candidate_history_alignment_control.py

python scripts/evaluate_scores.py \
  --run-id 20260710_kuaisearch_c5r3_d2s_{item,category}_only_dev_s{20260708,20260709,20260710}

python scripts/compare_runs.py \
  --request-ids artifacts/analysis/c5r_temporal_symmetric_identity/history_present_request_ids.txt \
  --samples 10000 --seed 20260708 ...

python scripts/finalize_candidate_history_alignment.py
```

## Label-free Integrity

- Requests: 12,229 total; 8,119 history-present; 4,110 history-absent.
- Candidate rows: 575,609.
- `item_component + category_component` max absolute error versus both the
  public scorer and actual upstream B0b scores: `7.105427357601002e-15`.
- Tolerance: `1e-12`; violations: 0.
- Item component nonzero: 3,442 requests / 9,577 candidates.
- Category component nonzero: 5,974 requests / 253,404 candidates.
- Candidate manifest hash:
  `94eb667000e0d0f389d0a2a4d4730683b71c129043edbfcf627590376e9c123e`.
- Materialization/scoring qrels read: false. Test read: false.

## Paired Results on 8,119 History-present Requests

| Comparison | 20260708 | 20260709 | 20260710 |
|---|---:|---:|---:|
| Item only − D2p | +0.032040 | +0.032140 | +0.032635 |
| Category only − D2p | +0.000594 | +0.000532 | -0.000028 |
| Full D2s − item only | -0.005383 | -0.005210 | -0.006336 |
| Full D2s − category only | +0.026063 | +0.026398 | +0.026327 |

Item-only and full-minus-category are significant in all seeds. Category-only
is significant in no seed. Full D2s is significantly worse than item-only in
all seeds. Category-only mean relative gain over D2p is 0.1148%, below the
fallback's frozen 2% minimum.

Overall three-seed means:

- D2p: 0.3239500729;
- full D2s: 0.3416289845;
- item-only: 0.3453755427;
- category-only: 0.3241930653.

## Decision

- Primary multi-granular alignment: **failed**.
- Sole coarse-semantic fallback: **failed**.
- Integrity: **passed**.
- Frozen outcome: **TERMINAL_FAIL / benchmark-analysis-only**.
- Architecture authorization: **none**.

The supported diagnostic insight is that the tested history gain is
concentrated in exact repeat-item memory. Category affinity does not establish
semantic history use and weakens the item-only ranker under the frozen mixture.
Item-only becomes the current static benchmark waterline; it is not promoted
post hoc into a proposed-system primitive.

## Audit Artifacts

- `reports/pps_c5r3_candidate_history_alignment.{json,md}`
- `reports/pps_c5_insight_audit.json`
- `reports/pps_architecture_readiness.md`
- `reports/pps_intro_motivation_dev_eval_reconciliation.json`
