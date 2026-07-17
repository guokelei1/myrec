# Experiment and dataset contract

Status: supporting technical contract for unified records, split isolation,
labels, and metrics. Current execution order comes from
`experiments/motivation/mechanism_analysis_plan.md`.

## Unified records

All methods consume the same standardized records. Training records contain
labels; dev/test records contain no label fields, with qrels stored separately
for the shared evaluator only.

```json
{
  "request_id": "…",
  "user_id": "…",
  "session_id": "…",
  "ts": 0,
  "query": "…",
  "history": [{"item_id":"…", "title":"…", "event":"click", "ts":0}],
  "candidates": [{"item_id":"…", "title":"…", "brand":"…", "cat":[]}],
  "masks": {"history_present": true, "text_coverage": 1.0}
}
```

The candidate order and `candidate_manifest.json` are frozen per standardized
version. A separate `request_manifest.json` hashes request ID, query and
candidate identity without history so true/null/wrong scoring can prove that
only the history assignment changed. History events must be strictly before
the request. Dataset-specific adapters may map fields into this object but may
not reopen raw data during scoring.

## Split and labels

Use a time-ordered split with session containment and a separate confirmation
cohort. Keep click and purchase/graded labels distinct. Development and
confirmation records are label-free by construction; training/scoring code must
not open `qrels_dev.jsonl`, `qrels_confirmation.jsonl`, or `qrels_test.jsonl`.
Confirmation labels may be opened only once by the shared evaluator after the
frozen score bundle passes its pre-label audit.

## Metrics

The shared evaluator remains the sole source of ranking metrics. The new
direction additionally requires request-level candidate-relative score deltas,
incremental NDCG, signed delta alignment, active-response precision, and
true-over-matched-wrong advantage. Freeze one primary utility endpoint and one
primary direction endpoint before confirmation; all other measures are
secondary diagnostics.

Every registered endpoint reports both the legacy all-request observed-label
aggregate and the conditional-positive estimand. Requests without a positive
gain are counted and reported separately; they cannot be interpreted as target-
nonrepeat ranking failures. Evaluator-side target-aware surfaces partition the
positive-eligible population into target recurrence, target-nonrepeat with
other-candidate overlap, target-nonrepeat with no candidate overlap, and
target-nonrepeat with no history. Label-free candidate overlap remains a
separate diagnostic.
