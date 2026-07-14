# Experiment and dataset contract

Status: active contract for doc 34. The former C0–C5 plan is archived.

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
cohort. Keep click and purchase/graded labels distinct. Development records
are label-free by construction; training/scoring code must not open
`qrels_dev.jsonl` or `qrels_test.jsonl`.

## Active phases

The scientific phases are E0–E8 in doc 34:

1. E0 source, collision, eligibility, and power admission;
2. E1 strong query-candidate bases;
3. E2 ordinary full-token family adequacy;
4. E3 label-free candidate-relative response;
5. E4 direction and user-specificity;
6. E5 train-only signal witness;
7. E6 simple alternative explanations;
8. E7 independent data/family replication;
9. E8 Failure Card or explicit terminal conclusion.

No later phase can be used to rescue a failed earlier premise.

## Metrics

The shared evaluator remains the sole source of ranking metrics. The new
direction additionally requires request-level candidate-relative score deltas,
incremental NDCG, signed delta alignment, active-response precision, and
true-over-matched-wrong advantage. Freeze one primary utility endpoint and one
primary direction endpoint before confirmation; all other measures are
secondary diagnostics.
