# Counterfactual score-bundle contract

Formal history-response analysis uses three independently materialized run
directories produced by one frozen checkpoint:

```text
runs/<true_run_id>/scores.jsonl
runs/<null_run_id>/scores.jsonl
runs/<wrong_run_id>/scores.jsonl
```

Every score row retains the shared fields:

```json
{"request_id":"r", "candidate_item_id":"i", "score":0.0, "method_id":"E-FULL"}
```

Every run's `metadata.json` must include:

```json
{
  "history_condition": "true",
  "checkpoint_id": "immutable checkpoint reference",
  "dataset_id": "kuaisearch",
  "dataset_version": "full_e0_v1",
  "split": "dev",
  "candidate_manifest_sha256": "...",
  "request_manifest_sha256": "hash of request/query/candidate identity without history",
  "history_assignment_sha256": "condition-specific history materialization hash",
  "scoring_signature": {
    "serialization_version": "...",
    "max_length": 0,
    "history_budget": 0,
    "candidate_scoring_head": "...",
    "dtype": "..."
  }
}
```

Only `history_condition` and `history_assignment_sha256` may differ across the
bundle. The shared evaluator rejects a bundle if checkpoint, dataset, split,
request/query/candidate identity, candidate hash, or scoring signature differs.
The scoring programs must not read qrels; `scripts/analyze_history_response.py`
is the qrels-reading shared boundary and appends every development call to
`reports/dev_eval_log.jsonl`.

Legacy score files without this metadata may be used only for the E-1
instrumentation pilot after a reviewed provenance shim. They cannot enter a
Failure Card.

The standardized version therefore contains both manifests:

- `candidate_manifest.json`: exact per-request candidate order and identity;
- `request_manifest.json`: per-request query/candidate identity with all
  history fields excluded.
