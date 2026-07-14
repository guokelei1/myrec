# Tests

Unit and integration tests for the shared data, baseline, evaluator, and
metric contracts. Metric code must have hand-computed assertions (doc 11).

Only tiny fixtures should be committed. Large fixture data belongs under
`data/` and should be regenerated or referenced through a small manifest.

The `.gitignore` allows files under `tests/fixtures/` so small JSONL/CSV/TSV
fixtures can be tracked intentionally. Historical R0 and architecture tests
are archived with their source modules.

The active suite covers ranking metrics, score comparisons, standardized
record/label-isolation contracts, and hand-computed history-response direction
metrics. Any new Full-track adapter must add a tiny source-to-standardized
fixture before it is used on raw data.
