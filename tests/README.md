# Tests

Unit and integration tests. Metric code must have unit tests with
hand-computed assertions (doc 11, C1 checkpoint).

Only tiny fixtures should be committed. Large fixture data belongs under
`data/` and should be regenerated or referenced through a small manifest.

The `.gitignore` allows files under `tests/fixtures/` so small
JSONL/CSV/TSV fixtures can be tracked intentionally.
