# C47 preselection v1 mechanical abort

The first locked selection command stopped before writing any selection,
feature, score, or label artifact. `records_train_blind.jsonl` contains valid
JSON strings with Unicode line-separator characters. Python
`read_text().splitlines()` treats those characters as line boundaries even
though they are not JSONL record delimiters, so one physical JSON record was
split and `json.loads` raised `JSONDecodeError`.

Containment checks:

- `artifacts/c47_posterior_supported_ridge_transformer/signal_gate_v1/selection.json`
  does not exist;
- no label-bearing path was opened;
- no cohort membership, feature, score, metric, dev/test record, or qrels was
  produced;
- the proposal, formula, lambda, counts, seed, thresholds, and input hashes
  remain unchanged.

The only authorized repair is a v2 reader that iterates physical file lines
with `for line in handle`. The original proposal lock and v1 script remain
immutable. A supplemental lock must bind this note and the v2 files before v2
executes.
