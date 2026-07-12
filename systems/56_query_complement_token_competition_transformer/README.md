# C56 — Query-complement token competition Transformer

C56 tests one architecture-level hypothesis after C55 closed residual-loss
changes on frozen pooled states: history must alter candidate representations
*before token pooling*, and only along content not already explained by the
query/strong base.

The proposed operator uses frozen contextual LM tokens, a trainable shared
token Transformer, and a candidate-set Transformer.  It has no dataset,
category, query-type, or rank branch.  Empty history and query-missing inputs
are structural no-ops; exact-recurrence requests retain the registered
item-only anchor.

This first gate is an exposed fit-internal signal/foundation test.  It cannot
read C26 internal-A/delayed-B/escrow, dev/test, or any qrels.  A positive gate
would authorize a fresh dual-domain proposal and a deeper novelty review; it
would not itself establish the final proposed system.

Tracked source, tests, configs, and concise notes live here.  Generated token
states, checkpoints, logs, and score rows belong under `artifacts/`, `models/`,
and `runs/`.
