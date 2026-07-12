# Pre-outcome execution abort

After the v1 proposal and G0 execution locks, the first real-data engineering
smoke invocation stopped at the model call because the runner forwarded the
registered tensor name `query_token_ids` while the model API requires
`query_ids`.  Python raised `TypeError` before the model forward returned.

The invocation produced no score, loss, optimizer step, checkpoint, formal
attempt or train-gate report.  It did not open internal-A, delayed-B, escrow,
dev or test labels.  The v1 proposal and execution locks remain immutable audit
records.  The superseding v2 lock admits only the argument mapping, a defensive
copy of the frozen embedding array, v2 lock paths/identifier, and regression
tests/documentation for this pre-outcome execution error.  The v1 G0 outputs
remain valid because neither token materialization nor any registered input was
changed.
