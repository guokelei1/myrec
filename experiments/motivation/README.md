# Current Motivation mechanism analysis

This directory is the canonical home for the active Motivation mechanism
analysis. Read
[`mechanism_analysis_plan.md`](mechanism_analysis_plan.md) before adding a
probe, diagnostic control, seed, slice, or evaluator endpoint.

The concrete recipes are indexed in
[`configs/methods/README.md`](../../configs/methods/README.md); the separate W0
witness config is indexed in
[`configs/baselines/README.md`](../../configs/baselines/README.md).

`protocol.yaml` is the immutable first-round V1.2 protocol. It keeps historical
authority strings and SHA256 so existing score bundles remain auditable; those
strings are evidence metadata, not active instructions. The
`experiments/motivation_v1_2` path remains only as a runtime compatibility alias
used by frozen configs and evaluator code. It is not a second active plan.

The completed first-round `plan.md` and `execution_prompt_zh.md` were deleted.
The frozen observations are preserved in `doc/motivation.md`,
`experiments/pps_results.md`, and `reports/motivation_current_summary.json`.
