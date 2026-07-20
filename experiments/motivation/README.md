# Motivation evidence and active architecture development

The Motivation mechanism experiments have been inventoried and are no longer an
active expanding queue. The active development authority is
[`candidate_contrast_architecture_plan.md`](candidate_contrast_architecture_plan.md).
Read it before adding model code, a training variant, seed, slice, loss, or
evaluator endpoint.

The completed M0--M4 authority remains
[`mechanism_analysis_plan.md`](mechanism_analysis_plan.md). The Transformer
deep-dive plans and N8--N34 manifests are preserved for provenance; their
partial or deferred jobs must not be resumed automatically.

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
The unified post-Motivation inventory is
[`../../reports/motivation_post_stage_experiment_inventory_zh.md`](../../reports/motivation_post_stage_experiment_inventory_zh.md).
