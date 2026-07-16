# Motivation V1 consolidation and cleanup

## Decision

The current concise motivation is now the three-selected-Transformer
recurrence--transfer observation recorded in
`doc/40_transformer_recurrence_transfer_motivation_v1_zh.md` and
`reports/pps_three_transformer_history_surface_audit.json`.

The binding pre-registered confirmation remains the fresh Qwen3 run. TEM and
InstructRec are same-population frozen-output extensions: both reproduce a
significant target-repeat positive control, no established no-overlap
nonrepeat recovery, and a positive repeat-minus-no-overlap contrast. Their
near-zero aggregate true-minus-null values are surface cancellation, not score
invariance.

## Synchronized active material

- `doc/README.md` now points first to Motivation V1;
- docs 10, 35, 36, 38, 39 and the final motivation audit carry the same claim
  and boundary;
- baseline cards distinguish frozen Qwen confirmation from single-seed TEM and
  InstructRec extensions;
- the experiment README, manifest, pipeline state, current status JSON, final
  audit JSON, completion audit, and baseline comparison point to the V1 report;
- the old `TEM_QC` label in the comparison report was corrected to
  `TEM_FULL_NULL`, because it is a same-checkpoint null counterfactual rather
  than an independently trained query-only model;
- pre-repair academic/narrative audits now have explicit historical banners and
  cannot be mistaken for current conclusions.

## Cleanup performed

The cleanup deliberately preserved canonical scores, best checkpoints,
commands, logs, per-request metrics, protocols, and formal negative-result
reports.

Removed as non-reproducible or redundant local state:

- nine interrupted InstructRec full-score directories (`true/null/wrong`,
  initial, batch-4, and batch-16) with only 612--5,253 of 39,269 score rows and
  no metadata;
- two superseded TEM initial training directories that had no metadata and
  never entered the evaluator;
- per-epoch TEM checkpoints from the two complete V2 training directories,
  retaining `model_best.ckpt`, commands, logs, ranklists, scores, and metadata;
- temporary `tmp/history_baseline_audit/` bootstrap/intersection files after
  promotion into the V1 report.

This released approximately 23 GB while preserving every artifact needed to
reproduce or rescore the canonical V1 baselines.

## Final validation

- all 10 retained JSON reports and four active YAML state/card files parse;
- all local links in the 20 retained project Markdown files resolve;
- all 10 active evidence paths named by the manifest, status, and V1 audit
  exist;
- `git diff --check` passes;
- 21 focused adapter, evaluator, metric, and target-surface tests pass.

## Second-pass document/report pruning

After explicit approval to keep only current useful material, the repository
was reduced to the active contracts, Motivation V1, the frozen Qwen protocol,
the three-model audit, and its direct confirmation inputs. Removed material
includes:

- superseded docs 24 and 34--39, the old motivation-audit directory, and old
  exploration logs;
- superseded cross-dataset, repair, representative-architecture, and extension
  protocols/manifests;
- 254 generated exploratory reports and old figures, including 179 previously
  tracked report files;
- the 922-line historical pipeline journal, replaced by a compact current-state
  record.

Raw scores, checkpoints, logs, model/data artifacts, baseline source, reusable
evaluation code, and the direct frozen-confirmation evidence chain were not
deleted. Old explorations can therefore still be reconstructed from local run
state when genuinely needed, but they no longer compete with V1 in the tracked
research narrative.

No proposed architecture or test action is authorized by this consolidation.
