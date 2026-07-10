# B9 ZAM/TEM Neighbor Baseline Summary

Status: **provisional_human_review_pending**.

Identity: `official-code, adapter to KuaiSearch interface, not externally aligned`.

## Results

| Model | Internal validity | 3-seed mean +/- std | Highest observed seed | Highest seed vs B7-bge |
|---|---|---:|---:|---|
| ZAM | provisional; human review pending | 0.2986 +/- 0.0006 | 0.2994 (s20260710) | -0.0311, CI [-0.0365, -0.0256] |
| TEM | provisional; human review pending | 0.2940 +/- 0.0009 | 0.2948 (s20260710) | -0.0358, CI [-0.0412, -0.0303] |

All paper metrics above are copied from shared evaluator `metrics.json` files; all CIs are from `scripts/compare_runs.py`.
A method is described as above a reference only when the paired-bootstrap result is significant and the relative gain is at least 2%, as frozen in doc/16.

## Internal Validity

- ZAM: `{'three_seeds': True, 'all_seeds_significant_above_random': True, 'determinism_exact_first_1000': True, 'loss_decreased_all_seeds': True, 'top5_review_confirmed': False}`
- TEM: `{'three_seeds': True, 'all_seeds_significant_above_random': True, 'determinism_exact_first_1000': True, 'loss_decreased_all_seeds': True, 'top5_review_confirmed': False}`

The label-free top-5 sheet has a preliminary review, but author confirmation provenance is pending; it retains 5 model-specific flag entries for category-adjacent or off-topic tail items. See `reports/pps_b9_top5_review.md`.

## Execution Recovery

The three ZAM runs resumed from complete checkpoints (epoch 13, epoch 13, epoch 12) after two modern-PyTorch compatibility faults were fixed. Prior epochs were not retrained, and this checkpoint continuation is not counted as a new full training attempt.

The upstream checkpoints preserve model and optimizer state but not RNG state, so resumed trajectories are not bit-identical to hypothetical uninterrupted runs. Checkpoint selection used only the upstream train-only validation; per-run checkpoint hashes and commands are retained in metadata.

## Boundary And Limitation

The adapter uses request-level queries and exact frozen histories, pads only for the upstream fixed-width scorer, then removes all fillers before shared evaluation.

Only 11.71% of unique dev candidates and 22.00% of dev candidate rows occur as clicked train targets. This official item-ID/PV cold-product limitation is retained rather than repaired with post-hoc text pretraining.

Frozen wording branch: `b9_human_review_pending`.
