# Experiments

Tracked experiment plans, config templates, and concise run manifests.

Raw run directories belong under `runs/` and are ignored. Promote only
the small information needed to reproduce or interpret an important run:

- run ID;
- command;
- config path;
- git commit;
- dataset version or manifest hash;
- checkpoint/model reference;
- metric summary;
- decision made from the run.

## Baseline cards

Register each baseline's boundary card (official code / adapter-only /
structural change / zero-shot) in `experiments/pps_baseline_cards.md`.

## Results registry

`experiments/pps_results.md` is the single place where baseline/method
numbers are registered. Numbers must be copied verbatim from evaluator
`metrics.json`; significance columns come from the shared compare script
(doc 11 §1.4). Test rows are filled once, after configs are frozen.
