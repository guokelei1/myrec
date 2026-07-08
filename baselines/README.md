# Baselines

Tracked working copies for upstream-code baselines and notes for all PPS
baselines (B0a-B8, defined in
[doc/11_experiment_and_dataset_plan.md](../doc/11_experiment_and_dataset_plan.md)).

## Layout

| Dir | Baseline | Code source |
|---|---|---|
| `kuaisearch_official/` | B5: DIN / DCNv2 | KuaiSearch official repo |
| `recbole/` | B4: SASRec / BERT4Rec | RecBole framework |
| `pps_classic/` | B6: HEM / ZAM / TEM | PPS classic paper code |

Self-implemented baselines (B0a Popularity, B0b Recent-behavior, B1 BM25,
B7 Static mixture) live as code under `src/myrec/baselines/`.

## For each upstream baseline

Track a short README or manifest with:

- upstream URL;
- upstream commit hash;
- license notes;
- setup commands;
- expected input/output format;
- any local patch summary.

Register each baseline's boundary card (official code / adapter-only /
structural change / zero-shot) in
`experiments/pps_baseline_cards.md`.

## What not to put here

Generated data, checkpoints, downloaded LLM weights, caches, raw logs, and
score dumps. Use `data/`, `models/`, `runs/`, or `artifacts/` for those.
