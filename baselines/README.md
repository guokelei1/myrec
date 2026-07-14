# Baselines

Tracked working copies for upstream methods that may serve the active control
roles in [doc/13](../doc/13_baseline_implementation_plan.md). Historical B0–B9
identifiers are provenance labels, not the active experiment structure.

## Layout

| Dir | Baseline | Code source |
|---|---|---|
| `kuaisearch_official/` | B5: DIN / DCNv2 | KuaiSearch official repo |
| `recbole/` | B4: SASRec / BERT4Rec | RecBole framework |
| `pps_classic/` | B6: HEM / ZAM / TEM | PPS classic paper code |

Self-implemented source-order/popularity/BM25/static controls live under
`src/myrec/baselines/`. E-QC/E-FULL and D-QC/D-FULL will be added only after E0
admits a standardized data version; they are ordinary controls, not proposed
systems.

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
