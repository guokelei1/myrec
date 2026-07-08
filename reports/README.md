# Reports

Curated, paper-ready results. Tracked selectively (small files only).

## Checkpoint audit reports

Each phase gate (C0-C5) produces a JSON audit report here:

```text
reports/pps_c0_data_audit.json
reports/pps_c1_protocol.json
reports/pps_c3_motivation.json
reports/pps_c4_data_final.json
reports/pps_c5_insight.json
```

The gate must pass before advancing to the next phase.

## What goes here vs. elsewhere

| Directory | Content | Tracked? |
|---|---|---|
| `runs/` | raw experiment output (logs, score dumps, raw metrics) | no |
| `artifacts/` | generated intermediate plots, tables, predictions | no |
| `reports/` | curated final results, audit JSONs, paper-ready tables | yes (small) |
| `paper/` | manuscript source and final selected assets | yes |

The `.gitignore` explicitly allows `reports/**/*.csv`, `*.tsv`, `*.json`,
`*.jsonl` so curated tables can be tracked.
