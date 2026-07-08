# Artifacts

Generated local artifacts. Ignored by Git except for this README.

This is the **staging layer between `runs/` and `reports/`/`paper/`**.
Use it for:

- temporary figures generated from run outputs;
- exported metric tables (before curation);
- packaged predictions for sharing or re-scoring;
- converted data formats (e.g. score dumps reshaped for plotting).

Everything here can be regenerated from `runs/` + scripts. Move only
small final assets into `reports/` or `paper/` after review.

## Boundary

| Directory | Content | Tracked? |
|---|---|---|
| `runs/` | raw experiment output | no |
| `artifacts/` | this dir: post-processed intermediates | no |
| `reports/` | curated final results | yes (small) |
