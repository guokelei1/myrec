# Scripts

Runnable command-line entry points for the full PPS workflow:

- dataset download and audit;
- preprocessing and standardized record export;
- baseline training / scoring;
- proposed-system training / scoring;
- evaluation (single shared evaluator for all methods);
- motivation experiment drivers (M1-M6);
- checkpoint report generation.

Scripts should be thin wrappers that import from `src/myrec/` and read
config paths from `configs/`. Keep logic in the library, not in scripts.

Naming convention:

```text
<verb>_<object>.py      e.g. download_kuaisearch.py
                         e.g. prepare_standardized_dataset.py
                         e.g. evaluate_scores.py
```
