# Scripts

Runnable command-line entry points for the full PPS workflow:

- dataset download and audit;
- preprocessing and standardized record export;
- baseline training / scoring;
- R0 observability, strong-baseline training/scoring, and failure discovery;
- future proposed-system training/scoring only after doc/31 authorization;
- evaluation (single shared evaluator for all methods);
- motivation experiment drivers (M1-M6);
- checkpoint report generation.

Current read-only evidence audits include:

- `audit_m3_ties.py` - tie-aware interpretation of frozen M3 choices;
- `audit_m3_m4_random_canary.py` - tests whether Random reproduces M3/M4;
- `check_b9_determinism.py` - exact score reproducibility for B9.

Current authorization is R0 only. C01--C80 candidate scripts are historical;
running them does not reopen a candidate. New architecture entry points require
a passed Failure Card and Hxx registration under `doc/31`.

Current motivation-repair entry points:

- `run_candidate_history_alignment_control.py` - exactly decomposes B0b into
  item/category components, audits reconstruction, and materializes six locked
  C5-R3 score runs without qrels;
- `finalize_candidate_history_alignment.py` - audits comparisons, hashes,
  no-history rank/metric equivalence, dev logging, and the finite C5-R3
  primary/fallback terminal decision;
- `run_temporal_identity_control.py` - materializes freshness-matched
  train/earlier-dev wrong-history B0b/D2s scores without qrels;
- `finalize_temporal_identity_control.py` - independently audits assignments,
  no-history fallback, comparisons, and the frozen C5-R2 decision (historical
  predecessor to C5-R3);

- `run_history_identity_controls.py` - materializes train-only matched
  wrong-user B0b/B7 scores without qrels (historical, temporally confounded);
- `summarize_history_identity_controls.py` - runs the locked subset comparisons
  and writes the C3-R report;
- `finalize_c5r_insight_gate.py` - historical C5-R promotion; superseded by
  the C5-R2 temporal control.

Current Amazon-C4 secondary-track entry points:

- `prepare_amazon_c4_standardized.py` - joins the official Amazon-C4 query
  table with the temporal history release, constructs a frozen sampled-1M
  BM25 candidate pool, joins Amazon Reviews 2023 metadata, and writes the
  unified label-isolated JSONL interface;
- `audit_amazon_c4_protocol.py` - runs the C1-style candidate/split/history,
  train-blind, qrels, metric, and candidate-hash checks without authorizing a
  model-side dev evaluation.

Current supervised strengthening entry points:

- `materialize_supervised_diagnostics.py`, `train_supervised_diagnostic.py`, and
  `score_supervised_diagnostic.py` - D1 frozen-embedding base/residual controls;
- `materialize_finetuned_query_tokens.py`, `train_finetuned_query_tower.py`, and
  `score_finetuned_query_tower.py` - D2 fine-tuned text/non-personalized controls;
- `calibrate_d2h_static_history.py` and `score_d2h_static_history.py` - train-only
  D2h alpha selection and true/wrong static history controls;
- `calibrate_d2s_static_full.py`, `score_d2s_static_full.py`,
  `audit_d2s_static_full_scores.py`, and `summarize_d2s_static_full.py` -
  complete D2p + bundled-history reference and audit (superseded as the numeric
  waterline by C5-R3 item-only);
- `audit_supervised_diagnostic_scores.py`, `audit_finetuned_query_scores.py`, and
  `audit_d2h_static_history_scores.py` - label-free score integrity checks;
- `summarize_supervised_diagnostics.py` and `summarize_d2_controls.py` - curated
  D1/D2/D2h decisions; the current binding static waterline is the C5-R3
  item-only control recorded by the C5-R3 finalizer.

Scripts should be thin wrappers that import from `src/myrec/` and read
config paths from `configs/`. Keep logic in the library, not in scripts.

Naming convention:

```text
<verb>_<object>.py      e.g. download_kuaisearch.py
                         e.g. prepare_standardized_dataset.py
                         e.g. evaluate_scores.py
```
