# 20260708 C0 KuaiSearch Lite Audit

## Command

```bash
python scripts/download_kuaisearch.py \
  --config configs/datasets/kuaisearch_lite.json \
  --raw-dir data/raw/kuaisearch \
  --connections 4

python scripts/audit_kuaisearch_c0.py \
  --raw-dir data/raw/kuaisearch \
  --report-path reports/pps_c0_data_audit.json \
  --sample-dir data/interim/kuaisearch/v0_lite/time_window_seed20260708

python scripts/check_kuaisearch_history_leakage.py \
  --raw-dir data/raw/kuaisearch \
  --report-path reports/pps_c0_history_leakage_check.json \
  --c0-report-path reports/pps_c0_data_audit.json

python scripts/prepare_kuaisearch_standardized.py \
  --raw-dir data/raw/kuaisearch \
  --window-requests-path data/interim/kuaisearch/v0_lite/time_window_seed20260708/requests.jsonl \
  --output-dir data/standardized/kuaisearch/v0_lite \
  --c0-report-path reports/pps_c0_data_audit.json
```

## Evidence

- Raw Lite files were downloaded from HF `benchen4395/KuaiSearch` into
  `data/raw/kuaisearch/`.
- C0 report: `reports/pps_c0_data_audit.json`.
- Deterministic time-window request manifest:
  `data/interim/kuaisearch/v0_lite/time_window_seed20260708/manifest.json`.

Key C0 checks:

- Candidate aggregation: passed. Ranking has 17,800,904 rows grouped into
  555,553 requests; median candidate count is 13; 1000 sampled recall
  requests matched ranking aggregation exactly.
- Label sanity: passed. CTR is 3.824%; purchase row rate is 0.286%;
  click and purchase labels are both nonzero.
- Text join coverage: passed. Catalog, candidate, and history text coverage
  are all 100% against `items_lite`.
- Time field: passed. `time_index` spans 1 to 965,265 and supports a
  250,000-request continuous window with 200k/25k/25k train/dev/test counts.
- Raw `recently_*` history future leakage: failed by log-internal
  cross-reference. On 1000 sampled recall requests at or after the global
  median `time_index`, the raw rank history had 35,935 history items:
  3,332 past-supported, 95 same-time-only, 135 future-only, and 32,373
  unobserved. Future-only over observed was 3.79%, above the 0.1% limit;
  past-supported over total history was 9.27%, below the 20% power floor.
- Fallback history construction: passed. Standardized records reject raw
  rank `recently_clicked_item_ids` / `recently_purchased_item_ids` and rebuild
  history from recall-window events for the same user with
  `event_time < request_time`.

Standardized KuaiSearch Lite v0 outputs:

- Output directory: `data/standardized/kuaisearch/v0_lite/`.
- Records after filtering: train 163,717; dev 12,229; test 12,224.
- Removed for candidate count <5: train 36,283; dev 4,304; test 4,694.
- Removed for no clicked positive in dev/test: dev 8,467; test 8,082.
- History source: recall prior events only, max length 50, ascending time.
  History-present rate is 59.69%; median length is 1; mean length is 5.44.
- Item join coverage: 2,966,962 / 2,966,962 needed item IDs loaded; missing 0.
- Record text coverage: 100%.
- Candidate manifest SHA-256:
  `94eb667000e0d0f389d0a2a4d4730683b71c129043edbfcf627590376e9c123e`.

## Conclusion

C0 overall status is passed after applying the registered fallback route.
The caveat for all downstream reports is that raw `recently_*` fields are not
used. History leakage is prevented by construction from log-internal
cross-reference events inside the selected recall window, not by an official
per-history timestamp guarantee. Interactions outside the observed recall
window remain unobserved and cannot be falsified.
