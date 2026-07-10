# PPS Batch 2b Completion Audit

Date: 2026-07-09

Status: Batch 2b official-baseline work is complete for the current decision
scope. B4o and B5o produced formal KuaiSearch dev results; B5o remains
explicitly `official-code, proxy-aligned (last-time 10% split)`. B6o is
permanently downgraded after failed external alignment. No B6o KuaiSearch dev
evaluation has been produced.

## Current Decisions

| Baseline | Decision | Consequence |
|---|---|---|
| B4o RecBole SASRec | Complete formal dev baseline | Keep in Batch 2b results |
| B5o KuaiSearch DIN/DCNv2 | Complete formal dev baseline under proxy-aligned identity | Keep in Batch 2b results with caveat |
| B6o HEM official | Permanently downgraded | No issue post, no rerun, no KuaiSearch adapter |

## Checklist Audit

| Requirement | Current evidence | Status |
|---|---|---|
| Step 0 budget amendment before Batch 2b dev evals | `reports/pps_batch2b_budget_amendment.md`; commit `acb3f39` | Done |
| Retire old B4/B5/B6 placeholder cards | `experiments/pps_baseline_cards.md` marks B4/B5/B6 retired placeholders | Done |
| Unified train-only interactions export and hash | `src/myrec/data/batch2b_interactions.py`; `scripts/export_batch2b_interactions.py`; `reports/pps_batch2b_interactions_train_manifest.json`; tests | Done |
| B4o official RecBole environment sanity | `reports/b4o_env_sanity.md` | Done |
| B4o data/scoring adapter and official dev run | `src/myrec/baselines/recbole_adapter.py`; `scripts/run_b4o_recbole.py`; `reports/pps_batch2b_b4o_summary.md`; result row in `experiments/pps_results.md` | Done |
| B4o frozen seeds, determinism, comparisons | `reports/b4o_determinism_check.json`; comparison reports vs Random/B0b/B7-bge | Done |
| B6o official HEM external alignment before KuaiSearch adapter | `reports/b6o_official_alignment.md`; `reports/b6o_hem_official_eval*.json` | Failed; permanently downgraded |
| B6o limited reconnaissance | `reports/b6o_official_alignment.md`; external source list | Done; no deterministic bug found |
| B5o official code/environment smoke | `reports/b5o_official_alignment.md`; ignored smoke artifacts | Done |
| B5o official-format materializer repair | `src/myrec/baselines/kuaisearch_materializer.py`; `tests/test_kuaisearch_materializer.py` | Done |
| B5o smoke AUC direction check | `reports/b5o_smoke_auc_direction_check.md` | Done; no reversal found |
| B5o full proxy DNN + DCNv2 Stage A | `reports/b5o_official_alignment.md`; `artifacts/batch2b/b5o_proxy_lasttime_full/` | Done; proxy-aligned |
| B5o Stage B KuaiSearch adapter/dev run | `reports/b5o_official_alignment.md`; `reports/b5o_protocol_diff.md`; `experiments/pps_results.md` | Done; proxy-aligned formal dev result |
| B5o determinism and comparisons | `reports/b5o_determinism_check.json`; comparison reports vs Random/B0b/B7-bge | Done |
| M3 reissue after Batch 2b | Should include qualified new methods only after final candidate decision | Deferred |

## Evidence Summary

Per doc/07 Section 11, paper-facing trainable results use the frozen-seed mean
and variability below. Highest-seed values are retained only for run-level
traceability and conservative paired comparisons.

B4o produced a protocol-valid formal KuaiSearch dev run:

- best seed NDCG@10: 0.2976
- mean over frozen seeds: 0.2972 +/- 0.0004
- vs Random: +0.0165, CI [0.0113, 0.0217]
- vs B0b: -0.0163, CI [-0.0201, -0.0125]
- vs B7-bge: -0.0329, CI [-0.0382, -0.0276]

B5o Stage A evidence:

- full proxy rows: 17,800,904
- proxy split: `time_index >= 867165`, actual test fraction 0.100003
- target item coverage: 1.0
- query embeddings: `(555553, 512)`
- item embeddings: `(6206709, 512)`
- DNN final LogLoss/AUC: 0.160731 / 0.613133 vs target 0.1588 / 0.6258
- DCNv2 final LogLoss/AUC: 0.162635 / 0.616348 vs target 0.1603 / 0.6239

Both B5o full proxy runs are within +/-10% of the paper Table 7 metric scale,
but the exact paper split is still unverified. The correct claim is
`aligned under proxy last-time split`.

B5o Stage B evidence:

- implementation identity: `official-code, proxy-aligned (last-time 10% split)`
- artifact root: `artifacts/batch2b/b5o_stageb_standardized`
- formal dev evals: 6/16
- best formal run: `20260709_kuaisearch_b5o_dnn_dev_s20260708`
- best NDCG@10: 0.3088
- DNN mean over frozen seeds: 0.3063 +/- 0.0030
- DCNv2 mean over frozen seeds: 0.3054 +/- 0.0002
- determinism: first 1000 dev requests, 42,968/42,968 score rows exact,
  `max_abs_score_diff=0.0`
- vs Random: +0.0277, CI [0.0224, 0.0331]
- vs B0b: -0.0051, CI [-0.0105, 0.0004]
- vs B7-bge: -0.0217, CI [-0.0272, -0.0162]

B6o is not eligible for KuaiSearch formal dev evaluation:

- best HEM MAP@100: 0.0759 vs target about 0.124
- best HEM NDCG@10: 0.0932 vs target about 0.153
- five public external sources were checked
- no original `query_split/` or checkpoint was found
- no deterministic reconstruction bug was found

## Remaining Work

- Reissue M3 only if the next phase needs the oracle candidate set to include
  B5o. The current Batch 2b completion decision does not require a rerun.
- Do not run B6o again or post the upstream issue draft.
