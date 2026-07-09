# PPS Batch 2b Interim Completion Audit

Date: 2026-07-09

Status: interim audit, not the final Batch 2b decision summary.

This audit records the current evidence against `doc/14_official_baseline_plan.md`
and the follow-up B5o/B6o repair prompt. It intentionally does not replace
`reports/pps_batch2b_decision_summary.md`, because B5o full Stage A alignment is
paused on an explicit split-policy decision.

## Current Blocking Decision

B5o public-file repair reached the point where the official-format materializer
and official loader/trainer smoke pass, but the full Table 7 alignment run needs
an authorized test split policy.

Evidence:

- `doc/baseline_notes/20260709_b5o_stage_a_split_decision.md`
- `reports/b5o_official_alignment.md`
- `reports/b5o_protocol_diff.md`

Decision needed:

| Option | Meaning | Consequence |
|---|---|---|
| A | Use last-time 10% by `time_index` as a proxy test split | Can run bounded DNN + DCNv2 Stage A, but result is `last-time proxy`, not exact official reproduction |
| B | Wait for upstream-confirmed last-day boundary | Cleanest official-reproduction path, but blocks B5o full Stage A |
| C | Use random row split | Technically easy, but conflicts with the last-day split story and is not recommended |

Until this is decided, do not start the full DNN/DCNv2/DIN Table 7 alignment
training. This follows the repair prompt instruction: if last-day split
construction is uncertain, record candidate options and stop for confirmation.

## Checklist Audit

| Requirement | Current evidence | Status |
|---|---|---|
| Step 0 budget amendment before Batch 2b dev evals | `reports/pps_batch2b_budget_amendment.md`; commit `acb3f39` | Done |
| Retire old B4/B5/B6 placeholder cards | `experiments/pps_baseline_cards.md` marks B4/B5/B6 retired placeholders | Done |
| Unified train-only interactions export and hash | `src/myrec/data/batch2b_interactions.py`; `scripts/export_batch2b_interactions.py`; `reports/pps_batch2b_interactions_train_manifest.json`; tests | Done |
| B4o official RecBole environment sanity | `reports/b4o_env_sanity.md` | Done |
| B4o data/scoring adapter and official dev run | `src/myrec/baselines/recbole_adapter.py`; `scripts/run_b4o_recbole.py`; `reports/pps_batch2b_b4o_summary.md`; result row in `experiments/pps_results.md` | Done |
| B4o frozen seeds, determinism, comparisons | `reports/b4o_determinism_check.json`; `reports/compare_b4o_h128_s20260708_vs_random.json`; `reports/compare_b4o_h128_s20260708_vs_b0b.json`; `reports/compare_b4o_h128_s20260708_vs_b7_bge.json` | Done |
| B6o official HEM external alignment before KuaiSearch adapter | `reports/b6o_official_alignment.md`; `reports/b6o_hem_official_eval*.json` | Failed alignment, correctly downgraded/blocked |
| B6o limited reconnaissance, no training unless deterministic bug found | `reports/b6o_official_alignment.md`; `doc/baseline_notes/20260709_b6o_upstream_issue_draft.md` | Done; no rerun justified |
| B5o official code/environment smoke | `reports/b5o_official_alignment.md`; `artifacts/batch2b/b5o_official_smoke/` ignored evidence | Done |
| B5o official-format materializer repair | `src/myrec/baselines/kuaisearch_materializer.py`; `tests/test_kuaisearch_materializer.py`; `configs/baselines/b5o_kuaisearch_din_dcnv2.yaml` | Done at smoke scale |
| B5o materializer target coverage and official DNN smoke | `artifacts/batch2b/b5o_materializer_smoke/materializer_manifest.json`; `reports/b5o_official_alignment.md` | Done at smoke scale |
| B5o DNN + DCNv2 Table 7 alignment, at least two methods | Requires authorized split policy | Blocked |
| B5o Stage B KuaiSearch adapter/dev run | Only allowed after Stage A alignment succeeds or a new explicit downgrade/adapter decision is made | Not started |
| M3 three-channel oracle after Batch 2b | Should include qualified new methods in channel candidates after B5o status is settled | Deferred |
| Final `reports/pps_batch2b_decision_summary.md` | Depends on settled B5o status and M3 rerun | Deferred |

## Evidence Summary

B4o is the only Batch 2b method that currently produced a protocol-valid formal
KuaiSearch dev run. It is official RecBole SASRec and remains below B0b and
B7-bge:

- best seed NDCG@10: 0.2976
- mean over frozen seeds: 0.2972 +/- 0.0004
- vs Random: +0.0165, CI [0.0113, 0.0217]
- vs B0b: -0.0163, CI [-0.0201, -0.0125]
- vs B7-bge: -0.0329, CI [-0.0382, -0.0276]

B6o is not eligible for KuaiSearch formal dev evaluation from the current
evidence:

- best HEM MAP@100: 0.0759 vs target about 0.124
- best HEM NDCG@10: 0.0932 vs target about 0.153
- public upstream reconnaissance found no original `query_split/` or checkpoint
- no deterministic reconstruction bug was found, so no additional 20-epoch run
  is justified

B5o is improved but not settled:

- official-format materializer smoke passed on 2000 raw ranking rows
- target coverage in smoke: 1.0000
- official BGE embedding process ran
- official DNN 1-epoch smoke ran through train/valid/test
- full Table 7 alignment is paused because public `rank_lite/train.jsonl` has
  only `split=train` and no verified paper test boundary

## Work Not To Do Before Decision

- Do not run B5o full DNN/DCNv2/DIN alignment with an implicit split.
- Do not run B5o Stage B on PPS standardized dev.
- Do not write the final Batch 2b decision summary.
- Do not rerun M3 as a final Batch 2b artifact.

The next authorized action is to choose the B5o split policy, preferably Option
A if a bounded proxy attempt is acceptable.
