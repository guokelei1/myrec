# PPS Batch 2b Budget Amendment

Date: 2026-07-09

Status: active amendment for Batch 2b execution.

Authorization source: `doc/dev_log/20260708_batch2b_kickoff_prompt.md` and
`doc/14_official_baseline_plan.md` step 0.

## Decision

Batch 2b creates three new method IDs:

| Method ID | Target | Budget |
|---|---|---|
| B4o | RecBole official SASRec/BERT4Rec | 16 KuaiSearch dev evaluations |
| B5o | KuaiSearch official DIN/DCNv2 pipeline | 16 KuaiSearch dev evaluations |
| B6o | HEM/ZAM/TEM official code or externally validated faithful reproduction | 16 KuaiSearch dev evaluations |

The first KuaiSearch dev evaluation for each method must use the official or
paper-default hyperparameters. After a config is frozen, the final 3-seed runs
follow doc 12/doc 13 and are reported as mean and standard deviation.

## Scope Change

The old Batch 2 B4/B5/B6 runs remain registered as evidence, but their status is
changed to:

```text
retired placeholder (superseded by B4o/B5o/B6o)
```

They may be used only in appendix or implementation notes. They must not be used
as main-table evidence for claims about official SASRec, KuaiSearch DIN/DCNv2,
or HEM/ZAM/TEM strength.

## External Alignment Runs

The following checks do not read the project KuaiSearch dev split and do not
count against the 16-dev-evaluation method budgets:

- B4o RecBole environment sanity on RecBole/ml-100k.
- B5o official KuaiSearch reproduction/alignment on official-format data.
- B6o HEM/ZAM/TEM official or Amazon PPS benchmark alignment.

Each external alignment run must still be recorded in the corresponding
baseline card, with environment, data source, command summary, metric, and
pass/fail or downgrade decision.

## Budget Symmetry

The future proposed system keeps the same trainable-method budget: 16
KuaiSearch dev evaluations before freeze, followed by 3 seeds for the frozen
configuration. This preserves the doc 07/doc 13 budget symmetry between
trainable baselines and the proposed method.

## Red Lines

This amendment does not relax any existing protocol rule:

- Training and scoring code must not read `qrels_dev.jsonl` or
  `qrels_test.jsonl`.
- Training data for Batch 2b must be derived only from
  `data/standardized/kuaisearch/v0_lite/records_train.jsonl`.
- Dev/test histories are the frozen `history` fields in each blind record,
  capped at 50 events; user-global sequences are not allowed at inference.
- The candidate manifest remains
  `94eb667000e0d0f389d0a2a4d4730683b71c129043edbfcf627590376e9c123e`.
- The shared evaluator and shared compare script remain the only sources of
  paper metrics and significance intervals.
- A Batch 2b KuaiSearch dev evaluation produced before this amendment is
  committed is invalid.

## Step 0 Audit

Before this amendment, `reports/dev_eval_log.jsonl` contains no B4o, B5o, or
B6o entries. Existing B4/B5/B6 entries are the retired Batch 2 placeholder
adapter runs, not Batch 2b official-method runs.

After this report and the B4o/B5o/B6o baseline cards are committed, Batch 2b may
proceed in the order required by doc 14:

```text
B4o -> B6o -> B5o
```
