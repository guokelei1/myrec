# Transformer mechanism 24-hour triage and analysis handoff (v1)

Status: active closeout, user-directed on 2026-07-20.

## Decision

The current run set is too broad for a useful first diagnosis.  The 24-hour
closeout therefore keeps only the scorer processes that are already running at
the time of this decision and prevents any downstream GPU queue from starting.
Once those retained scorers reach a terminal state, the work moves to evidence
analysis.  No new layer, block, seed, dataset, control family, or outcome-
dependent branch may be started during this closeout.

## Retained until terminal

- `20260718_kuaisearch_mech_d2_q3_selected_branch_fold1_shard0of2_v1`
- `20260718_kuaisearch_mech_d2_q3_selected_branch_fold1_shard1of2_v1`

Only these two Q3 resume loops may continue.  Their lane shells must not launch
registered attention/MLP/RoPE/context follow-ups.  Q0 b20 was later removed by
the 50-percent threshold decision below; Q0 b27 and all Q0
trajectory/readout/replay continuations are deferred rather than started.

## Deferred and recorded for a later decision

- Q3 lane-0/lane-1 follow-ups (D3/D4/D5/D6/D7 breadth and readout jobs);
- Q0 b27 and all Q0/Q1 breadth, trajectory, readout, and optimizer-replay
  continuations;
- Q2 post-selected breadth and D2 synthesis queues;
- D4 MLP-formation recovery, component-necessity lanes/evaluator, and
  component-design synthesis;
- N8--N16, N17--N20, N25, and N26 queues/evaluators;
- watcher processes whose only purpose is to release one of the deferred
  bundles or evaluators.

These are not scientific null results.  They are `deferred_24h_triage` run
states and retain their manifests, commands, and existing partial outputs for a
later explicitly authorized batch.

## Handoff deliverable

After the two retained Q3 shard scorers close, run the shared integrity/ownership
audit, preserve all terminal metadata and resume lineage, and assemble the
current H0--H5 evidence view from completed first-round and mechanism bundles.
The next action is analysis and diagnosis, not automatic queue release.

## Execution record

At the closeout transition, the N8--N25 daemons, D4 recovery/necessity and
synthesis daemons, Q2/Q3 post-selected breadth shells, D2 synthesis shell, and
the deferred-result evaluator watchers were sent `SIGTERM`.  The three
retained resume loops and their scorer children were not stopped:

- Q3 shard0: resume loop PID 665028, scorer PID 1250481, physical GPU 1;
- Q3 shard1: resume loop PID 665303, scorer PID 1250599, physical GPU 3;
- Q0 b20: resume loop PID 1738828, scorer PID 1738839, physical GPU 0.

The initial post-stop ownership audit showed exactly these three active workers,
with no qrels opened by scorers and no source-test access.  After the
50-percent action below, the ownership audit must show only the two Q3 shard
workers.  A later batch may reuse the deferred queue scripts only after an
explicit review.

## 50-percent threshold decision

Progress was checked immediately before the threshold action:

| Run | Progress at decision | Action | Triage status |
|---|---:|---|---|
| `20260718_kuaisearch_mech_d2_q3_selected_branch_fold1_shard0of2_v1` | 1272/1959 (64.9%) | retain to terminal | `retained_over_50pct` |
| `20260718_kuaisearch_mech_d2_q3_selected_branch_fold1_shard1of2_v1` | 1404/1959 (71.7%) | retain to terminal | `retained_over_50pct` |
| `20260718_kuaisearch_mech_d6_q0_branch_b20_v1` | 1985/8000 (24.8%) | stop scorer and resume loop | `deferred_below_50pct` |

The Q0 b20 process pair (resume loop PID 1738828 and scorer PID 1738839) was
terminated after the check.  Its partial score/progress files remain untouched
for later inventory; they must not be interpreted as a completed scientific
result or rewritten as a zero-request run.  “未开始” here means not started in
the retained 50-percent closeout set, while the actual run-state remains a
truthful partial/deferred run.

## Full pause order

The user subsequently requested a complete pause.  The two remaining Q3
resume-loop/scorer pairs were stopped explicitly.  The final observed partial
progress was:

| Run | Progress at full pause | Metadata state preserved |
|---|---:|---|
| Q3 selected shard0 | 1290/1959 (65.8%) | `running`, resumable |
| Q3 selected shard1 | 1419/1959 (72.4%) | `running`, resumable |
| Q0 b20 | 1996/8000 (25.0%) | `running`, resumable |

All queue shells, watchers, resume loops, and scorer children are now stopped.
The GPU ownership audit reports zero active workers and no qrels/source-test
access.  The `running` metadata states are intentionally preserved as partial
interruptions; the triage record, not a falsified terminal status, records the
pause.  No process may be resumed until the existing evidence has been
inventoried and a new explicit decision is made.
