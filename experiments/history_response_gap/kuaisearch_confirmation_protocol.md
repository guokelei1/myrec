# KuaiSearch Full disjoint-window motivation confirmation lock

Frozen: 2026-07-15, after completion of target-aware construct repair,
matched-null controls, KuaiSAR functional replication, and the two-seed Qwen
family adequacy decision; before materializing or reading any outcome from this
confirmation population.

Status update, 2026-07-16: execution completed once and all five gates passed.
This file is retained as the frozen reproduction protocol, not an active plan.
Further confirmation tuning, test access, and proposed architecture remain
locked.

## 1. Confirmation question

Can a task-adequate ordinary decoder full-token ranker be aggregate-successful
while its history utility remains sharply concentrated in target recurrence and
fails to establish a practically meaningful recovery on target-nonrepeat traffic
whose history is disjoint from the complete candidate slate?

This confirms empirical state separation. It does not test a universal
direction-blindness claim, causal base erosion, or an architecture mechanism.

## 2. Independent population lock

Source: immutable KuaiSearch Full public `recall/train.jsonl` and
`items/train.jsonl`; source rows with `split != train` remain excluded.

Dataset version: `full_confirm_preceding10k_v1`.

Label-free selection:

1. retain source-train requests with 2--100 exposed candidates;
2. require request time index strictly less than `1897788`, the minimum time
   index of the already inspected `full_scout10k_query_history_v1` window;
3. take the latest 10,000 eligible requests below that cutoff;
4. split the selected block internally by time into approximately 80% train and
   20% confirmation, moving all boundary-time ties to confirmation;
5. require zero request/session overlap across train and confirmation and zero
   request overlap with the explored latest-window dataset;
6. use at most 20 causal history events and retain their prior query text;
7. write labels only to `qrels_train.jsonl` and
   `qrels_confirmation.jsonl`; confirmation records remain label-free.

This is an older, disjoint historical block with its own train-to-confirmation
time order. It supports independent population confirmation, not a claim of
future temporal generalization from the already trained exploratory checkpoint.
No exploratory checkpoint is reused.

Before any confirmation-label access, freeze and hash records, candidate and
request manifests, source files, true/null/wrong assignments, and exact request
overlap audits. Any leakage, noncausal history, candidate duplication, or
non-empty label field in confirmation records invalidates the population.

## 3. Frozen model and controls

Primary family: `Qwen/Qwen3-Reranker-0.6B`, loaded from the existing verified
local checkpoint. Train fresh QC and FULL checkpoints on this population's
train split only.

Both variants use the already frozen Full-source recipe:

- official CrossEncoder prompt/preprocessing;
- pointwise binary cross entropy;
- two deterministic negatives per observed positive;
- one epoch, batch size 8, gradient accumulation 2;
- learning rate `1e-5`, weight decay `0.01`, warmup ratio `0.1`;
- seed `20260714`, float32 master parameters and bfloat16 autocast;
- max length 768 and `longest_first` truncation;
- QC uses query only and history budget 0;
- FULL uses query plus the six most recent history events;
- no dev selection, retry tuning, history dropout, anchoring, or architecture
  change.

The train-derived QC/FULL example count and optimizer update count must match.
Mechanical failures before optimization or score completion may be repaired only
without changing scientific fields and must receive a new run ID.

Task controls:

- request-local BM25 is the lexical sanity anchor;
- target-repeat recovery is the binding history-learnability positive control;
- FULL true/null/wrong use one unchanged checkpoint, request-aligned batches,
  identical scoring signature, and exact candidate/request hashes.

## 4. Frozen endpoints and ordered decision

Registered label mode: graded NDCG@10. Report both all-request and
conditional-positive estimands. Activity epsilon is `0.01`; utility epsilon is
`0.0`. Query-cluster bootstrap uses 5,000 draws and seed `20260715`.

The confirmation decision is hierarchical, so later steps are interpreted only
after earlier gates pass:

1. **Population power:** at least 80 target-repeat and 400
   target-nonrepeat/no-candidate-overlap confirmation requests. Otherwise the
   result is underpowered/inconclusive, not a failed motivation.
2. **Task adequacy:** Qwen-QC is not significantly below request-local BM25 in
   paired 95% bootstrap (`QC - BM25` interval upper bound is non-negative),
   training/scores are finite, and all hashes match.
3. **History positive control:** target-repeat `FULL_true - FULL_null` has a
   query-cluster 95% interval whose lower bound is above zero.
4. **Recurrence--transfer separation:** the request-level contrast
   `mean recovery(target_repeat) - mean recovery(target_nonrepeat/no-overlap)`
   has a query-cluster 95% interval whose lower bound is above zero.
5. **Practical nonrepeat bound:** the upper 95% bound for
   target-nonrepeat/no-overlap recovery is below `0.03` NDCG@10.

The `0.03` materiality bound is frozen after exploration and before confirmation.
It is deliberately larger than the cross-seed no-overlap fluctuations observed
during exploration, while remaining far below the recurrence effect that a
useful personalized ranker demonstrably learns.

The motivation is confirmed only if all five ordered conditions pass. If task
adequacy or power fails, the result is inconclusive. If adequacy passes but a
scientific condition fails, the confirmation rejects that bounded claim; no
new slice, threshold, seed, or checkpoint may rescue it.

## 5. Secondary diagnostics

Reported regardless of sign, but not used to rescue the primary decision:

- aggregate and observed-positive QC/null/true accounting;
- target-nonrepeat/other-candidate-overlap recovery;
- pairwise direction on every target-aware surface;
- true-versus-wrong history recovery and signed alignment;
- response activity and common-mode ratio;
- full-true versus QC endpoint;
- target-surface traffic and signed contribution to aggregate recovery.

Direction may be above chance without reliable recovery; that outcome is
compatible with the state-separation thesis and must not be rewritten as
universal direction blindness.

## 6. Execution and stopping boundary

Training and scoring may read only train qrels; confirmation qrels are opened
once, by the shared evaluator, after QC, BM25, and the complete FULL
true/null/wrong score bundle have passed label-free integrity checks. The
confirmation evaluator appends to `reports/confirmation_eval_log.jsonl`, not the
development ledger.

All outputs, including negative or inconclusive results, are retained. There is
one seed and one frozen recipe because seed variability and model-family
adequacy were already evaluated in exploration. No confirmation tuning,
probability rescue, alternative baseline selection, test access, or proposed
architecture is authorized.
