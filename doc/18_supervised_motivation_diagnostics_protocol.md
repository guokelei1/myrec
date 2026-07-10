# 18 - Supervised Motivation Diagnostics Protocol

Status: locked before internal calibration and before any new dev evaluation on
2026-07-10.

## 1. Purpose

Historical note: C3-R originally appeared to show identity-specific predictive
value, but its train-frozen donor interpretation is superseded by failed C5-R2
(`doc/22`), and the finite C5-R3 component gate terminates as benchmark/analysis
only (`doc/23`). The D1 diagnostic below remains valid as a bounded supervised
negative result. It answers two questions
that a reviewer can still raise:

1. Is the apparent weakness of query-only ranking merely caused by using
   zero-shot scorers?
2. Can a train-fitted model extract history value beyond a supervised query
   base, and does query-conditioned event weighting help beyond an
   unconditioned history summary?

This protocol answers those questions before the proposed system is designed.
The models are diagnostics and baselines, not the paper's method.

## 2. Frozen Data and Embeddings

- Train/dev records and candidate manifest remain `v0_lite`.
- Labels are read only from `records_train.jsonl`.
- Dev records remain label-free during training and scoring.
- The frozen B5o Stage-B BGE artifact supplies 512-dimensional query and item
  embeddings for all 163,717 train and 12,229 dev requests and all required
  candidate/history items.
- Histories remain the causally constructed, at-most-50 prior events.
- No test record, qrels, or metric is read.

A new compact array materializer may transform these existing fields but may
not alter candidate order, labels, histories, or embedding indices. It records
all source and output hashes.

Pre-training implementation amendment: the first materialization attempt
stopped before writing arrays because some train exposure requests contain no
clicked candidate, for which the frozen multi-positive listwise objective is
undefined. These requests are excluded from the optimization/calibration
arrays only; their count is reported. No model was trained and no dev metric was
read before this amendment. Dev evaluation scope is unchanged.

Pre-training batching amendment: retained candidate counts have a long tail
(maximum 2,196). Batches therefore contain at most 512 requests and are closed
early when padded candidate slots would exceed 65,536 or padded history slots
would exceed 32,768. No candidate/history is truncated, and the rule is shared
by calibration, final training, and scoring.

## 3. Train-Only Calibration

The final 10% of train requests in frozen record order is the only calibration
set. The first 90% is the internal training set. Model selection uses internal
NDCG@10 only.

One calibration run per variant may train for at most eight epochs with
patience two. One retry is allowed only for numerical failure or failure to
exceed both of the variant's frozen input features on internal validation. The
retry and reason must be logged before any dev evaluation. Dev never selects an
epoch, learning rate, architecture, or retry.

After calibration, the selected epoch count and unchanged hyperparameters are
written to a separate frozen final config. Each final model then trains on all
train requests with seeds 20260708/09/10.

## 4. Diagnostic Model Family

All variants use frozen BGE embeddings and the same multi-positive listwise
softmax objective.

### D1q - Supervised Query Base

Candidate scores combine three within-request standardized features:

1. frozen query-item BGE cosine;
2. train-only item log-click count;
3. a learned low-rank query-item interaction.

This is a supervised, non-personalized ranker. It is stronger than the existing
zero-shot query controls without using user history.

### D1m - Mean-History Residual

Load the seed-matched D1q checkpoint and freeze the complete query base. Add a
masked residual containing the frozen B0b score and a candidate interaction
with a recency-weighted mean of history embeddings. No query controls event
weights.

### D1a - Query-Attentive History Residual

Use the same frozen D1q base and residual features, but compute the history
summary with query-to-event attention before candidate interaction.

For both residual variants, empty history forces the residual exactly to zero,
so predictions must be byte-identical to the seed-matched D1q base on those
requests. Only residual parameters are trained.

## 5. Evaluation and Budget

Nine final dev evaluations are authorized: three variants times three seeds.
They are fixed motivation diagnostics, not proposed-system tuning. Every run
must use the shared evaluator and candidate hash. Seed 20260708 is the
preselected paired-significance reference; all three seed directions and the
mean/variation are reported.

Required comparisons:

- D1q versus B2z, B0a, and B7;
- D1m and D1a versus their seed-matched D1q;
- D1m and D1a versus B7;
- D1a versus D1m;
- exact no-history score equality for each residual/base seed pair;
- matched wrong-history rescoring of the strongest residual if it exceeds D1q.

The wrong-history rescore does not retrain and receives at most three additional
fixed evaluations, one per seed.

### 5.1 Label-free explanatory slices

Before any supervised diagnostic dev evaluation, the following descriptive
slices were locked. They do not alter the headline tests or authorize model
selection:

- history availability: empty versus non-empty;
- non-empty history length: 1, 2--5, 6--20, and 21--50 events;
- exact history/candidate overlap: no history item appears in the current
  candidate set versus at least one overlap;
- query/history semantic affinity: quartiles of the maximum frozen BGE cosine
  between the query and its history events, with cut points estimated from
  history-present train requests only;
- exact query frequency in train: unseen, 1--4, and at least 5 train requests.

For each slice, report request count and paired NDCG@10 deltas for D1m-D1q,
D1a-D1q, and D1a-D1m using seed 20260708. Slice findings are exploratory and
must be presented as design observations, not new gates.

## 6. Locked Interpretation

- If D1q reaches or exceeds B7, withdraw any broad query-saturation story and
  make D1q the new baseline-to-beat.
- If a residual beats D1q but not B7, history is learnable but the remaining
  problem is interaction/representation quality, not oracle headroom.
- If a residual beats B7, train-fitted history use is viable. D1a must also beat
  D1m before query-conditioned event selection becomes a positive observation.
- If D1a does not beat D1m, query-conditioned event selection remains a design
  hypothesis and cannot appear as an established motivation fact.
- If neither residual beats D1q, retain identity specificity from C3-R but do
  not claim learnable history gain for this representation family.

All outcomes are publishable diagnostics. No branch authorizes changing the
metric, split, model family, or dev budget after seeing results.

## 7. Boundary

This is still an internal dev-stage gate. Any paper-level performance or causal
claim requires the untouched test or a secondary track after the proposed
system and all baselines are frozen.
