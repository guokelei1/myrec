# Amazon token-level history observability protocol

Status: pre-outcome protocol.  This is the final representation-boundary test
before deciding whether another proposed architecture is scientifically
justified.

Pooled-state HSO found a small, stable Amazon text-history direction
(`true-null +0.00166`, `true-wrong +0.00130`) but it missed the frozen practical
threshold.  KuaiSearch had no corresponding direction.  The remaining untested
possibility is that pooling item texts before query/history/candidate
interaction destroys the useful relation.  This protocol therefore gives a
standard full-token cross-encoder—not a novel proposed model—the opportunity to
recover it.

## Data boundary

- Fit uses C38's 6,000 already-exposed train fit requests, after label-free
  removal of exact candidate/history recurrence.
- Evaluation uses 1,200 requests selected by the existing frozen C38 request
  order from its 1,599 never-assigned reserve.  Remove exact recurrence and any
  user appearing in fit before taking 1,200.  No prior C38--C42 feature, score,
  or label exists for this reserve.
- Wrong history is selected before labels from a different user in the same
  frozen history-length bin.  Query, candidates, and labels never change.
- Training may read only compact fit labels after the execution lock.  Reserve
  labels open only after all three seeds have produced true/null/wrong scores
  and passed mechanics.  Dev, test, and qrels remain closed.

## Diagnostic model

For every candidate, construct one BGE-small-en sequence:

`[CLS] query [SEP] candidate [SEP] recent history items [SEP]`

The query and candidate keep up to 32 WordPieces.  The six most recent history
items keep up to 20 WordPieces each; the final sequence is capped at 192 tokens.
All four pretrained BGE layers and one scalar CLS readout train end to end.
This is ordinary candidate-wise cross-encoding: candidates do not communicate,
so list order cannot create an advantage.

Each fit request samples one positive plus seven negatives.  Request-level
history dropout is fixed at 15%, so the same checkpoint learns both factual and
empty-history paths without giving null examples equal majority weight.  Three
fixed seeds train four epochs with final-checkpoint-only selection.  No scale,
length, dropout, learning-rate, epoch, candidate, seed, or checkpoint retry is
allowed.

## Decision

The token-level semantic source is observable only if:

1. the three-seed ensemble true-minus-null NDCG@10 is at least `+0.002`, its 95%
   user-cluster interval is above zero, and every seed is positive;
2. ensemble true-minus-wrong has a positive interval and every seed is
   positive;
3. all seeds reduce fit loss and pass finiteness, determinism, candidate-order,
   candidate-hash, and label-stage checks.

Passing establishes a token-level semantic information source and authorizes
one architecture primitive that preserves token interactions while enforcing
Kuai recurrence/no-history safety.  Failure closes current raw-text, pooled
semantic, and hashed-ID history sources as sufficient motivation.  The next
step must then change the data contract or narrow the PPS claim, not invent
another Transformer routing law.
