# Amazon-C4 history-signal observability protocol

Status: pre-outcome protocol.  This is the preregistered cross-domain branch of
the KuaiSearch HSO decision, not a proposed-system result.

KuaiSearch HSO found no stable strict-nonrepeat value from text, hashed item ID,
or their combination.  Before changing the data contract or resuming
architecture search, this protocol asks whether the same information sources
are observable on Amazon-C4.

The diagnostic reuses the frozen, label-free BGE-small-en feature surface
created for C38: 7,200 train requests, their complete candidates, true and
different-user wrong histories, and immutable query/item embeddings.  C38
outcomes do not select requests, modes, folds, checkpoints, or thresholds here.
This is an exposed train-internal formulation audit and cannot support a final
paper number.

## Isolation and modes

- Remove every exact candidate/history recurrence before labels.
- Assign users to three SHA-256 folds; each fold trains on the other users and
  scores complete held-out candidate sets before held-out labels open.
- `full`, `text`, and `id` use the identical HSO Transformer, parameter count,
  initialization, candidate samples, optimizer, and fixed final checkpoint.
  Query/candidate semantics, hashed candidate IDs, and fold-train popularity
  remain common.  Only the history carrier differs.
- Every batch has fixed 15% request-level empty-history dropout.  Every final
  checkpoint scores true, stored matched-wrong, reversed-event, and empty
  history.  The trained same-checkpoint empty path and fold-legal frozen BGE
  plus popularity anchor are both binding controls.
- No independently trained null model is used; Kuai HSO established that this
  control's fixed optimization was unstable in all folds.

The fixed budget matches Kuai HSO except for the frozen English embedding width:
four epochs, 16 sampled candidates, 20 recent events, width 256, four heads,
three context and three candidate-cross-attention layers, FFN 768, and 262,144
hashed item buckets of width 64.

## Decision

A source is observable only if true history:

1. beats its same-checkpoint empty path by at least `+0.002` NDCG@10 with a
   positive 95% user-cluster interval and every fold positive;
2. beats the fixed fold-legal base by at least `+0.002`, again with a positive
   interval and every fold positive;
3. beats matched wrong history with a positive interval and every fold
   positive; and
4. passes candidate hash, finiteness, determinism, permutation, and label-stage
   contracts in all folds.

If text passes, semantic cross-item preference is observable on Amazon but not
Kuai; the next architecture may target semantic history while retaining exact
fallback and must treat this domain asymmetry as a data boundary.  ID-only and
full-only outcomes map to collaborative identity memory and a coupled carrier,
respectively.  If no mode passes, current text/ID/history inputs do not support
continued strict-nonrepeat architecture invention; the next step must change
the observable data contract or narrow the claim, not create another attention
law.

Dev, test, and qrels remain closed in every branch.
