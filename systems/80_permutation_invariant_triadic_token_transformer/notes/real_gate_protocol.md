# C80 fresh Amazon real-gate protocol

## Data and label boundary

- fit: the existing 5,966 strict-nonrepeat token-HSO fit requests and compact
  fit labels;
- fresh: all 365 remaining C38-unused, strict-nonrepeat, fit-user-disjoint
  requests, selected from `records_train_blind.jsonl`;
- wrong history: different user, same frozen history-length bin, selected by
  request hash before labels;
- fresh labels open only after three seeds x five modes produce complete
  true/wrong/shuffle scores and all mechanical reports pass;
- dev/test/qrels remain closed.

## Frozen model and training

- BGE-small-en-v1.5 initialized from each corresponding token-HSO final
  checkpoint;
- frozen clone supplies base; adaptive clone trains all layers with dropout
  disabled in the paired path;
- two epochs, eight requests/batch, one positive + seven negatives;
- backbone LR `1e-5`, head LR `5e-5`, weight decay `1e-4`, final checkpoint;
- candidate token budget 8; history token budget 4 per event; six recent
  events; no retry or checkpoint selection.

The frozen admission reductions use the same backbone and trainable parameter
count.  `triadic_set` ranks C/H tokens by the registered triangle product;
`query_filtered_set` ranks each side only by maximum positive Q cosine;
`pairwise_set` ranks each side only by maximum positive C--H cosine;
`triadic_positional` changes only history position IDs to ordinary absolute
positions; and `ungated_full` retains every valid token.  All non-positional
modes reuse within-event positions.  The correction bound is inherited from
C78 (`rho=3`); it is not tuned on Amazon outcomes.

## Gate

Before labels: finite/decreasing loss, deterministic and candidate/event
permutation `<=2e-6`, frozen anchor/base hashes, active adaptive gradients,
complete candidate hash, and closed labels.

After labels, the three-seed ensemble primary must:

- exceed frozen base by at least `+0.002` NDCG@10 with positive 95% user-cluster
  CI and every seed positive;
- beat wrong history, every trained mode, and external ordinary full-token HSO
  with positive intervals and every seed positive;
- have true-shuffle absolute mean `<=0.002`;
- retain exact no-history behavior.

Any failure closes C80 and starts the mandatory C01--C80 retrospective.
