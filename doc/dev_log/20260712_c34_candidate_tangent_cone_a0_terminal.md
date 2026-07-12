# C34 candidate tangent-cone A0 terminal

C34 was the first post-C33 candidate to replace one request-global query state
with a candidate-specific Transformer read.  It also used a wholly new target
fit cohort, so its gate did not inherit C31--C33 fit or outcome requests.

Mechanically the architecture worked: it trained, changed rankings, reacted to
wrong history, differed materially from ordinary target attention and global
tangent transport, and preserved every exact fallback.  Scientifically its
admission law failed without reading labels.  With multiple history events,
the chance that at least one query-centred cosine is positive is so high that
only 2--5 of 15,424 active candidate rows abstained per seed.  An absolute
positive half-space therefore confuses chance compatibility with discriminative
evidence.

This is a problem-level, not dataset-slice, diagnosis.  The admissible next
primitive is candidate-relative event surplus: an event may write candidate
`i` only if it supports `i` more than the contemporaneous candidate set.  That
requires a new candidate-axis attention normalization and an untouched gate;
raising the C34 cosine threshold would be post-outcome tuning and is forbidden.

Curated report: `reports/pps_c34_train_gate.json`; no internal-A label, dev,
test, delayed-B, or escrow surface was opened.
