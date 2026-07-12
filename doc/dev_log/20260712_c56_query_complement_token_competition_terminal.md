# C56 query-complement token competition terminal

C56 tested the first trainable token-formation successor after C55.  Frozen
BGE contextual tokens covered 267,276 items and 6,256 requests.  A single
dataset-agnostic model rejected query-explained candidate/history token
content, formed a factual/null token-state delta before pooling, and allowed
candidate-set attention to transport only that history delta.  Four modes and
three GPU seeds shared capacity, initialization, loss, request order, strong
base units, and exact fallbacks.

Two pre-outcome status aggregations incorrectly treated the expected boolean
`fit_labels_read=false` as a failed all-true condition.  Both immutable aborts
are recorded in the candidate notes.  They happened before model/ranking
outcomes; v3 changed only those predicates and reused the exact hash-verified
selection/contextual shards.

The clean v3 A0 failed.  Ensemble primary/base, true/wrong, and primary/edge
changes were only 2, 1, and 4 of 1,200 complete orders; all Top-10 changes were
zero.  Holdout labels stayed closed.  A label-free checkpoint diagnostic found
that raw-candidate list reranking was highly active, but the history branch
either became exact zero or a candidate-common direction removed by
centering.  This repeats C02's common-mode lesson at token granularity and
shows that later list competition cannot repair a carrier already collapsed.

The next architecture search is therefore narrowed to the attention kernel:
history evidence must compete across candidates during normalization, rather
than letting each candidate independently read history and comparing the
result afterward.  This is a problem-derived change, not a dataset-specific
branch; it must still beat standard target attention, pooled, and raw-list
controls before any label utility gate.
