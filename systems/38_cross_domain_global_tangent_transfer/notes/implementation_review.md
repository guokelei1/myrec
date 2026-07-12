# C38 pre-outcome implementation review

## Verdict

The implementation is suitable for a confirmatory cross-domain falsifier.
Amazon-C4 C0 and C1 have passed; proposal and execution locks remain to be
created in order.  C38 is not suitable for a novelty claim and cannot be
promoted as the proposed architecture even if A1 passes.

## Contract review

- The frozen BGE Transformer state is the ranking state space; only the shared
  rank-16 down/up projection is trainable.
- The primary implements exactly one candidate-shared query-attended tangent
  write.  The unprojected and mean-history modes remove one defining operation
  each while retaining identical parameters, initialization, data, order,
  optimizer, and loss.
- There is no dataset/category/query-type branch, candidate-specific parameter,
  scalar candidate head, fixed-score router, or learned gate.
- Exact candidate recurrence is part of every mode's common fixed base.  A
  recurrence request receives exactly zero transport correction.  Empty
  history and absent query also receive exactly zero correction.
- Wrong-user donors are chosen from blind records by history-length bin only;
  target category and labels are not available to donor selection.
- Cohort selection, feature collection, encoding, G0, and A0 read only
  `records_train_blind.jsonl`.  A dedicated byte-offset opener parses only fit
  labels after proposal lock and only internal-A labels after A0.  C38 source
  contains no path or reader for upstream dev/test qrels.
- Candidate construction is outside the model and common to every mode:
  official sampled-1M catalog, frozen low-document-frequency query terms,
  SQLite FTS5 BM25 top-100, positive union, deterministic item-id tie break.
- C0 retains 93.07% of source history events after the nonblank-title mask,
  drops 6.93%, leaves every train history nonempty, and exposes 100% text
  coverage to all models.  C1 verifies train-blind equivalence, label
  isolation, candidate hashes, and history causality.

## Implementation evidence before real data

- Data conversion tests cover deterministic BM25 retrieval, history sorting
  and truncation, positive inclusion, train-blind export, and physical
  dev/test label isolation.
- Model tests cover paired initialization, equal capacity, exact no-history and
  no-query fallbacks, tangent orthogonality, exact reductions, candidate
  permutation equivariance, and finite adapter gradients.
- Selection/store tests cover label rejection, deterministic role assignment,
  wrong-user constraints, role-scoped feature materialization, candidate order
  preservation, recurrence detection, and role-scoped label access.
- A real CUDA smoke with `BAAI/bge-small-en-v1.5` produced finite distinct
  corrections for all three 12,288-parameter modes.

## Known limitations that must not be reinterpreted after outcome

1. Amazon-C4 queries are review-derived rather than organic search logs.
2. Each request has one known positive; an unobserved BM25 candidate may still
   be relevant.  This label incompleteness applies equally to all modes.
3. The history release does not expose the target timestamp.  It guarantees a
   temporal cutoff upstream; standardization verifies target-item absence,
   sorts retained events, and uses a surrogate request timestamp strictly
   after them.
4. JDsearch full data currently requires an interactive JD Cloud login.  Its
   GitHub sample is schema evidence only and cannot count as transfer utility.
5. A C38 pass supports the shared-write transfer premise, not candidate-level
   evidence fidelity or global novelty.  A failure closes this transport
   lineage without a threshold, encoder, candidate-pool, or loss rescue.

## Authorization boundary

Source implementation and synthetic/real-encoder smoke are complete.  Formal
fit labels, model training, internal-A scores, and any utility metric remain
unauthorized until C0/C1, selection, proposal lock, label-free embeddings, G0,
and execution lock complete in that order.
