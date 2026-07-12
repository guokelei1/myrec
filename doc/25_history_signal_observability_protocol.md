# Train-internal history-signal observability protocol

Status: pre-outcome protocol.  This is a diagnostic prerequisite for the next
proposed-system candidate, not a proposed architecture and not a paper result.

## Question

Before another architecture is designed, determine whether the current input
contract contains a reproducible, learnable signal for strict-nonrepeat
personalization, and identify whether that signal is carried by item text,
collaborative item identity, or their interaction.

The diagnostic uses only KuaiSearch `records_train` derivatives.  It must not
read dev/test records, qrels, evaluator outputs, or any C74/C75 validation
label.  All requests have a clicked candidate and nonempty strictly-prior
history.  Requests with any exact candidate/history item recurrence are
excluded before labels are opened.

## Split and controls

- Assign users, not requests, to three SHA-256 folds.  No user may occur in
  both fit and evaluation portions of a fold.
- Train each fold on the other two folds and produce scores for the held-out
  fold without reading held-out labels.  Labels may open only after all four
  modes have scored all three folds.
- Construct one label-free wrong-history donor per evaluation request by
  nearest frozen-query similarity, requiring a different user and a history
  length within a factor of two where possible.
- `full`, `text`, `id`, and `null` instantiate the same Transformer and have
  identical parameter counts, optimizer budget, candidate samples, batch
  order, initialization, and fixed final checkpoint.  Candidate/query
  semantics, candidate hashed-ID capacity, and fold-train popularity are
  common to every mode.  Only the history carrier differs:
  - `full`: history text plus hashed item identity;
  - `text`: history text only;
  - `id`: history hashed item identity only;
  - `null`: no history.
- For each history-aware checkpoint, score true, matched-wrong, reversed-event,
  and null history.  Reversal is diagnostic: order sensitivity is not required
  when a stable unordered preference signal exists.

The ranking core is a high-capacity diagnostic Transformer: a query/history
context encoder followed by candidate-to-context Transformer cross-attention.
It writes a learned residual over a fold-legal raw-BGE plus train-fold
popularity anchor.  This deliberately gives the signal every reasonable chance
to be observable; novelty is neither claimed nor scored.

## Fixed budget

- Three user-disjoint cross-fit folds.
- Four modes, one fixed seed per fold shared across modes.
- Four epochs, 16 sampled candidates per fit request, most recent 20 events.
- Width 256, four heads, three context layers, three candidate cross-attention
  layers, FFN width 768, 262,144 hashed item buckets of width 64.
- No checkpoint selection, retry, hyperparameter change, subset change, or
  mode-specific optimization.

## Predeclared interpretation

The primary metric is per-request NDCG@10 on the complete held-out candidate
sets, computed by the shared metric implementation.  Confidence intervals use
user-cluster bootstrap; fixed-fold signs are also reported.

A history source is **observable on KuaiSearch** only when all conditions hold:

1. OOF true-history minus the independently trained `null` mode is at least
   `+0.002` NDCG@10, its 95% user-cluster interval is above zero, and every fold
   is positive.
2. OOF true-history minus matched wrong history has a 95% interval above zero
   and every fold is positive.
3. Candidate hashes, user isolation, label staging, finiteness, determinism,
   and candidate permutation checks all pass.

Interpretation is frozen as follows:

- `text` passes: semantic cross-item history is a supported design target.
- `id` passes and `text` does not: the next architecture must create an
  internal collaborative/item-identity memory rather than another semantic
  attention law.
- only `full` passes: the next primitive must couple semantic and collaborative
  carriers; neither alone is an adequate premise.
- no source passes: the current Kuai input does not justify another
  strict-nonrepeat architecture based on these available sources.  Run the
  preregistered Amazon counterpart before deciding whether to narrow the claim
  or change the data contract.

Passing this diagnostic identifies an information source.  It does not validate
a proposed system, authorize dev/test access, or permit retrospective promotion
of any C01--C75 candidate.
