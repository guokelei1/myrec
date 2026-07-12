# Amazon full-token attention-edge attribution protocol

Status: frozen formulation after the outcome of the full-token history
observability probe.  This is a post-outcome mechanism attribution on the
same 1,200 Amazon reserve requests; it is not a fresh utility gate.

The ordinary joint cross-encoder in `doc/28` established a practical and
specific history source (`true-null +0.02530`, `true-wrong +0.03594`).  Pooled
item representations did not expose that source.  Before defining another
architecture, this protocol asks which full-token attention edges are
load-bearing.  It changes no weight, checkpoint, text, history donor,
candidate set, or label.

## Frozen interventions

For the sequence

`[CLS] query [SEP] candidate [SEP] history [SEP] ...`,

tokens are assigned to query/readout (`Q`), candidate (`C`), or history (`H`)
before scoring.  The original three final checkpoints are evaluated with the
following layer-shared attention masks:

- `history_isolated`: remove both directions of `Q-H` and `C-H`;
- `no_query_history`: remove both directions of `Q-H`;
- `no_candidate_history`: remove both directions of `C-H`;
- `no_query_reads_history`: remove only queries in `Q` reading keys/values in
  `H`;
- `no_candidate_reads_history`: remove only queries in `C` reading
  keys/values in `H`;
- `no_history_reads_context`: remove history queries reading `Q` or `C`.

Padding remains masked and all remaining token edges are unchanged.  The
first two separators stay with `Q` and `C`, respectively.  Every non-isolation
mask is scored with true and matched wrong-user histories.  The original
full/null scores are reused byte-for-byte.

## Mechanical contract

Because BERT normalization is token-local, `history_isolated` must reproduce
the checkpoint's null score for the readout token within `2e-6`.  Candidate
permutation and deterministic rescoring must remain within `2e-6`; score and
checkpoint hashes must match the prior locked run.  Failure is mechanical and
does not support an architectural conclusion.

## Interpretation rule

For each mask, compute ensemble NDCG@10 on complete candidate sets and
user-cluster bootstrap intervals for masked true-minus-null, masked
true-minus-wrong, and full-true-minus-masked-true.  Relative to the original
full true-minus-null effect:

- `retained`: masked true-minus-null retains at least 80% and masked
  true-minus-wrong has a positive 95% interval;
- `destroyed`: it retains at most 20%, or the true-minus-wrong interval is
  non-positive;
- otherwise: `partial`.

If `no_candidate_history` retains the source, a query-mediated path is
sufficient.  If `no_query_history` retains it, candidate-mediated history
attention is sufficient.  If neither retains it, the next primitive must
preserve both raw-token edge families rather than reduce history to a relay or
pooled profile.  Directional masks refine the intervention locus.  These
categories formulate C76; they do not constitute fresh-domain confirmation.
