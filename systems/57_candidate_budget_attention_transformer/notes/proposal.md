# C57 proposal — candidate-budget attention

Status: pre-outcome fit-internal architecture gate.  C57 reuses the immutable
C56 4,800/1,200 split and contextual-token shards.  C56 never opened the 1,200
holdout labels.  C57 may train on already opened fit-train labels only after
this proposal and execution are locked; holdout labels require a passed A0.

## Failure-derived law

C53/C54/C56 repeatedly let every candidate read history independently and
only compared candidate carriers afterward.  The learned history state then
became either zero or candidate-common, while a history-free raw candidate
list path remained strongly trainable.  Post-carrier centering or list
attention is too late.

C57 changes the attention normalization axis.  Candidate/query token states
form exchangeable candidate slots, and ordered history-event token states are
the evidence inputs.  For head `r`, event `j`, and candidate `i`, the shared
Transformer produces compatibility `ell_ijr`.  Each event must allocate one
finite evidence budget across all valid candidates and a learned NULL sink:

```text
alpha_ijr = exp(ell_ijr) /
             (exp(ell_NULL,jr) + sum_k exp(ell_kjr)).
```

Candidate updates use only the history values assigned by this competitive
normalization.  With normalized event weights `omega_j`,

```text
u_ir = (C+1) sum_j omega_j alpha_ijr V_r(h_j)
f_ir = <u_ir, V_r(c_i)> / sqrt(d_r)
delta_i = W_out(f_i - mean_k f_k).
```

The `(C+1)` factor removes the trivial cardinality shrinkage of uniform
allocation; candidate centering removes the uniform solution itself.  The
NULL sink preserves unassigned evidence mass.  There is no output router,
dataset/category branch, candidate rank position, or external history score.
Empty history/query inputs are exact strong-base no-ops, and exact-recurrence
requests return the registered item-only anchor.

The token Transformer and query-to-item content attention are shared across
query, candidates, and history.  Thus the LM/Transformer remains the ranking
core; the new primitive is the candidate-axis normalization inside its
history attention, not a loss or scale change.

## Binding reductions

All five modes instantiate the same parameters, initialization, optimizer,
request schedule, strong base, and listwise loss:

- `candidate_budget`: primary candidate-axis allocation with NULL;
- `slot_budget_no_null`: the direct Slot Attention-like candidate-axis
  normalization without the NULL sink;
- `history_softmax`: ordinary candidate-independent softmax over history plus
  NULL, the DIN/ZAM-style target-attention axis;
- `pooled_history`: query-conditioned event mean matched independently to each
  candidate;
- `raw_candidate`: query/candidate token score with no history value.

The trained primary is also scored with wrong-user history and, at the same
checkpoint, with the normalization axis changed to `history_softmax`.  The
latter is the binding load-bearing intervention; retrained controls are the
utility comparisons.

## Gate and stop rule

Three fixed seeds train all five modes for two epochs.  Before holdout labels
open, every seed must be finite, deterministic, candidate-equivariant, exact
on real no-history/repeat roles, and the primary must materially change base
orders.  Wrong history and same-checkpoint axis ablation must each change at
least 5% of complete orders and 1% of Top-10 sets in every seed and ensemble.

Only a passed A0 permits the common metric to read compact holdout fit labels.
A1 then requires positive-CI NDCG gains over base, wrong history, and every
retrained control, with all seed and hash-fold directions positive.  Failure
closes this attention-axis law; no temperature, NULL bias, scale, candidate
count, width, epoch, loss, seed, threshold, or subset rescue is allowed.

A pass would authorize a separately frozen dual-domain/fresh proposal and
novelty review.  This exposed gate alone cannot establish a final architecture
or global novelty.
