# C56 proposal — query-complement token competition

Status: pre-outcome architecture/foundation proposal.  Only C26's already
opened fit labels may be reused after the split, implementation, and execution
lock are frozen.  C26 internal-A/delayed-B/escrow, dev/test, and qrels remain
closed.

## Failure-derived hypothesis

C53 showed that an ordinary joint-context Transformer learns a
history-invariant list-reranking shortcut.  C54 algebraically restricted the
candidate-list value stream to a pooled factual-minus-null history carrier,
but the carrier still reproduced the strong base.  C55 then standardized base
units and used the exact probability residual target; pooled history states
still did not beat a history-free control.

The remaining hypothesis is narrower: the strong base already represents the
query-supported part of candidate relevance, so cross-item personalization
must be formed from candidate/history *token content that the query stream does
not already explain*.  It must alter candidate token states before pooling and
must survive a candidate-relative, rather than independent scalar, readout.

## Single operator

A frozen pretrained LM supplies contextual query, candidate-title, and
history-title tokens.  A shared trainable projection/refinement block maps
them to the same hidden space.  For any candidate or history token `x`, shared
query cross-attention gives its query-explained component `A_Q(x)` and the
query complement is

```text
x_perp = x - A_Q(x).
```

Candidate-complement tokens attend to all ordered history-complement tokens.
History event weights multiply only values; a learned event-position embedding
retains order.  A zero-valued NULL token is always present, so empty history
produces exact zero transport with bias-free attention.

The transported token state is injected before pooling through one shared
token Transformer evaluated on factual and null inputs:

```text
d_i = Pool(T(c_i + Attn(c_i_perp, H_perp, H_perp)) - T(c_i)).
```

Candidate-set attention uses candidate/base states as Q/K and `d_i` as its
only V stream.  The final state is the candidate's carrier relative to the
leave-one-out carrier message from the other candidates:

```text
r_i = d_i - Attn_LOO(c_i, C, D)_i
delta_i = head(r_i) - mean_k head(r_k)
score_i = standardized_strong_base_i + delta_i.
```

Thus a history-free candidate path cannot create a primary correction;
candidate-common history writes cancel; and caller candidate order remains a
set permutation.  Query-missing/no-history requests return the strong base
exactly.  Exact-recurrence requests bypass the cross-item residual and return
the registered item-only anchor exactly.  These are evidence-mask conditions,
not dataset branches.

## Binding reductions

All trainable controls have identical parameters, initialization, optimizer,
request order, contextual LM tokens, and loss.

- `unprojected_token`: retains token transport/competition but uses original
  candidate/history tokens instead of `x - A_Q(x)`;
- `pooled_complement`: computes the same query complement but pools each item
  before history transport, reducing the hypothesis to a C54-like pooled
  carrier;
- `raw_candidate`: sends history-free candidate token states through the same
  leave-one-out candidate competition, exposing the C53 shortcut;
- `edge_ablation`: same trained primary checkpoint with the leave-one-out
  message set to zero;
- `wrong_history`: same trained primary checkpoint with the frozen C26 donor.

The primary must beat the retrained reductions, not merely change numbers.
The edge and wrong-history interventions must change complete and Top-10
orders before holdout labels are opened.

## Objective and gate

The frozen strong-base logits are standardized per request.  All modes train
with the same listwise cross-entropy on `base + centered_correction`, from an
exact-zero output head.  The probability residual `y-softmax(base)` is logged
as a diagnostic target, not given a mode-specific loss or scale.

The C26 fit role is hash-split without reading labels into 4,800 train and
1,200 holdout requests.  Candidate hashes are asserted before all scoring and
again before the shared metric reads compact fit labels.  A0 requires finite
training, deterministic/candidate-equivariant scores, exact real no-history
and recurrence fallbacks, material order sensitivity to wrong history and the
leave-one-out message, and paired initialization/capacity.  Only then may A1
open compact holdout labels.  A1 requires positive-interval NDCG and residual
MSE gains over base, wrong history, and every retrained control, with every
seed/fold direction positive.

Any failure closes this operator on the exposed fit surface.  There is no
width, layer, epoch, scale, seed, threshold, subset, or loss rescue.  A pass
authorizes a separately frozen dual-domain/fresh proposal; it is not itself a
novelty or final-system claim.
