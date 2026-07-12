# C77 proposal — Query-Authenticated Token-Subgraph Transformer

Status: pre-outcome.  C77 is architecture update 1 of at most 3 after C76; no
C77 outcome or repository data has been observed.

## Observation → architecture consequence → falsification

**Observation.** C76 proved that a factual/history-cut trajectory is not an
evidence identity.  A candidate-only nuisance can change the candidate state
that conditions history attention, so the nuisance appears inside every
factual-minus-cut residual even though there is no direct factual-state bypass.
All five C76 modes fit the nuisance and reached zero supported accuracy under
its held-out reversal.  The intervention must therefore occur *before* label-
adaptive token mixing, at the eligibility of cross-history token edges.

**Architecture consequence.** A frozen local pretrained LM produces normalized
semantic anchors `a_t` for query, candidate, and history WordPieces.  These
anchors receive no ranking gradient.  For candidate token `c` and history token
`h`, define query-authenticated support

```text
g(c,h|Q) = max_q [<a_c,a_q>]_+ [<a_h,a_q>]_+ [<a_c,a_h>]_+.
```

Candidate/history tokens belong to the personalized subgraph only if they
participate in at least one positive `g` triangle.  Query/readout tokens and
all admitted candidate/history tokens then enter a trainable, multi-layer,
bidirectional interaction Transformer.  Attention is dense *inside* this
subgraph, including direct C-H and H-C edges; ineligible candidate/history
tokens cannot write to or read from it.  The subgraph is frozen for the
request, so later adaptive layers cannot manufacture a new edge from a
ranking-label shortcut.

One shared interaction Transformer is evaluated on the authenticated graph
with and without its H cross-edges.  Only its candidate logit difference can
modify a separately protected query-candidate LM base:

```text
delta_i = head(T_auth(q,c_i,H)) - head(T_auth(q,c_i,H; H cut))
score_i = base_i + center_candidates(rho*tanh(delta_i)).
```

Here the subtraction is a safety interface, not the novelty claim; the
`ungated_full` control uses the identical interface.  Empty history or masked
query creates no admissible triangle and returns the base exactly.  Exact
recurrence retains the universal item-identity coordinate.  History order is
not claimed and is expected to be stable under event permutation.

The single primitive is the **frozen query-authenticated token subgraph**.  It
retains raw token and direct bidirectional Q-H-C interaction after admission,
but ranking labels cannot redefine admission.  There is no dataset/category/
query-type branch, user table, learned threshold, offline LLM call, or pooled
history profile.

**Falsification.** C77 reuses C76's exact frozen raw-token train/validation
nuisance surface, split, steps, and thresholds.  No generator or nuisance
change is permitted.  It fails unless all three seeds pass mechanics,
supported utility, wrong/query-mask specificity, shuffle stability, and beat
four capacity-matched graph reductions:

1. `ungated_full`: every token is admitted (ordinary full Transformer);
2. `query_history_filter`: query filters H but every candidate token is
   admitted, so candidate-only shortcuts remain possible;
3. `query_candidate_filter`: query filters C but all H tokens enter, removing
   shared C-H authentication;
4. `pairwise_candidate_history`: frozen C-H similarity without the query
   factor, admitting irrelevant-preference matches.

A data-free pass authorizes only a fresh real fit protocol.  Failure closes
this frozen-anchor subgraph; there is no anchor layer, threshold, similarity,
width, step, seed, or pretrained-model rescue.

## Efficiency and real instantiation

The real model uses one frozen lower/pretrained token encoder to build anchors
and one compact interaction Transformer as the end-to-end ranking core.  Anchor
states may be cached per query/item text; the request-specific triangle and
interaction remain online.  Latency, active-token fraction, and the ungated
full-token cross-encoder are binding comparisons.
