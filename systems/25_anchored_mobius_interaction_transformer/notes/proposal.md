# C25 proposal — anchored Möbius interaction Transformer

Status: pre-outcome design; no C25 label has been opened.

## Observation → architecture consequence

C23 and C24 both allowed the proposed information object to enter an otherwise
strong recurrence path.  The models changed rankings but ignored the suffix or
candidate-candidate edges.  Their learned trajectories were nearly identical
to independent recurrence controls.  The next residual must therefore be
unable to reconstruct an independent recurrence/base shortcut.

For projected query `q`, candidate `c`, history event `h`, let `G` be one
shared nonlinear potential.  C25 forms the anchored third discrete derivative

```text
M(q,c,h) = G(q+c+h) - G(q+c) - G(q+h) - G(c+h)
           + G(q) + G(c) + G(h) - G(0).
```

Every additive main effect and every pairwise function represented by the
shared potential cancels.  `M` is exactly zero if any one of `q,c,h` is the
registered null.  A position-free Transformer receives only `[READ, M_1, ...,
M_H]`; it never receives raw `q`, `c`, D2p, recurrence mass or a candidate
token through another residual path.  Candidate-centred readouts modify D2p
only on history-present requests with no exact candidate recurrence.  Any
repeat-present request returns registered item-only exactly.

## Falsifiable claim

The claim is not that higher-order interaction is new.  It is that *purifying
the event write before aggregation* prevents lower-order shortcuts and exposes
a useful query-history-candidate state on strict non-repeat requests.  The
primary must beat three equal-parameter and equal-potential-evaluation
alternatives:

- `joint_delta`: ordinary nonlinear history increment `G(q+c+h)-G(q+c)`;
- `pairwise_ch`: query-independent candidate-history Möbius term;
- `trilinear`: direct projected `q*c*h` tri-attention token.

It must also lose its clean gain under frozen wrong-history donors.  A pass on
one internal role is insufficient: the same frozen checkpoints must replicate
on a delayed role before any promotion.

## Claim boundary

Generic triple attention, HyperAttention and functional-ANOVA/Möbius
decomposition are prior art.  C25 is an architecture signal gate with novelty
status `distinct restriction, uncertain contribution`.  A positive result
would justify a full nearest-neighbour review and end-to-end LM implementation;
a failure closes this anchored-purification primitive without tuning.
