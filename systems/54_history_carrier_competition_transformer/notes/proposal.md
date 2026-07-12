# C54 proposal — history-carrier competition

Status: pre-outcome mechanics gate.  C53 A features are already exposed but
their labels remain unopened.  This gate cannot read those labels or advance a
utility claim.

## Observation and single law

C53 learned stable list reranking while true- and wrong-history corrections
remained 0.984--0.999 correlated.  Ordinary joint attention permits a direct
`candidate list -> candidate value -> logit` shortcut.

C54 first computes, with shared weights,

```text
d_i(H) = CrossAttn(c_i, [q,H]) - CrossAttn(c_i, [q,NULL]).
```

Candidate self-attention then uses candidate/base states only as `Q/K`, while
its `V` stream is restricted to `d_i(H)`.  Its residual score is centered per
request and added to the frozen strong base.  Bias-free attention makes the
list message exactly zero when the carrier is zero.  This is a change to the
Transformer information-flow law, not a score router or dataset condition.

## Binding reductions

- `independent_carrier`: the same list-attention parameters operate with a
  diagonal mask, so no other candidate can be read;
- `factual_carrier`: candidate competition transports the factual context
  state without factual-minus-null subtraction;
- `raw_candidate`: candidate competition transports raw candidate states and
  is therefore the C53-style history-free shortcut;
- `no_history`: primary carrier, message, and correction are exactly zero.

The first real-data gate trains only the primary and uses same-checkpoint
reductions to ask whether both cross-candidate edges and history contrast are
load-bearing.  It opens no A labels.  A pass authorizes only a new proposal on
a fresh role with separately trained matched controls.

## Stop rule

All three seeds in both domains must reduce mean epoch loss, be deterministic
and candidate-equivariant, preserve exact no-history base, and change at least
5% of orders / 1% of Top-10 sets under each of: diagonal candidate edges,
removal of the null contrast, and matched wrong history.  Any failure closes
this primitive on the exposed mechanics surface; no scale, width, layer,
epoch, seed, threshold, or domain rescue is allowed.
