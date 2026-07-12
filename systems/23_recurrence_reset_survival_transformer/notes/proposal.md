# C23 proposal — Recurrence-Reset Survival Transformer (RRST)

Status: design formulation; no C23 label or outcome observed.

## Observation → consequence → falsification

**Observation.** C5-R3 established exact candidate recurrence as the only
stable history component: item-only beats D2p in all three seeds, while coarse
category transfer does not.  That item-only score is nevertheless a static
sum of action-weighted inverse-square-root recency.  It cannot represent a
different question: after the candidate last occurred, did the user's later
trajectory preserve or displace that recurrence intent?

**Architecture consequence.** For candidate `c`, let
`tau(c)=max{t: item(h_t)=c}`.  If it exists, construct a candidate-local token
sequence

```text
[RESET(q,c,h_tau,count), h_{tau+1}, ..., h_T, READ(q,c)] .
```

A shared causal Transformer evolves this reset state through only the
post-anchor suffix.  Its centered readout is a bounded correction to the
registered item-only ranking.  Events before `tau(c)` are outside the
attention graph, not merely assigned a learned small weight.  If no candidate
has an exact anchor, the personalized path is structurally zero and the ranker
returns D2p exactly.

**Falsification.** On a request-hash-frozen, previously unused repeat cohort,
RRST must beat the registered static item-only score and every equal-parameter
control across three seeds.  Shuffling only the post-anchor suffix must remove
the learned advantage, while changing masked pre-anchor events must be exactly
irrelevant.  Query masking must remove the learned correction; no-history and
non-repeat requests must be bitwise D2p.  Failure closes this primitive before
dev and before a soft-anchor extension.

## Primitive

For projected token states `x`, a shared causal Transformer `T_theta` gives

```text
z_c = T_theta([a_c, x_{tau(c)+1:T}, r_c])[-1]
u_c = w^T LN(z_c)
d_c = gamma * center_C(I[tau(c) exists] * tanh(u_c)).
```

The stage-A score is

```text
s_c = s_item-only,c + d_c                       if any candidate repeats,
s_c = s_D2p,c                                   otherwise.
```

`s_item-only` is recomputed with the registered C5-R3 semantics:
`0.3*z(D2p)+0.7*z(3*sum exact_event_weight/sqrt(reverse_position))`.
It is an input contract and a binding control, not a tunable channel.  RRST has
one learned primitive: candidate-identity reset of the Transformer attention
graph followed by suffix evolution.

## Stage boundary

Stage A may establish only that exact recurrence should be **stateful rather
than additive**.  It cannot establish non-repeat personalization or a final
paper system.  If and only if stage A passes, stage B may freeze a separate
pre-outcome protocol for a nullable soft last-passage anchor on untouched
non-repeat requests.  No stage-B formula or threshold may be chosen from C23-A
labels.

## Complexity and unified interface

- one primitive and two named pieces: reset-token construction and a shared
  candidate-local Transformer;
- `O(C H^2 d)` in the bounded probe (`H<=50`), with no online LLM call;
- identical code for every dataset; item equality and evidence-presence masks
  are interface fields, never dataset branches;
- query, candidate, strictly-prior history and readout all occur inside the
  Transformer ranking path.
