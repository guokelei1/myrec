# Symbolic reduction audit

## Exact decomposition

For tied linear decoder embeddings `E_v`, bias `b_v`, and
`A(h)=log sum_v exp(E_v^T h+b_v)`, each token likelihood ratio is

```text
g_i,e,t = E_c^T (h^H_i,e,t - h^0_i,t)
          - [A(h^H_i,e,t) - A(h^0_i,t)].
```

Define `DeltaA_i,e,t=A(h^H_i,e,t)-A(h^0_i,t)`.  Candidate centring gives

```text
P_C g = P_C[E_c^T Delta h] - P_C[DeltaA].
```

In C11, the predictor states did not contain a candidate prefix, so `DeltaA`
was candidate-common and the second term vanished exactly.  In C12, both states
depend on `c_i,<t`; `DeltaA_i,e,t` is not generally constant over candidates.
The normalized-prediction term therefore survives centring.

## Constructive witness against hidden similarity

Use a two-token vocabulary with embeddings `E_x=(1,0)` and `E_y=(0,1)`, zero
bias, and two candidate prefixes that currently predict the same token `x`.
Let both NULL states be `(0,0)`, while one event produces
`h^H_A=(0,a)` and the other prefix/event interaction produces
`h^H_B=(0,-a)`, for `a>0`.

For both candidates,

```text
E_x^T(h^H-h^0) = 0,
```

so the target-embedding hidden-similarity control ties them.  But

```text
g_A = log 2 - log(1+exp(a)),
g_B = log 2 - log(1+exp(-a)),
```

which differ and remain non-zero after candidate centring.  The distinction is
entirely the candidate-specific vocabulary normalizer.  This is a valid
function-class witness, assuming the shared Transformer uses candidate prefixes
to produce the two event-conditioned states.

## Witness against scalar paired final-logit delta

Let two candidate token embeddings be orthogonal `E_1` and `E_2`.  Candidate A
has token gains `[a,-a]`; candidate B has `[-a,a]`.  Both have summed likelihood
ratio zero, so a paired control that adds `sum_t g_i,t` after the ranking head
ties them.  The vector writes are respectively proportional to
`tanh(a)(E_1-E_2)` and its negative, so a non-rank-one hidden readout can
distinguish them.  This separates C12 from the registered scalar paired control,
not from an unrestricted cross-encoder capable of simulating any function.

## Witness against ordinary event-only cross-attention

With one history event, ordinary candidate-query/event-key attention assigns
weight one to that event.  If values are event-only, every candidate receives
the same value and candidate centring removes the write.  The prefix witness
above produces different normalized likelihood ratios for two candidates even
with that single event.  A control whose values are themselves a full
candidate/event cross-encoder is no longer ordinary cross-attention and must be
registered separately as a stronger capacity control.

## Conditions that collapse C12

C12 reduces or approximately reduces in any of these cases:

1. `t=1` has no candidate prefix and no other token position is load-bearing;
2. the LM ignores `c_i,<t>`, making history/NULL states candidate-independent;
3. `h^H_i=h^0_i+delta_e` and
   `A(h^0_i+delta_e)-A(h^0_i)` is constant over candidates;
4. the decoder log-partition is locally affine with nearly identical slope over
   all candidate-prefix states, making `P_C DeltaA` negligible;
5. logits rather than log probabilities are used, the partition term is
   dropped, or it is replaced by a candidate-common approximation—each gives
   hidden similarity exactly;
6. target tokens leak through a non-causal mask, producing trivial copy rather
   than predictive evidence;
7. event integration becomes uniform and position-blind, collapsing eventwise
   structure to pooling;
8. token embeddings/write/readout collapse to rank one, reducing the hidden
   write to a scalar delta;
9. the monotone exact coordinate dominates all transfer requests, leaving the
   cross-item primitive unevaluated.

## Mandatory diagnostics if revisited

- candidate-centred RMS of `DeltaA`, reported separately from target-logit RMS;
- fraction `||P_C DeltaA|| / ||P_C g||`, with a predeclared non-collapse floor;
- correlation and affine regression residual versus the hidden-similarity
  control at every event/token position;
- contribution by `t=1` versus `t>=2` and a strict causal target-leak test;
- event/token matrix effective rank and final write singular values;
- same-checkpoint removal of the partition term.
