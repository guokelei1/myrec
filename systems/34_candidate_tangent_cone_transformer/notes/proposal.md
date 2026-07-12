# C34 proposal: candidate-specific tangent-cone attention

Status: pre-outcome draft; immutable after proposal lock.

## Observation and architecture consequence

C32 and C33 reproduced a positive direction across independent cohorts, but a
single transported query was statistically weak and did not establish rent
over unprojected transport.  The candidate correction also remained poorly
aligned with clicked versus unclicked candidates.  The new hypothesis is that
the missing object is candidate-specific evidence admission, not another
query-global projection or larger correction scale.

For adapted unit states `q`, candidate `c_i`, and causally authenticated event
`h_j`, define query-centred tangent states

```text
v_i = c_i - <c_i,q>q
u_j = h_j - <h_j,q>q
k_ij = <normalize(v_i), normalize(u_j)>
g_ij = ReLU(k_ij)
t_i = sum_j g_ij u_j / (1 + sum_j g_ij)
q_i^H = normalize(q + t_i)
d_i = 2 [<c_i,q_i^H> - <c_i,q>].
```

`1` in the denominator is a fixed null unit, not a learned gate.  When all
events are orthogonal/opposed to candidate `i`, `g_i*=0`, `t_i=0`, and its raw
history write is exactly zero.  Positive support has a fixed geometric sign;
there is no free scalar head or pair MLP.  A shared rank-16 residual adapter is
the only trained representation change.  Candidate corrections are centred
before being added to D2p.  Exact recurrence returns item-only; absent query,
history, or authenticated evidence returns D2p.

## Matched reductions

All modes have identical tensors, initialization, optimizer, data, loss, and
request order:

- `candidate_tangent_cone` — primary law above;
- `standard_target_attention` — same tangent states and candidate-specific
  transported queries, but history is forced through row softmax;
- `global_tangent_transport` — one candidate-shared C32-style tangent write.

The primary must beat both.  Otherwise any gain belongs to ordinary target
attention or the already known global transport direction.

## Isolation and falsification

C34 fit/A/B/escrow and structural roles are all newly selected, label-free,
and disjoint from every C32/C33 target role.  A prior wrong-history donor may
enter C34 because its own query/candidate target surface was never materialized,
scored, or labeled; its prior use exposed only label-free history as a
corruption.  G0 first
checks strict-past authentication.  A0 requires exact fallbacks, determinism,
permutation, rank activity, candidate-distinct writes, an actual zero-support
surface, wrong-history response, and material differences from both controls.
A1 requires positive-CI utility over D2p and each control, with every seed and
fixed fold positive.  Failure closes this cone law without parameter, ReLU,
temperature, layer, or cohort tuning.  Passage authorizes only a separately
frozen delayed-B gate; dev/test remain closed.
