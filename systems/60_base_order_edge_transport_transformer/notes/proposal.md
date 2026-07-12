# C60 proposal — base-order edge transport

Status: pre-outcome exposed-role formulation.  C59 showed that fixed semantic
history is user-specific but that an independently standardized history score
overwrites the strong query base.  C60 asks whether the residual interface,
rather than the semantic direction, is the failure.

## Primitive

Let `s_1 >= ... >= s_C` be candidates in the shared strong-base order and let
`e_i` be the frozen Transformer history evidence correction.  For every
adjacent edge `(i,i+1)`, define base margin `g_i=s_i-s_{i+1} >= 0` and evidence
for correcting the base order `d_i=e_{i+1}-e_i`.  The baseline and
history-conditioned Bradley-Terry upset probabilities are

```text
p0_i = sigmoid(-g_i)
p1_i = sigmoid(-g_i + d_i)
rho_i = clamp((p1_i - p0_i) / (1 - p0_i), 0, 1)
t_i = rho_i g_i.
```

The final update is the graph divergence of this edge transport:

```text
delta_i     -= t_i
delta_{i+1} += t_i
s'_i = s_i + delta_i.
```

Thus history may only move score mass from the base-preferred candidate toward
its immediate challenger, only when evidence increases the upset probability,
and by no more than the base margin on that edge.  The request score sum is
preserved exactly; zero evidence returns the base exactly.  Adjacency is used
at every rank, not at an evaluation cutoff.  No dataset, category, query type,
rank-k, temperature, coefficient, or learned gate exists.

## Binding controls

- `signed_adjacent`: admits evidence in both directions, testing the one-sided
  error-correction law;
- `hard_adjacent`: transfers the full margin only when `d_i > g_i`, testing
  whether soft normalized confidence pays rent;
- `history_axis_adjacent`: same transport with C59 ordinary history-axis
  evidence, testing candidate-budget dependence;
- `raw_query_adjacent`: same transport with history-free evidence;
- `wrong_history`: primary transport with C59 wrong-history evidence;
- `direct_additive`: the terminal C59 score, exposing whether the new write
  interface—not semantic scoring alone—causes any gain.

## Gate and evidence boundary

C59's 1,200 holdout labels are already exposed, so C60 is only a formulation
screen.  A0 first requires hash alignment, exact zero-evidence identity,
candidate permutation, determinism, score conservation, edge-capacity bounds,
rank activity, and wrong-history load-bearing behavior.  Only then is A1
computed on the exposed labels.

A1 requires a positive-CI gain over base, wrong history, and every control,
with every hash fold positive.  Passing would authorize—but not validate—a
fresh-role trainable compare-exchange Transformer.  Failure closes this edge
law.  There is no threshold, depth, neighborhood, scale, sign, or cohort
rescue.
