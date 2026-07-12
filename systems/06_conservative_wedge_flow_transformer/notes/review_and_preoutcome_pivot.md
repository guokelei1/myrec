# C06 architecture review and pre-outcome pivot

Date: 2026-07-11

No C06 cohort, label, GPU outcome, dev result or test result existed during this
review. The changes below are therefore architecture corrections, not rescue
tuning.

## Rejected primary

The first draft multiplied each event potential by one global Hodge fraction:

```text
d_ij = kappa_j u_ij.
```

This is exactly reducible to an ordinary event-gated centered-unary scorer by
absorbing `kappa_j` into the event weight. It cannot express that one event is
reliable for one candidate but contradictory for another. It is retained only
as a matched control.

The first FP32 low-rank energy formula was also numerically invalid near
coincident factors because it subtracted two large Gram terms. A stress case
with `B=A+1e-4*noise` changed the inferred consistency by orders of magnitude.
The primary now computes centered-factor energy moments in FP64 and must match
an explicit small-graph FP64 oracle.

## Rejected local alternative

Directly gating the original field as `W F W` preserves skew symmetry, but it
can convert a divergence-free cycle into a signed ranking update whenever the
node weights differ. That contradicts the intended semantics: a cycle may
indicate uncertainty, but its orientation must not decide the ranking.

## Current primary

C06 now performs candidate-local Hodge trust on the projected gradient only:

```text
F = G + C
t_i = row_energy(G)_i / (row_energy(G)_i + row_energy(C)_i + eps)
T_ik = (t_i * t_k / 2) * (u_i - u_k)
delta_i = rho * divergence(T)_i
```

The product avoids the zero-gradient singularity of `sqrt(t_i*t_k)`. `T` is
skew, its divergence is zero-sum, each edge is bounded, pure cycles abstain,
and reversing a cycle cannot reverse a preference. Trust is candidate/event
local and therefore cannot be folded into a scalar history-event weight.

This remains a high-risk architecture hypothesis. The same trainable field can
learn to inject cycle energy to manipulate its own trust. A parameter-matched
direct learned candidate gate is therefore mandatory; C06 earns an
architecture claim only if the Hodge-derived gate beats that control as well as
`t=1`, global-event trust, centered attention, pairwise-additive and
MIR/SetRank-style controls.

Current authorization remains CPU structural contracts only.
