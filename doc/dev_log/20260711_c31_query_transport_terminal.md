# C31 authenticated collaborative query transport terminal

C31 tested a new request-level architecture instead of another dataset router
or candidate-local scalar head.  A frozen BGE Transformer embedded query,
strictly authenticated history, and candidates in one space; a 16,384-parameter
shared low-rank adapter deformed that space and one history profile transported
the query for all candidates.

Protocol integrity held.  C30-A was used only for hypothesis formulation and
was excluded from formal fit/gates.  Formal A was former C29 delayed-B, never
previously feature-materialized, scored, or labeled.  Proposal and execution
locks were frozen before their authorized stages.  Fit candidates, D2p scores,
and authentication rows reproduced C29 exactly.  No delayed-B/escrow/dev/test
artifact was opened.

G0 passed 5/5.  Three seeds completed one epoch (183 steps each).  A0 passed
19/19, including exact candidate permutation and all registered fallbacks.  A1
was directionally positive but terminal under the preregistered rule:

- mean primary-minus-D2p: +0.0017455;
- 95% bootstrap CI: [-0.0010277, +0.0044739];
- seed deltas: +0.0025424 / +0.0012590 / +0.0014351;
- fold deltas: -0.0013161 / +0.0042609 / +0.0027537;
- clicked-direction point: +0.0009417, interval crossing zero.

This is materially better than the earlier near-zero or negative candidates:
the request-level query displacement generalized in sign across seeds, but its
fixed geometry was not stable across request partitions.  A post-terminal
six-operator audit on now-open A isolated a plausible architectural cause.  C31
computes event attention in the old semantic space and writes both the profile
component parallel to the query and its candidate-relative tangent component.
Using adapted-space attention plus tangent-only spherical transport gave
+0.002506 and was positive in every seed, while remaining nonsignificant.  A
post-audit correction found that the first diagnostic had changed the hash-fold
seed across variants; with the formal fixed partition, one fold was still
negative (-0.001357).  This correction occurred before choosing any successor
after C32 and does not alter the formal C31 gate.

Next action: freeze a new candidate on former C31 delayed-B.  Keep capacity,
temperature, scale, losses, fit data, and training budget fixed; change only the
transport geometry to adapted-attention tangent projection.  C31 itself remains
closed and must not be retuned.
