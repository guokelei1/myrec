# 2026-07-12 — C60 safe residual interface, formulation failure

C60 replaced C59's independent history score with one-sided conservative flow
on adjacent strong-base edges.  Eight tests and two locks preceded GPU A0.
All mechanics passed; labels on this role were already exposed by C59, so the
result is formulation-only.

The edge law reduced C59's `-0.070103` base deficit to `-0.001672` with a CI
crossing zero.  It significantly beat wrong-history, history-free, signed,
and direct-additive controls, but did not beat base or the same edge law driven
by ordinary history-axis evidence.  The authoritative report is
`reports/pps_c60_base_order_edge_transport_gate.json`, SHA-256
`45e85db975610555c8247f9c1f2fe0790e9ef57f3c3ea2c037adec3a0cad6c7d`.

Decision: close the fixed evidence/edge combination.  Do not tune adjacency,
capacity, threshold, depth, or scale on the exposed role.  The only justified
new hypothesis is supervised Transformer estimation of a history likelihood
ratio for whether a specific base edge is wrong, evaluated on an untouched
role with C60 as a fixed control.
