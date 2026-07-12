# 2026-07-12 — C52 query-concept attention terminal

C52 was the first post-C51 intervention at token representation formation
rather than another pooled confidence formula.  It used a history KRR
projection only as a candidate-specific bias over query-token concepts and
kept all values in the frozen LM semantic path.  Four GPUs encoded and scored
600 KuaiSearch and 300 Amazon-C4 exposed requests under one fixed operator.

The mechanism was fully load-bearing: both domains passed determinism,
permutation, empty-history, finite-value, candidate-hash, reserve, and
dev/test-lock checks; it changed 44.3% and 87.7% of Top-10 sets.  Utility still
failed.  C52 reached 0.304532/0.259458 versus raw LM bases
0.300870/0.253202, but both base intervals crossed zero.  It lost to its
linearized token reduction in both domains and to the strongest pooled
controls.  Kuai specificity was unstable.

A post-terminal waterline audit found D2p at 0.603050 on the same Kuai cohort,
roughly twice the raw-BGE NDCG.  Adding C52 unchanged to D2p gave a small
`+0.001365` with a zero-crossing interval; the simpler linearized reduction
gave `+0.001912`.  This exposes an important search correction: C47--C52 were
valid signal probes, but their raw-BGE improvements cannot be described as
good rankers.  Future architecture gates must retain or internally reproduce
the strong D2p path from the outset.

C52 closes nonlinear KRR query-concept allocation before training/fresh data.
The next problem is no longer how to force history to move ranks; it is how to
produce stable, user-specific direction while preserving the strong base.
