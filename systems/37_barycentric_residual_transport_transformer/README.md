# C37 Barycentric Residual Transport Transformer

C37 isolates the sub-primitive that survived C36's label-free gate: one shared
authenticated query-tangent history write plus a candidate-specific residual
whose mean over admitted candidates is exactly zero. It contains no soft trust
coefficient, learned gate, threshold, category branch, or dataset branch.

The primary is trained against three equal-capacity reductions: global-only,
uncentered additive, and relative-only transport. C36-A is an excluded
label-free formulation surface. C37-A is untouched C36 delayed-B and C37
delayed-B is untouched C36 escrow. Dev and test are forbidden.

```bash
/data/gkl/conda_envs/myrec-c37/bin/python -m pytest \
  systems/37_barycentric_residual_transport_transformer/tests -q
```
