# C37 train-only protocol

Selection SHA-256:
`9e80f235543ec821f98bc095a1957b12d3c858aa35ad9a6ea699bd0d3729cbf7`.

Reuse C36's fixed 10,000-request fit. Promote untouched C36 delayed-B to C37-A
and untouched C36 escrow to C37 delayed-B. Hash-select new escrow, structural
roles, and matched wrong-history donors without labels. C36-A is excluded.

Train four modes for seeds `20261041/42/43`, bound to physical GPUs 0/1/2,
with identical rank 16, initialization per seed, one epoch, complete candidate
lists, request order, optimizer, and loss:

- `barycentric_residual_transport` (primary);
- `global_tangent_transport`;
- `uncentered_additive_transport`;
- `relative_surplus_only`.

All twelve fits and A scores must exist before aggregation. A labels remain
closed until every A0 authentication, conservation, natural-subordination,
activity, corruption, determinism, permutation, capacity, and fallback check
passes. D2p and every control share one fold partition/bootstrap. No retry,
scale/temperature sweep, C36-A rescue, delayed-B rescue, dev, or test is
authorized.
