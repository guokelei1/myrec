# C33 pre-lock implementation review

Decision: accept as confirmation, not as another architecture.

This design avoids two invalid next steps.  It does not change the failed C32
fold seed, and it does not consume C32 delayed-B or escrow.  All new outcome
roles come from a fresh reserve.  It also rejects the unsupported mid-layer
variant, which failed to beat C32 despite greater Transformer execution.

The tangent and unprojected modes have exactly the same parameter tensors and
initial state for each seed.  Both use the frozen BGE Transformer as the
load-bearing representation and the same end-to-end ranking path.  The paired
comparison therefore attributes any stable margin to the orthogonal projection
rather than capacity, data, optimization, or a score router.

The gate is intentionally harder than C32: positive D2p utility is insufficient
unless tangent also beats the unprojected nearest reduction.  No result from
the already-open C32-A cohort is a C33 gate input.

A pass would establish fresh within-KuaiSearch mechanism replication, not
cross-dataset validity.  Amazon-C4/JDsearch transfer remains a separate gate;
no dataset-specific branch, category slice, or query-type rule is permitted.
