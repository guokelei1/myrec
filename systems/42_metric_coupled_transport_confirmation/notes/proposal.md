# C42 weights-preserving metric-coupled confirmation

## Trigger

C41's preregistered semantic-routing primary failed, but its equal-parameter
`coupled_content` control was trained and scored before labels opened. On fresh
C41-A it reached 0.356457 NDCG@10 and beat C38 by `+0.006002`, with a positive
paired interval and all seed/fold signs positive. It also had positive
true-over-wrong and clicked-direction intervals. The machine trigger report is
`reports/pps_c41_coupled_control_diagnostic.json`.

## Frozen primary

C42 primary is exactly C41 `coupled_content`:

```text
T_r(x) = normalize(x + U_r V_r x)
a_jr = softmax_j(<T_r(q),T_r(h_j)>/tau)
p_r = sum_j a_jr T_r(h_j)
q'_r = normalize(T_r(q) + p_r)
d_ir = <T_r(c_i),q'_r> - <T_r(c_i),T_r(q)>
d_i = mean_r d_ir.
```

The same metric closes selection, value, transport, and candidate readout in
each head. The three exact C41 checkpoints are immutable; C42 performs zero
optimizer steps.

## Cohort and controls

C42-A is exactly C38 escrow: 1,200 requests never previously feature-
materialized, scored, or label-opened. Controls are the matching C41
`semantic_routing`, `single_wide_routing`, and `asymmetric_routing` checkpoints,
plus fixed semantic, uniform history, frozen base, and C38 unprojected.

All primary/control scores are produced before A labels open. A0 checks
checkpoint hashes, exact fallbacks, deterministic/permutation behavior,
primary/control activity, true/wrong activity, and closed dev/test. A1 requires
the primary to reproduce stable rent over C38 and the matched controls plus
true/wrong and clicked-direction specificity.

## Interpretation

Passing C42 would establish a replicated Amazon architecture candidate, not a
finished paper claim. The mechanism remains close to tied QKV/metric attention
and Hopfield retrieval, so novelty review and a newly frozen KuaiSearch transfer
are still required. Failure closes metric coupling without retraining rescue.
