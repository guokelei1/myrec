# C52 exposed formulation outcome

C52 passed every label-free structural check in both domains.  The fixed
history KRR bias changed query-concept attention for essentially every
candidate, changed 88.3%/100% of complete Kuai/Amazon orders, and changed
44.3%/87.7% of Top-10 sets.  This repairs C26's rank-inactivity failure.  The
utility direction nevertheless failed.

| domain | primary | raw LM base | linearized token KRR | token softmax | pooled plain KRR | best pooled |
|---|---:|---:|---:|---:|---:|---:|
| KuaiSearch | 0.304532 | 0.300870 | 0.304779 | 0.312551 | 0.310162 | token softmax 0.312551 |
| Amazon-C4 | 0.259458 | 0.253202 | 0.259986 | 0.256702 | 0.268905 | posterior 0.277001 |

The primary improved nominally over raw LM base by `+0.003661` on Kuai and
`+0.006256` on Amazon, but both intervals crossed zero.  More decisively, the
nonlinear query-concept allocation was worse than its first-order linearized
reduction in every hash fold on Amazon and nominally worse on Kuai.  It also
lost to pooled KRR/softmax controls in both domains.  Kuai clicked direction
and true-minus-wrong specificity crossed zero; Amazon specificity was positive
but did not rescue the nearest-control failures.

Decision: `failed_formulation_terminal`.  Close C52 before training or fresh
reserve.  The result closes history KRR as a nonlinear query-token attention
bias, not token-level modeling in general.  It also shows that rank activity
alone is no longer the bottleneck: semantic direction and strong-base
complementarity are.

Authoritative report SHA-256:
`4cadfc53d6b44fc7e594cc55c40b7a20b5a4e3fe4e0a774b28d6603ba373b8c0`.
