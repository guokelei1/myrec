# C80 terminal real-gate outcome

Decision: `close_c80_before_fresh_labels_and_end_architecture_search`.

All three registered seeds trained the primary and four controls to their fixed
two-epoch final checkpoints.  Every mode had 33,360,384 trainable parameters;
all 15 fits were finite, all loss windows decreased, backbone/head gradients
were active, frozen anchors were unchanged, candidate permutation and repeated
scoring were exact, and no-history scores equalled the protected base exactly.

The binding failure was the primary's event-permutation contract.  Against the
frozen `2e-6` tolerance, maximum score errors were:

| Seed | `triadic_set` event-permutation max abs error |
|---:|---:|
| 20266001 | 0.0318687 |
| 20266002 | 0.0684173 |
| 20266003 | 0.0393717 |

Thus all three seeds failed before outcome labels.  The 365-request fresh role
remains entirely unopened and no NDCG, control comparison, or utility claim is
available for C80.

A post-terminal, label-free diagnostic checked the first request.  True and
shuffled inputs contained identical token multisets for all 101 candidates and
none hit the sequence-length ceiling.  Re-evaluating seed 20266001 in float32
reduced the maximum error to `3.33786e-6`, but this still exceeded the frozen
tolerance.  The construction is algebraically event-set equivariant; the
pretrained Transformer's finite-precision reductions are not numerically exact
under the registered implementation.  This diagnostic cannot authorize an
fp32/canonicalization/tolerance rescue.

C80 is the final candidate.  There is no C81, no fresh-label opening, and no
new architecture or mechanical continuation.  The authoritative report is
`reports/pps_c80_amazon_real_gate.json`; the C01--C80 causal retrospective is
mandatory and follows this closure.
