# C47 fresh two-domain S0 outcome

C47 completed its frozen fixed-operator gate on 2026-07-12.  Feature and score
materialization remained label-free through A0; candidate hashes, deterministic
rescore, candidate/history permutation, finite state, support bounds,
contraction, and exact no-history fallback all passed on 600 KuaiSearch
strict-nonrepeat requests and 300 Amazon-C4 requests.  Only then were the two
locked train-internal label roles opened.  No dev/test record, label, or qrels
was read.

## Result

| domain | posterior NDCG@10 | vs query base | vs plain ridge | vs softmax | true vs wrong |
|---|---:|---:|---:|---:|---:|
| KuaiSearch | 0.307291 | +0.006421, CI crosses 0 | -0.002871, all folds negative | +0.000324, CI crosses 0 | +0.005171, CI crosses 0 |
| Amazon-C4 | 0.277001 | +0.023799, CI [0.001665, 0.046113] | +0.008096, CI crosses 0 | +0.000113, CI crosses 0 | +0.031113, CI [0.013282, 0.048860] |

Amazon clicked-correction direction and true-minus-wrong specificity were both
positive with positive intervals.  KuaiSearch had positive point estimates for
both but both intervals crossed zero.

## Decision

`failed_S0_terminal`: close C47 before any trainable PSRT implementation.

The common history subspace remains useful: plain ridge reached 0.310162 on
KuaiSearch and posterior support retained a significant base and wrong-history
margin on Amazon.  The C47-specific self-support contraction did not pay
incremental rent.  It stably suppressed useful Kuai ridge writes and was
indistinguishable from ordinary softmax attention on both domains.  No support
exponent, ridge, temperature, scale, cohort, sign, or dataset-specific gate may
be tuned as a C47 rescue.

The authoritative report is `reports/pps_c47_signal_gate.json`.
