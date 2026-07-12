# C43 cross-domain metric-coupled terminal outcome

C43 closed at KuaiSearch train-internal A1. The authoritative report is
`reports/pps_c43_train_gate.json`, SHA-256
`85a9e347495af87f558e417745f8a31c35d111db6e491deb7eac39c7c54459b7`.

C43 mechanically changed only the frozen LM width from Amazon's 384 to
KuaiSearch's registered 512. It preserved the exact C40/C42 operator, four
heads, rank 16, optimizer, loss, one epoch, 6,000-request fit size, and all
thresholds. C43-A was exactly the union of C37 delayed-B and escrow: 1,200
strict non-repeat requests that had never been feature/score/label-opened.
Proposal lock, G0, and execution lock completed in order. All 18 A0 checks
passed before labels opened; dev/test and qrels remained closed.

The broad history-interaction signal transferred weakly. Seed-averaged
NDCG@10 was 0.595612 for D2p and 0.599736 for multi-head coupling. The gain was
`+0.004124`, CI `[0.000479,0.007820]`, and every seed was positive. One fixed
fold was negative, so even the base comparison missed the full stability gate.

The metric-coupling attribution failed more decisively. `shifted_loop` reached
0.600184 and exceeded the primary by `+0.000448`; `single_wide_coupled` was
effectively tied. Primary-minus-selection-only was `+0.001719`, but its CI
`[-0.000158,0.003767]` crossed zero and one seed/fold was negative. The gain
over fixed semantic attention also had a CI crossing zero.

Most importantly, correct-user specificity did not transfer. True-minus-wrong
history was only `+0.000487`, CI `[-0.002641,0.003776]`; two seeds and one fold
were negative. Clicked-minus-unclicked correction was `-0.000597`, CI
`[-0.005989,0.004877]`. The model therefore acts mainly as generic request
history expansion on KuaiSearch rather than evidence-faithful personalization.

Status is `failed_A1_terminal`. No C43 seed, cohort, width bridge, rank/head,
temperature, loss, scale, epoch, or encoder rescue is allowed. C43 closes
metric coupling as the proposed primitive. The transferable observation is
narrower: query/history interaction can weakly improve D2p, but request-global
history pooling does not reliably distinguish the correct user's evidence.
