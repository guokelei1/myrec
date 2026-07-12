# C41 semantic-carrier routing terminal outcome

C41 closed at train-internal A1. The authoritative report is
`reports/pps_c41_train_gate.json`, SHA-256
`4b49da43dacc91c63a05a1bc743acf57d197426d71a9d0148701d4b19f24e20c`.

Proposal lock, four-way label-free encoding, G0, and execution lock completed
in order. Three GPUs trained four 49,152-parameter modes for 373 steps each.
All 29 A0 checks passed: paired capacity/initialization, active gradients,
updated parameters, raw semantic profiles within `8.94e-8`, simplex attention,
and exact permutation/fallback contracts. C41-A was untouched C38 delayed-B;
C41 delayed-B, dev, and test stayed closed.

The primary was useful but not best. NDCG@10 was 0.222010 for base, 0.259971
for fixed semantic attention, and 0.305617 for semantic routing. Primary-minus-
base was `+0.083607`, CI `[0.061037,0.106209]`; primary-minus-fixed was
`+0.045646`, CI `[0.034508,0.056975]`. True-minus-wrong was `+0.147505`, CI
`[0.127310,0.167873]`, and clicked direction was positive.

C41 failed because C38 unprojected reached 0.350455 and the equal-parameter
`coupled_content` control reached 0.356457. They exceeded the primary by
`+0.044838` and `+0.050840`; `single_wide_routing` also nominally exceeded it.

The pre-trained coupled control produced the first promising architecture
result in this branch. A post-terminal diagnostic on now-open C41-A found
coupled-minus-C38 `+0.006002`, CI `[0.001014,0.011102]`, with all three seeds
and fixed folds positive. Coupled true-minus-wrong was `+0.029935`, CI
`[0.018765,0.041242]`, and clicked direction was strictly positive. This cannot
promote a C41 control to primary, but it defines an exact C42 confirmation.

C41 remains `failed_A1_terminal`; no rescue is allowed. C42 may use the still-
unmaterialized C38 escrow, keep metric-coupled multi-head transport unchanged
as primary, and train strong reductions before labels open. A positive C42 must
reproduce the C38 margin and true/wrong specificity; otherwise the mechanism
closes.
