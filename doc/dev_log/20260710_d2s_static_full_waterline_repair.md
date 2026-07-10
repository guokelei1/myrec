# D2s Complete Static Waterline Repair

Date: 2026-07-10

The final repository audit found a fairness omission after D2h had been
evaluated: D2p showed that train popularity materially strengthens D2t, but
D2h combined only D2t with B0b. Treating D2h as the strongest static baseline
would therefore understate the comparator available to the proposed system.

Doc 21 and the base config were locked before any D2s calibration or dev score.
Train-only calibration selected beta 0.3 for
`D2s = beta*z(D2p) + (1-beta)*z(B0b)`. No model was retrained. Six fixed
true/wrong-history dev evaluations were then run through the shared evaluator.

D2s reaches 0.3416 +/- 0.0004 over three seeds and significantly exceeds D2h
by +0.0064, CI [+0.0037, +0.0090], at the preselected seed. The correct-user
identity effect remains significant for every seed, including same-query
donors, and no-history requests exactly preserve D2p rankings and metrics.

D2s replaces D2h as the binding static baseline. The full-claim 2% relative
target is correspondingly raised from approximately 0.3419 to 0.3485. D2h is
retained as a valid chronological intermediate and ablation.
