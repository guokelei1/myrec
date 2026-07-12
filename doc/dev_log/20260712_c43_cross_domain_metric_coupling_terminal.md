# C43 cross-domain metric coupling terminal result

C43 tested the exact C40/C42 metric-coupled operator on KuaiSearch without
architecture or hyperparameter tuning. The 384/512 LM-width difference was
handled only by mechanically resizing the registered tensors. C43-A combined
C37 delayed-B and escrow, both previously feature/score/label unopened.

All structural and access gates passed. Four equal-parameter modes in three
seeds trained for 375 steps each; every parameter received gradients, all
fallbacks were exact, candidate-set hashes matched, and A labels stayed closed
until A0 passed. Dev/test and qrels remained closed.

The primary beat D2p by `+0.004124` NDCG@10 with a positive interval, but one
fixed fold was negative. It did not beat the shifted-loop or single-wide
controls, and its advantage over selection-only was not statistically stable.
True history tied wrong-user history and the clicked correction direction was
near zero. Thus Amazon's coupled-checkpoint result does not support a general
metric-coupling primitive.

Decision: close C43 and metric coupling without rescue. The next architecture
must eliminate request-global history pooling before the candidate decision.
However, prior candidate-relative cones, generic set competition, pairwise
contests, and strict triadic OT already failed selectivity, utility, or
tractability gates. A successor must change the attention information object
without reducing to those mechanisms and must use a newly isolated cohort.
