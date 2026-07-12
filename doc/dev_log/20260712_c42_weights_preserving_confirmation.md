# C42 weights-preserving confirmation

C42 evaluated the exact C41 metric-coupled checkpoints on untouched C38 escrow
without retraining. The 1,200-request cohort was disjoint from all previously
opened C38/C39/C41 feature cohorts. Feature construction read only
`records_train_blind.jsonl`; proposal/G0/execution locks preserved zero label
access before scoring, all A0 checks passed, and dev/test stayed closed.

The strongest architecture signal replicated: coupled transport reached
0.333347 NDCG@10 versus 0.323097 for C38 and 0.222629 for base. The gains over
C38 (`+0.010250`, CI `[0.004672,0.015830]`) and base (`+0.110718`, CI
`[0.085689,0.135329]`) were positive in every seed and fold. True history also
beat a matched wrong-user history by `+0.035234`, CI
`[0.024063,0.046410]`, with every seed/fold positive.

The all-conditions gate still failed. Coupled transport's nominal gains over
semantic and asymmetric multi-head routing were positive in every seed but
their confidence intervals crossed zero, with a negative third hash fold.
Thus this is evidence against cohort-specific fitting and for the broader
query-conditioned history-interaction family, but not sufficient attribution
to metric coupling. It also remains Amazon-internal evidence, not
cross-dataset evidence.

Decision: close C42 without rescue. Do not continue Amazon tuning. The next
design step must discriminate coupling from its close routing controls under a
newly frozen mechanism test and, before any paper claim, test transfer on
KuaiSearch using the same label/mask/evaluator contract.
