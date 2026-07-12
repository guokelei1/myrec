# C75 staged train gate

C75 uses the same C64/C74 exposed-fit 4,800/1,200 split for direct comparison.
No C74 checkpoint or score is reused.  The role is formulation-only and cannot
support a final paper claim.

G0 is label-free and must establish backbone byte invariance, forced eval,
zero backbone gradients, route gradients, equal mode capacity, rank activity,
wrong-history response, exact fallbacks, deterministic scoring, permutation,
and candidate hashes.

After the execution lock, only the 4,800 training labels may be read.  A
256-request anchor is selected by request-ID hash before training.  Its eight
candidate positions are sampled once by the registered seed.  Every mode is
evaluated on those identical anchor examples at initialization and after the
fixed final checkpoint.  `final_loss / initial_loss <= 0.995` in every mode and
seed is binding; arbitrary first/last minibatch windows are descriptive only.

Validation scoring is label-free.  A0 binds primary/base, true/wrong, and all
three control activity thresholds plus backbone hashes and exact contracts.
Only A0 pass permits A1 labels.  A1 uses the shared NDCG implementation,
candidate-set hash, paired bootstrap, fixed folds, and all-seed signs.

No hyperparameter, seed, split, anchor, threshold, or checkpoint selection can
change after lock.
