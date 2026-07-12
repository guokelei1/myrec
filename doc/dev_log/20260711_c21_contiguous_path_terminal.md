# 2026-07-11 — C21 contiguous-path signal terminal

C21 was introduced as an architecture-precondition audit after C18--C20: before
building another Transformer, test whether real train-only frozen states contain
the exact temporal relation the architecture would assume.

The label-free 9,000/3,000 split, five matched operators, three seeds, two
epochs, full candidate sets, thresholds and corruptions were locked before C21
opened C06's compact fit labels.  The formal run completed without engineering
failure, but the contiguous-path primary averaged `-0.0000643` NDCG@10 versus
D2p and lost to one-step by `-0.0006272` (paired CI entirely below zero).
Wrong and shuffled histories did not remove the write direction.  The operator
was active, centred and deterministic, so this closes the path law rather than
an implementation.

Do not rerun, tune the horizon, select favourable history lengths or turn this
probe into a Transformer.  One-step and pooled controls had small positive point
estimates but CIs crossed zero; they are diagnostics, not successor claims.
The next candidate must either change how history representations are learned
inside the LM or make proven exact recurrence load-bearing.  Further geometry
over the same frozen D2 states is not authorized by C21.

Authority: `reports/pps_c21_train_signal_gate.json`.
