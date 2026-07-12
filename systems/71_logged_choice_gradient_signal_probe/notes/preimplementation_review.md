# C71 preimplementation review

Decision: `authorize_train_only_parameter_free_signal_gate`.

- C71 tests a new information object, not another learned attention law.
- Targets come from outside the entire historical packed pool; a preliminary
  label-free audit found 20,700 eligible strict-nonrepeat requests with at
  least two linked episodes and complete frozen embeddings.
- Historical selected items are inferred from the recipient's strictly past
  history; source candidate label fields are neither necessary nor authorized.
- The primary and controls are parameter-free, so no seed, checkpoint, epoch,
  or capacity selection exists.
- Candidate/hash, deterministic, permutation, no-history, gradient activity,
  correction activity, and true/wrong activity gates precede target labels.
- C71 remains KuaiSearch-only and cannot relax C70's two-domain requirement.

Forbidden: packed target reuse, source episode labels, target labels before
A0, alternate temperature/normalization/scale, donor rematching after scores,
fresh cohort retry, dev/test/qrels, or C70 implementation based on a failure.
