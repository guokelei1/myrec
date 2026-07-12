# C35 pre-lock implementation review

Decision: provisionally accept for the minimal train-only gate.

The candidate-axis subtraction directly addresses C34's label-free failure:
absolute BGE cosine has a large positive common level, so `cos>0` cannot mean
discriminative support.  Subtracting each event's candidate-set mean removes
that common level before rectification.  It is a new attention normalization,
not a tuned cutoff, dataset slice, score router, or added capacity.

The mechanism is still close to rectified attention, Slot Attention, and
candidate-aware user modeling.  Therefore activity or sparsity alone earns no
claim.  The primary must beat the exact absolute predecessor, generic
candidate-axis softmax, and global transport under identical parameters and an
untouched A cohort.  A KuaiSearch pass would still require cross-dataset tests.
