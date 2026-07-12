# C35 train-only protocol

Reuse C34's fixed 10,000-request fit and promote its untouched delayed-B to
C35-A and untouched escrow to C35 delayed-B.  Hash-select a new C35 escrow and
fresh structural controls without labels.  Train all four modes for seeds
20260981/82/83 with identical rank 16, one epoch, complete candidate lists,
request order, optimizer, and equal listwise/direction losses.

A labels remain closed until all relative-support, activity, matched-control,
corruption, determinism, permutation, and fallback checks pass.  The one fixed
fold partition is shared by D2p and every control.  No second attempt,
threshold/temperature sweep, C34-A reuse, delayed-B rescue, dev, or test is
authorized.
