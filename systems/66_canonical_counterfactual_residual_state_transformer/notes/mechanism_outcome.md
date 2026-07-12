# C66 outcome

Status: **failed terminal at label-free A0**.

Canonical item-ID serialization repaired C65 exactly.  Before training and for
all 12 trained seed/mode checkpoints, candidate permutation, determinism,
no-history fallback, and repeat fallback errors were all zero.  The wrapper
added no parameter and every intended gradient group was active.

The scientific mechanism still failed.  The primary changed `20.67%--30.25%`
of complete orders and `1.50%--2.00%` of Top-10 sets relative to the strong
base, so it was not inert.  Replacing correct history with a matched wrong
history changed `14.25%--20.67%` of complete orders but only `1/10/7` of 1,200
Top-10 sets (`0.08%/0.83%/0.58%`), below the preregistered 1% requirement in
every seed.  Several mode/seed loss trajectories also failed the frozen
end-window decrease check, although all values and gradients remained finite.

A0 therefore closed before validation labels.  C65--C66 establish a useful
negative boundary: subtracting a stopped NULL internal state and penalizing a
wrong-history residual removes the explicit generic output path, but does not
make user history load-bearing at ranking depth.  Canonicalization fixed only
numerics; it did not rescue the architecture.

No validation, fresh role, Amazon, dev, test, or qrels outcome exists.  No
wrong-loss weight, LM depth, epoch, precision, seed, threshold, history length,
or candidate-sampling rescue is authorized.
