# C66 proposal — canonical mechanical continuation

Status: pre-outcome numerical continuation.  C65 read no labels and produced no
trained checkpoint.

C65 passed every G0 check except candidate permutation.  Its factual and NULL
listwise states were individually well behaved, but subtracting near-equal
states and applying LayerNorm amplified caller-order reduction differences to
`3.64e-5`.

C66 makes exactly one change:

1. derive a stable signed-int63 key from SHA-256 of each item ID;
2. sort valid candidates by that key before all shared LM/joint Transformer
   branches;
3. run the unchanged C65 factual, NULL, and wrong-history computation;
4. restore scores, corrections, and states to caller order.

Canonicalization has no trainable parameters and sees no label, rank position,
base score, category, query, or history.  All C65 modes, model initialization,
last-two-layer adaptation, stop-gradient, wrong-neutrality weight, candidate
sampling, split, optimizer, epochs, seeds, thresholds, and fp32 scoring remain
identical.

If G0 fails, C66 closes.  If G0 passes, C66 may execute the previously
unconsumed C65 training seeds.  Validation labels remain closed until A0.
