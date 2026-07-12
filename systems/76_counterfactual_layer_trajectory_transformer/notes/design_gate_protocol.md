# C76 data-free design-gate protocol

Status: pre-outcome protocol.

## D0 — algebra and mechanics

Three fixed seeds instantiate the primary and four parameter-matched controls.
Before optimization, every seed must pass:

- same factual/cut token IDs and positions;
- exact factual=cut trajectory with absent history;
- history-cut Q/C state equivalence to a null-history forward;
- protected-base zero gradient;
- active adaptive-LM and trajectory gradients;
- finite values, deterministic rescore, and candidate permutation;
- nonzero Q, C, and H trajectory coordinates on a hand witness;
- output changes when earlier layer tokens are removed;
- exact repeat/item-only and no-history/base contracts;
- equal trainable parameter counts across modes.

## D1 — frozen synthetic shift

The generator uses discrete raw tokens, not pooled vectors.  A query names one
attribute; history tokens reveal the user's preferred value for that attribute;
the positive candidate contains that value.  Distractors match query-only,
history-only, or an irrelevant attribute.  A train-only query-candidate
nuisance token reverses on validation.  History order is exchangeable.

Every mode trains for the same fixed steps, optimizer, batches, and seeds.
The primary must, in every seed:

- improve supported-nonrepeat accuracy over the history-free base by `>=0.10`;
- reach supported accuracy `>=0.75` and repeat/no-history accuracy `>=0.95`;
- lose at least 70% of its clean supported margin under wrong history or query
  mask;
- retain at least 80% of clean margin under event permutation;
- change at least 5% of supported candidate orders relative to base;
- beat `final_logit_delta`, `final_hidden_delta`, `factual_trajectory`, and
  `ordinary_full` by `>=0.02` worst-stratum accuracy, or be closed for lack of
  architecture rent.

The last comparison is deliberately binding: the constructed nuisance gives
the provenance-restricted path a falsifiable reason to generalize.  Passing D1
does not establish real PPS utility; it permits only a new real execution lock.

No seed, width, depth, nuisance, threshold, steps, loss, or generator retry is
allowed after the proposal lock.
