# C67 — Cross-Validated Fast-Weight Transformer

C67 tests one architecture primitive: a history event may update the
request-local fast weights only to the extent that its exact one-step update
improves reconstruction of other held-out history events. Query and candidates
are excluded from the write and may only read the frozen request-local learner.

Plain test-time training, self-validated writing, and first-order gradient
agreement are binding equal-parameter controls. C67 claims neither that
test-time training is new nor that a synthetic pass establishes ranking
utility.
