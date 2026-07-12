# Pre-outcome G0 abort

The first locked G0 invocation stopped before feature materialization because
`materialize_g0.py` looked for the registered item-embedding keys under
`model.*`; the frozen D2 configuration stores them under `encoder.*`.

The failed invocation created no G0 report or feature arrays, did not train a
model, did not observe a C24 ranking outcome, and did not open internal-A,
escrow, dev or test labels.  `proposal_lock.json` is retained as the immutable
v1 audit record.  The two key lookups, lock paths and lock identifier are the
only execution changes admitted to the superseding pre-outcome v2 lock.
