# Active baseline source boundary

The existing modules are retained shared controls and upstream adapters. Some
still carry historical method IDs or `v0_lite` metadata defaults; they must not
be relabeled as E-QC/E-FULL or D-QC/D-FULL without a reviewed E0 data version,
new config, current metadata, and a source-specific test.

The active ordinary encoder/decoder family implementation should use the
counterfactual score contract under `experiments/history_response_gap/` and
keep one checkpoint fixed across true/null/wrong scoring. Proposed
architecture code does not belong here or under `systems/` before a replicated
Failure Card.
