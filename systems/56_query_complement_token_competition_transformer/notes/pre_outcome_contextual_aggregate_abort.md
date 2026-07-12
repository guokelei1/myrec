# C56 v2 contextual-aggregate pre-outcome abort

All four frozen-LM contextual shards completed, and their item/query coverage,
uniqueness, and file hashes passed.  The aggregate manifest nevertheless
received `failed` because the same expected sentinel
`fit_labels_read: false` was again included in an all-true check dictionary.

No model was initialized or trained, no train/holdout label was read during
contextual materialization, and no score or ranking outcome existed.  The v2
selection and 8.2 GiB of hash-verified contextual shard files remain immutable.

The v3 mechanical supersession reuses those exact shard files, writes a new
manifest whose positive check is `fit_labels_closed: true`, and uses disjoint
v3 model/artifact paths.  No architecture, data row, token value, split,
training setting, seed, threshold, or label boundary changes.
