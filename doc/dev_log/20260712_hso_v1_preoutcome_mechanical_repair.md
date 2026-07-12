# HSO v1 pre-outcome mechanical repair

The first HSO execution lock (`fdcb6c09485b854c45dd38d61b1256df052211f0c04621f0e9aa2201c4cc7db5`)
was not scientifically executed.  YAML parsed the unquoted resource-map key
`null:` as a null object, so the null-mode process exited before model creation.
The full/text/id processes were then interrupted during the beginning of fold-0
training.  No checkpoint, fold report, score artifact, held-out label access,
metric, or outcome was produced by any mode.

The interruption also exposed a performance-only defect: the compact fit-label
lookup rebuilt its request-to-row dictionary on every row access.  The bounded
repair caches that immutable dictionary once in `CompactFoldLabels.__post_init__`.

The v2 continuation changes only:

1. quote the YAML resource key as `"null"`;
2. cache the existing compact-label index;
3. write a new execution lock path.

The label-free selection, wrong-history donors, user folds, compact fit labels,
model equations, modes, parameter sizes, initialization, candidate sampling,
epochs, optimizer settings, seeds, thresholds, evaluation rules, and all data
boundaries are unchanged.  V1 remains preserved as mechanical provenance; v2
must rehash all source/input files and run all four modes from initialization.
