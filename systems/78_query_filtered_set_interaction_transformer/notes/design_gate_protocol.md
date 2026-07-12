# C78 data-free design-gate protocol

The C76 surface is reused byte-for-byte.  Three seeds x five equal-parameter
modes train for the same 500 steps and batches.

G0 requires exact no-history/query-mask/repeat, deterministic and candidate
permutation behavior, zero unsupported-candidate gradient, active C-H/H-C
edges, frozen anchors, and exact complete-event permutation invariance for all
set modes.

D1 requires every seed primary to:

- reach clean and shuffled supported accuracy `>=0.75`;
- gain `>=0.10` over base on both;
- keep wrong/query-mask margin retention `<=0.30`;
- keep shuffle margin retention in `[0.98,1.02]`;
- retain repeat/no-history accuracy `>=0.95` and order activity `>=0.05`;
- beat every control by `>=0.02` in
  `min(clean_supported_accuracy, shuffled_supported_accuracy)`.

All training losses must decrease and remain finite.  Failure closes C78 and
leaves only one post-C76 architecture update before the C80 retrospective.
