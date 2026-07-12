# C07: Pairwise Dead-zone Signed-Kernel Transformer

Status: **architecture proposal and CPU structural prototype complete; semantic
synthetic gate not run; real-data/GPU training not authorized**.

C07 changes the normalization inside a Transformer history-attention block.
For each history event, candidates enter pairwise margin contests.  An odd
soft-threshold removes weak contests, and the surviving margins are normalized
as bounded signed mass whose sum across candidates is zero.  The resulting
history update can change candidate margins, cannot waste capacity on a common
candidate shift, and is exactly zero on an open dead zone.

The proposal deliberately does **not** claim that signed attention, sparse
attention, candidate-conditioned history attention, or abstention is new.  The
primary-source audit shows prior art for every one of those ingredients.  The
only C07-specific hypothesis is their narrow composition as pairwise,
candidate-axis, dead-zone competition inside the ranking Transformer.  The
pre-outcome gate requires it to beat a scalar-gated centered control; otherwise
the direction stops as a generic gate/normalizer variant.

## Files

- `proposal.md` — observation, equations, information flow, reduction audit,
  and risks.
- `mechanism_fingerprint.md` — minimal algebraic fingerprint and collision
  tests.
- `nearest_neighbor_audit.md` — primary-source nearest-neighbor audit.
- `pre_outcome_gate.md` — frozen CPU-only synthetic falsifier; not executed.
- `src/signed_kernel_transformer.py` — minimal end-to-end Transformer and
  signed operator.
- `tests/test_signed_kernel_transformer.py` — hand-computed and structural
  tests.
- `TEST_RESULTS.md` — local CPU verification record.

## Local verification

```bash
python -m pytest -q \
  systems/07_signed_kernel_transformer/tests/test_signed_kernel_transformer.py
```

The module has no filesystem, dataset, evaluator, qrels, or label API.  It
accepts embedded query/history/candidate tensors plus availability and exact-ID
match tensors.  The prototype was exercised on CPU only.
