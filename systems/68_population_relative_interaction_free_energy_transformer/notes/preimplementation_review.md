# C68 preimplementation review

## Entry decision

`authorize_data_free_G0_only`.

The proposal modifies the Transformer ranking computation rather than adding a
dataset rule or output router.  Query/candidate-only and separable
query-history paths cancel algebraically.  The same masks and operator apply to
both target datasets.  A population reference is a new information edge, so a
real implementation is not authorized until the data-free gate shows that the
finite-temperature interaction pays rent over its exact reductions.

## Rejected drafts before lock

- empirical Fisher energy is `sum_e <g_e,g_qc>^2`, a second-order kernel over
  gradient features;
- inverse-Fisher/natural-gradient reads reduce to ridge/KRR in that feature
  space;
- a global-minus-user pooled vector is the RESUS/mean-residual boundary;
- semantic/collaborative agreement is directly crowded by CLLM4Rec,
  SeLLa-Rec, IDIOMoE, CCLRec, and C09-style agreement;
- a learned reference token alone is a C04/C65 NULL subtraction.

None is registered as C68.

## G0 scope

Authorized: source, tests, proposal and execution locks, three fixed synthetic
GPU seeds, ignored artifacts/checkpoints, and one concise promoted gate report.

Forbidden: any standardized/raw repository record, train/dev/test label, qrel,
existing feature artifact, dev evaluator, real checkpoint, generator rescue,
or post-outcome threshold change.
