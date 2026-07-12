# C47 — Posterior-Supported Ridge Transformer

Status: **pre-outcome formulation and CPU operator prototype**.

C47 tests one narrow architecture hypothesis: a history KRR write should be
contracted by the candidate's support under the *same* posterior geometry
before it enters a candidate-specific query token. The proposal does not claim
that KRR token mixing is new; Cubit is the binding nearest control.

No C47 cohort, label, checkpoint, dev score, test score, or qrels has been
opened. The already-open C46/C42 diagnostics are formulation evidence only.

Run the structural tests from the repository root:

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest -q \
  systems/47_posterior_supported_ridge_transformer/tests
```
