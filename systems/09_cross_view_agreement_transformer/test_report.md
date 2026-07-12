# C09 CPU Structural Test Report

- Date: 2026-07-11
- Device binding: `CUDA_VISIBLE_DEVICES=""` (CPU only)
- Python: 3.13.12
- PyTorch: 2.12.1+cu130
- Repository data/cohorts/labels/qrels read: none
- GPU used: no

Command:

```bash
CUDA_VISIBLE_DEVICES="" \
python -m unittest discover \
  -s systems/09_cross_view_agreement_transformer/tests -v
```

Final rerun result: **14/14 passed in 0.668 seconds**.

Covered contracts:

- strict positive-margin conjunction and all-disagreement blocking;
- hand-computed three-candidate update `(35/19, 2/5, 0)`;
- bit-exact no-history and query-mask fallback;
- candidate permutation equivariance at operator and full-model levels;
- common-mode invariance;
- gradients to both agreeing views, contrast values, token encoder, shared
  mediator attention, and shared rank Transformer;
- off-diagonal counterexample to candidate-local diagonal gating;
- singleton degeneration;
- Q-first candidate blindness and C-first query blindness.

This report validates software/algebra contracts only.  G1 synthetic
learnability and every data/dev gate remain unrun.
