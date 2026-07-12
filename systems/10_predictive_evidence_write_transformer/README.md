# C10 Predictive Evidence-Write Transformer

C10 changes the late Transformer history-to-candidate residual interface.
History earns a hidden-state write only through candidate-token conditional
predictive information gain computed by the same shared LM under `(q,H)` and
`q-only` masks.  See `notes/proposal.md` for the primitive and
`notes/reduction_and_neighbor_audit.md` for the pre-outcome reduction audit.

Current boundary: structural implementation plus a hash-locked synthetic GPU
falsifier.  The runner cannot read project data, dev, test, or qrels.
Synthetic category/attribute factors are generator-only; the architecture sees
only generic token IDs and evidence masks and has no factor/dataset branches.

```bash
python -m pytest -q systems/10_predictive_evidence_write_transformer/tests
CUDA_VISIBLE_DEVICES=3 python systems/10_predictive_evidence_write_transformer/run_synthetic_gpu_gate.py \
  --output runs/20260711_kuaisearch_c10_synthetic_gpu_gate/result.json
```
