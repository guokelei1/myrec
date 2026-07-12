# C11 Eventwise Predictive-Write Transformer

C11 keeps candidate-token × history-event predictive information gain intact
until a late Transformer, repairing C10's global history pooling bottleneck.
The base path is history-blind; history writes are bounded and candidate
zero-sum; exact identity is one monotone coordinate inside the same ranking
head.

Current status: **pre-lock review only**.  The runner deliberately refuses to
run because no approved `frozen_manifest.json` exists.  Do not create that file
or use GPU 3 until independent review approves the architecture, construct
audit, controls, seeds, thresholds, and hashes.

```bash
python -m pytest -q systems/11_eventwise_predictive_write_transformer/tests
python systems/11_eventwise_predictive_write_transformer/run_synthetic_gpu_gate.py \
  --output runs/20260711_kuaisearch_c11_synthetic_gpu_gate/result.json
# Expected now: refusal because review is pending.
```
