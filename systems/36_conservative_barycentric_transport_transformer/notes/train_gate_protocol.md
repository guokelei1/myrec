# C36 train-only protocol

Selection SHA-256:
`ad7aab244a21d6eae9b4d5d310cacb8b9b31b27160323733069d76fb81f7a146`.

Reuse C35's fixed 10,000-request fit. Promote untouched C35 delayed-B to
C36-A and untouched C35 escrow to C36 delayed-B. Hash-select a new C36 escrow,
structural repeat/no-history roles, and fresh matched wrong-history donors
without labels. C35-A is excluded.

Train all five modes for seeds `20261021/22/23` with identical rank 16,
initialization per seed, one epoch, complete candidate lists, request order,
optimizer, and listwise/direction losses:

- `conservative_barycentric_transport` (primary);
- `global_tangent_transport`;
- `unbounded_barycentric_transport`;
- `uncentered_trust_transport`;
- `relative_surplus_only`.

The three seeds are bound to physical GPUs 0/1/2. All fifteen fits and A scores
must exist before aggregation. A labels remain closed until every A0
authentication, conservation, activity, matched-control, corruption,
determinism, permutation, and fallback check passes. The same hash folds and
bootstrap are used for D2p and every control. No retry, coefficient/temperature
sweep, C35-A rescue, delayed-B rescue, dev, or test is authorized.

Commands after the proposal lock:

```bash
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /data/gkl/conda_envs/myrec-c36/bin/python \
  systems/36_conservative_barycentric_transport_transformer/train/materialize_g0.py \
  --config systems/36_conservative_barycentric_transport_transformer/configs/train_gate.yaml \
  --device cuda:0
```

After G0, freeze the execution lock, run each registered seed/mode with
`run_train_gate.py --stage seed`, then call `--stage aggregate` once.
