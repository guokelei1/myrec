# C36 Conservative Barycentric Transport Transformer

C36 tests one architectural law inside candidate/history cross-attention: a
candidate-specific history write may redistribute a query-conditioned global
write, but it may neither move the request barycenter nor reverse the global
direction. The only trainable representation change is the same shared rank-16
LM adapter used by every mode.

The primary is compared with four parameter/initialization/fit-matched
reductions: global-only transport, unbounded centered transport, bounded but
uncentered transport, and C35-style relative-only transport. C35-A is exposed
and used only for a label-free formulation audit. C36-A is untouched C35
delayed-B; C36 delayed-B is untouched C35 escrow. Dev and test are forbidden.

Minimal local checks:

```bash
/data/gkl/conda_envs/myrec-c36/bin/python -m pytest \
  systems/36_conservative_barycentric_transport_transformer/tests -q
```

The staged GPU commands are frozen in `notes/train_gate_protocol.md`.
