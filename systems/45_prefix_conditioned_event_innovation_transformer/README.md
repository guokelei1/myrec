# C45 — Prefix-Conditioned Event Innovation Transformer

C45 tests one representation hypothesis: a behavior event should enter the
ranking Transformer through the state change it causes relative to a NULL
event under the exact same strictly-prior prefix, rather than through its raw
item embedding or a pooled history state.

The current authorization is a data-free synthetic design gate only. The
candidate has no repository-data reader, evaluator integration, dev/test
access, or full-training authorization.

## Minimal commands

```bash
/data/gkl/conda_envs/myrec-c37/bin/python -m pytest -q \
  systems/45_prefix_conditioned_event_innovation_transformer/tests

/data/gkl/conda_envs/myrec-c37/bin/python \
  systems/45_prefix_conditioned_event_innovation_transformer/probe/audit_generator.py \
  --config systems/45_prefix_conditioned_event_innovation_transformer/configs/design_gate.yaml
```

Formal GPU commands are recorded in `notes/design_gate_protocol.md`. They are
valid only after `notes/proposal_lock.json` exists and verifies.
