# C08 — Reversible Write–Probe–Undo Transformer

Status: **proposal and pre-outcome gate locked; structural CPU gate passed;
learned synthetic G1 failed 0/3 seeds; C08 stopped; real-data probe forbidden**
(2026-07-11).

The candidate has one primitive: an evidence-conditioned, volume-preserving
coupling update is written by each history event; a query/candidate probe is
then applied; history and probe are undone in reverse order; only the resulting
closed-loop displacement is injected into a candidate token between Transformer
blocks. Empty history is exactly the identity.

Files:

- `PROPOSAL.md` — locked observation, equations, information flow, and scope;
- `NEAREST_NEIGHBOR_AUDIT.md` — literature and reducibility audit;
- `FINGERPRINT.json` — collision-resistant mechanism fingerprint;
- `GATE.md` — frozen structural, synthetic, and possible dev gates;
- `reversible_memory.py` — minimal end-to-end CPU Transformer prototype;
- `test_reversible_memory.py` — structural and gradient contracts;
- `SYNTHETIC_REPORT.md` — executed structural results and go/stop decision.
- `G1_PROTOCOL_AMENDMENT.md` — fully specified learned-synthetic protocol;
- `G1_EXECUTION_LOCK.json` — source/test/config hash lock;
- `G1_OUTCOME.{json,md}` — concise locked G1 failure and terminal decision.

Run the only currently authorized check from this directory:

```bash
CUDA_VISIBLE_DEVICES="" pytest -q test_reversible_memory.py
```

The code has no dataset reader, score exporter, evaluator, qrels access, or GPU
placement. G1 has now failed under its locked protocol, so this candidate must
remain a negative design result and must not be extended to repository data.
