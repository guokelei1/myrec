# C18 — Evidence-Constrained Order Transformer (ECOT)

Status: **design and synthetic pre-outcome falsifier only; no repository-data,
dev, test, or qrels access is authorized**.

ECOT changes the final ranking operator of an LLM4Rec-style Transformer.  A
semantic history path may propose candidate-relative transfer, but it cannot
directly overwrite reliable exact recurrence.  The proposal is projected onto
a request-specific order cone whose inequalities are supplied by exact
candidate/history identity evidence inside the same ranking core.

```text
query/candidate Transformer -> base logits b
semantic history Transformer -> bounded proposal y
exact recurrence relation -> order cone K
final logits = EuclideanProjection_K(y)
```

The current authorization is limited to the locked, in-memory synthetic probe
defined in `notes/gate_protocol.md`.  A failure closes C18 before standardized
records or GPU real-data training.  A pass authorizes only a separately frozen
train-internal real gate; it does not authorize dev or test.

Candidate-local layout:

- `model/` — minimal Transformer and order-projection operator;
- `train/` — synthetic generator, trainer, audit and one-shot runner;
- `tests/` — structural and protocol contracts;
- `configs/synthetic_gate.yaml` — all learned-probe constants;
- `notes/` — proposal, mechanism fingerprint, neighbours and frozen gate.

Raw outcomes belong under ignored `artifacts/c18_evidence_constrained_order_transformer/`.
