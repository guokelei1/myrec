# C22 — Evidence-Filtration Transformer

Status: **design formulation and synthetic falsifier only; no real-data, dev,
full-training or paper claim is authorized**.

C21 closes short directed path geometry over the existing frozen D2 states.  It
does not close architectures that change how evidence survives through the LM.
C22 therefore modifies the residual stream itself.  Candidate tokens carry an
ordered representation filtration:

```text
query/candidate anchor  ⊂  exact-recurrence evidence  ⊂  speculative transfer
```

Every attention, FFN and normalization operation preserves that filtration.
Later, less reliable coordinates may read earlier reliable coordinates, but
they cannot write back into them.  Exact recurrence is injected into the middle
quotient and remains causally available at every layer; semantic history enters
only the final quotient.  The model is still one Transformer ranker, not a
router over fixed scorers.

The underlying block-triangular and prefix-normalization devices have close
architecture neighbours.  C22 does not claim that generic nested Transformers
are new.  Its testable claim is narrower: reliability-ordered one-way residual
coupling is a useful inductive bias for preserving recurrence while learning a
separate non-repeat transfer path.  It must beat parameter-matched dense,
parallel-stream and final-projection controls before any real-data gate.
