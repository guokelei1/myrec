# C19 — Oriented-Lag Transformer (OLT)

Status: **design and synthetic falsifier only; no repository data, dev, test,
or qrels access is authorized**.

OLT replaces an unconstrained history value write with a structured temporal
cofactor between two affinity traces: query-to-history and
candidate-to-history.  The diagonal trace supports same-event recurrence; a
skew one-step trace rewards candidate evidence following query-like evidence
and subtracts the reverse direction.

```text
A_j   = affinity(query, history_j)
B_ij  = affinity(candidate_i, history_j)
E_i   = A^T [I + lambda (S - S^T)] B_i
c'_i  = c_i + alpha * tanh(center_candidates(E_i)) * W(c_i)
score = shared_rank_head(c'_i)
```

The evidence law therefore writes inside each candidate token state rather than
adding a handcrafted final-score bonus.  The one-shot synthetic gate compares
the oriented operator with diagonal-only,
forward-induction, symmetric-lag and free signed-lag controls using identical
Transformer parameters and initialization.  A gate failure closes C19 before
any standardized record or real-data training.
