# Current direction decision

Status: active scope note, 2026-07-14.

The project remains in PPS, but the scientific question has changed. The
active question is the bounded hypothesis in doc 34:

> ordinary full-token rerankers can respond to history while failing to turn
> that response into a stable, query-conditioned, candidate-relative ranking
> direction on an eligible search population.

This is unproven. It must be tested and may be falsified.

## Tracks

- **Main:** KuaiSearch Full, if source/collision/power audit admits it.
- **Replication:** KuaiSAR Full if its search slate and temporal history can
  be reconstructed; otherwise the pre-registered JDsearch fallback.
- **Non-binding stress test:** Amazon-C4 plus its history companion. It cannot
  carry the main natural-search claim.

No C81, C80 rescue, old R0 round, or new architecture implementation is part
of this decision.
