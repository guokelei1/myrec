# C02 proposal — CHHT

Status: pre-outcome proposal.  No C02 dev metric informed this design.

## One-sentence insight

**Observation:** when strictly-prior history is empirically uneven, a candidate
should change the ranker's function only through the query-compatible
cross-event geometry that it induces, rather than receiving a pooled history
score or a user-wide adapter.

**Architecture consequence:** CHHT composes a bounded skew-symmetric kernel
from `(query, candidate, history event)` interactions and maps it through a
Cayley transform into a candidate-specific low-rank rotation of the compact
Transformer's final FFN computation.

**Falsification:** the hypothesis is too weak if this internal update does not
add positive value on the frozen 4,677 non-repeat history-present requests, if
ordinary static LoRA/output gating/mean-history/history-only conditioning
matches it under the same training envelope, if corrupt evidence preserves the
same functional update, if repeat-present quality falls below the frozen
item-only control, or if any no-history score differs from D2p.

## Evidence-to-design trace

The empirical premise is deliberately bounded:

- the C5-R3 item-only mean NDCG@10 `0.3453755427` is the static waterline;
- 4,677 history-present requests have no exact-repeat candidate and define the
  transfer surface;
- 4,110 requests have no history and require exact D2p degeneration;
- coarse category transfer, generic query attention, same-label oracle
  selection, and identity causality are not established;
- choosing a compact LLM4Rec/Transformer is a project constraint, not a result
  implied by those diagnostics.

CHHT therefore does not assume that semantic transfer works.  It tests whether
a constrained internal functional update can discover any transfer without
erasing the recurrence behavior that is already reliable.

## Architecture

The frozen D2t query representation and item text representation are inputs to
a compact trainable Transformer ranking core; they are not sent to an MLP-only
ranker.  A request Transformer contextualizes the query and ordered history.
A shared two-token query/candidate Transformer produces the activation of each
candidate.  At its final FFN map, and nowhere at the score output, the C02
primitive applies

`Delta W_l(q,c,H) = U_l [R_l(q,c,H) - I] U_l^T W_l`,

where `R_l` is the Cayley rotation of a bounded skew kernel assembled from all
strictly-prior history events.  The score head reads the resulting internal
activation change.  D2p is the legal, non-personalized anchor coordinate; it is
identical for every candidate and control before the internal history update.
No fixed history score is present at inference.

The three named mechanism components are:

1. **Triadic skew-kernel generator** — event contributions depend jointly on
   query, candidate, ordered event representation, event type, recency, and the
   observable exact-recurrence bit.
2. **Cayley HyperAdapter** — turns the skew kernel into a norm-bounded,
   non-diagonal rotation inside one FFN map.
3. **Recurrence preservation constraint** — a training-only listwise
   distillation/hinge constraint keeps repeat-present behavior close to the
   frozen item-only teacher; it is not an inference scorer.

## Why this is not the rejected diagonal dynamic-LoRA proposal

The initial natural design, `U diag(alpha(q,c,H)) V^T`, is reducible to recent
input-sensitive LoRA work (DISeL, Gated LoRA, and Ouroboros) and was discarded
before lock.  CHHT has an off-diagonal skew core, a nonlinear Cayley map, a
single shared rotation subspace, and a per-candidate event-composition rule.
Its diagonal core is identically zero, so the load-bearing operator cannot be
recovered by renaming a rank-wise gate.

## Efficiency and inference lifecycle

History is encoded once per request.  All candidates are processed in a
vectorized batch, and only `r x r` kernels are candidate-specific.  With
`r=8`, the added work is `O(C H r^2 + C r^3 + C d r)` after the compact
Transformer states, not a separate LM pass per candidate.  The update is
ephemeral: it is generated for one `(q,c,H)` forward pass, is never optimized at
test time, and is neither stored nor reused as a per-user model.  Online LLM/API
calls are zero.

## Authorized scope

This track implements unit tests, a train/internal probe, four controls, one
deterministic 1,000-request rescore, and one primary dev evaluator call at seed
`20260708`.  A passing screening can only recommend `advance-to-full-gate`.
Multi-seed training, extra dev calls, secondary datasets, and test access remain
unauthorized.
