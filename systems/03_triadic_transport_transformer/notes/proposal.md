# C03 Proposal — Candidate-Anchored Cycle-Intersection Transport

Status: proposal content frozen before any C03 dev outcome.

## Insight template

**Observation.** In the frozen KuaiSearch history heuristic, exact candidate
recurrence is reliable, while coarse cross-item transfer is not; therefore a
history event should affect ranking only when query, event, and candidate
representations support the same non-null correspondence.

**Architecture consequence.** Use one candidate-anchored cycle-intersection
partial-transport primitive inside a Transformer ranker.  A shared local
Transformer first contextualizes query, history-event, and candidate states.
Three entropy-regularized transport plans (`q↔h`, `h↔c`, `q↔c`) each include a
learned dustbin.  Only the intersection of their real-real mass updates the
candidate state.  Unmatched mass goes to null.  The history-induced change in
the candidate logit is centered within the request, yielding a signed residual.

**Falsification.** The primitive is too weak or reducible if any of the
following occurs under the frozen probe: (1) it loses the item-only repeat
surface; (2) it has no positive value over D2p on the 4,677 non-repeat,
history-present requests; (3) wrong-user, event-shuffled, query-masked, or
coarse-only history does not move evidence mass toward null and erase the
benefit; (4) a parameter-identical softmax/no-null/no-cycle degeneration shows
the same behavior; or (5) no-history residuals are not exactly zero.

## Pre-result pivot

The initial literal idea—three pairwise Sinkhorn plans plus a generic
cycle-consistency regularizer—was rejected before implementation.  Optimal
Multiple Transport and multi-marginal OT already cover that mathematical core.
The locked operator is narrower:

1. each pairwise plan is a **partial assignment with an explicit learned
   dustbin**, not a balanced all-to-all plan;
2. ranking uses an explicit **candidate-anchored intersection of non-null
   event mass**, rather than adding a generic cycle penalty to a downstream
   scorer;
3. that mass is the only history path into the candidate hidden-state update;
4. exact identity is a protected transport cost atom, never a separate score;
5. unsupported evidence produces zero residual through the null path.

The pivot makes C03 mechanism-distinct enough to test, but it does not justify
a global novelty claim.  The preregistered nearest-neighbor verdict is
`uncertain` pending broader review.

## Scope and claim boundary

The minimal probe may freeze the BGE text encoder while training the local
interaction Transformer and transport parameters.  D2p is retained as the
exact no-history skip score; transport is computed over hidden states, not over
static channel scores.  If the probe survives, a full implementation must
place the D2p-equivalent text/popularity base and transport update in one
training graph.  The present screening cannot validate transferable
personalization, global novelty, or a paper claim.

No category effectiveness, generic query-attention effectiveness, user-identity
causality, or oracle headroom is assumed.
