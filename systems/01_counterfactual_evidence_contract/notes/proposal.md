# C01 Proposal — Counterfactual Evidence-Contract Transformer (CECT)

Status: frozen before any C01 train-internal or dev outcome.

## One-sentence insight

**Observation.** A history event should influence a candidate only when a shared
query-candidate-event Transformer can distinguish that event from several
meaning-destroying counterfactual twins at a train-calibrated false-admission
rate; exact candidate recurrence is the protected high-fidelity atom of the same
contract.

**Architecture consequence.** CECT forms a contextual state for every
`(query, candidate, history_event)` inside one Transformer, learns an amortized
event certificate against wrong-user, order-shuffled, query-masked, and
coarse-only twins, and admits a non-exact personalized residual only above a
counterfactual quantile fixed on a train-only calibration slice.

**Falsification.** CECT is false or too weak if the certificate does not separate
true evidence from all twins, if a parameter-matched plain target-attention
Transformer reproduces its internal non-repeat gain, if the certificate is
nearly constant, if the protected repeat behavior degrades, or if no-history
scores are not exactly rank-equivalent to D2p.

## Evidence boundary

The proposal starts from the bounded C5-R3 result, not from an assumed semantic
history effect.  The exact-item component is stable; the tested category-only
component is not significant and dilutes item-only; query attention and
same-query identity causality are not established.  CECT therefore treats
transfer as something that must earn admission.  It does not claim that a
transferable signal already exists.

The binding static waterline is item-only D2s mean NDCG@10
`0.3453755427` (seed `20260708`: `0.3450873589`).  The key non-repeat surface is
4,677 history-present requests; 4,110 no-history requests require exact D2p
fallback.  The candidate-manifest SHA256 is
`94eb667000e0d0f389d0a2a4d4730683b71c129043edbfcf627590376e9c123e`.

## Architecture and information flow

For each candidate, frozen local BGE text states for query, candidate, and
strictly-prior history items are projected into a compact trainable Transformer.
Candidate and history tokens also receive event type, recency position, deepest
category, and candidate-event relation embeddings.  The sequence is

```text
[QUERY] [CANDIDATE] [EVENT_1] ... [EVENT_L]
                    -> shared Transformer -> contextual event states h_i
```

The same encoder processes the observational sequence and every training-only
counterfactual twin.  Each contextual event state produces (i) a certificate
energy and (ii) a signed transfer value.  Counterfactual energies determine a
train-only quantile threshold.  At inference, only the observational sequence is
encoded; an event passes if its amortized energy exceeds that fixed threshold.

Exact recurrence is not scored by a separate expert.  It is a relation atom in
the event contract with guaranteed admission and the frozen C5-R3 recency/event
weight as a lower-bound value.  Non-exact evidence enters the same contracted
history score only through a passed certificate.  The contracted history score
is combined with the legal D2p query/text/popularity anchor; when history is
missing, the personalized term is structurally empty and the output is the
registered D2p score byte-for-byte.

## Named components (three)

1. **Triadic Event Transformer (TET):** the load-bearing joint
   query-candidate-event ranking core.
2. **Counterfactual Quantile Contract (CQC):** shared-twin margin training plus a
   train-only counterfactual false-admission threshold; exact recurrence is its
   protected atom.
3. **Contracted Residual Readout (CRR):** aggregates only admitted event values
   and produces the candidate ranking logit anchored to D2p.

No router, per-query-type branch, online LLM call, online wrong-user history, or
dataset identifier is used.

## Why this is a minimal probe

The BGE tower and item states are reused read-only.  Only the compact interaction
Transformer, relation embeddings, certificate/value heads, and readout are
trained.  The probe answers the C01 falsifier on one main-track seed and one
shared dev evaluation.  It is not the later full implementation, and it stops
after its final report whether the screening passes or fails.
