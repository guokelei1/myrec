# C26 proposal — query-pivot token bridge

Status: pre-outcome design; no internal-A/delayed-B label opened.

## Mechanism

A shared compact Transformer contextualizes BGE WordPiece embeddings for query,
candidate title and every history title.  For each query token `q_k`, soft
late-interaction independently retrieves a candidate value `c*_{ik}` and a
history-event value `h*_{jk}`.  The only personalized event write is

```text
b_ijk = sum_k softmax_k(match(q_k,c_i)+match(q_k,h_j))
              FFN[(c*_{ik}-q_k) * (h*_{jk}-q_k)].
```

Thus candidate and history must support the same query-token pivot.  A second
Transformer aggregates `[READ,b_i1,...,b_iH]`.  It sees no D2p, recurrence
scalar or raw pooled candidate bypass.  Candidate-centred corrections operate
only on strict non-repeat history-present requests; repeat requests return
registered item-only and no-history/query-absent requests return D2p exactly.

## Controls

All modes execute the same token encoders, alignments, bridge FFN and history
Transformer with identical parameters and schedules:

- `token_bridge` — primary same-query-token agreement;
- `generic_token_triadic` — additive query/candidate/history matched values;
- `candidate_late` — query-candidate late interaction without history value;
- `pooled_history` — token-encoded pooled query/candidate/history event.

The primary must beat `candidate_late`; otherwise any gain is finer query-item
matching, not personalization.  It must beat the generic triadic and pooled
controls to pay architectural rent, and its gain must disappear with frozen
wrong histories on two sequentially opened train-internal roles.

## Boundary

C26 is not a claim that ColBERT, fine-grained interaction or triple attention
is new.  The potential contribution is only the shared-query-token bridge as a
restricted history-write law.  A positive A0/A1/A2 would authorize a deeper
novelty and full-LM review; a failure closes this token-bridge primitive.
