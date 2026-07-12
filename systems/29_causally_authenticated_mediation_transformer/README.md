# C29 Causally Authenticated Mediation Transformer

C29 tests one new primitive: a current history event may enter the pretrained
Transformer ranking path only if that item occurs in the same user's strictly
earlier history snapshot.  The persistent user memory cannot score candidates;
it only creates an event-level attention-admissibility mask.  The shared BGE
Transformer then scores factual and structurally null history streams, and only
their candidate-specific difference may modify frozen D2p.

This is not a user-ID scorer, fixed score router, semantic-category rule, or
soft-token novelty claim.  It is a falsifiable causal mask on the LM's internal
history-to-candidate information path.  Empty authentication, no history, and
query absence reduce exactly to D2p; exact candidate recurrence retains the
registered item-only fallback.

The proposal follows C28's terminal utility failure and separate post-terminal
audits.  Those audits found strong true/wrong authentication separation but no
stable candidate direction from small random token Transformers, fixed BGE
geometry, attributes, causal co-occurrence, semantic codes, event type, or
query recurrence.  A full pretrained mediation probe was the sole path with
positive average gain, but 3,000 fit requests were seed-unstable.  C29 freezes
10,000 unsliced train requests for the 23,954,432-parameter LM and validates on C28
escrow, which has never been feature-materialized, scored, or labeled.

No dev/test evaluator call or qrels access is authorized.  The earlier
interactive qrels schema-inspection incident is separately quarantined in
`doc/dev_log/20260711_qrels_schema_inspection_incident.md`; no C29 code may read
standardized qrels.
